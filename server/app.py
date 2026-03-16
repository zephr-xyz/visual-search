#!/usr/bin/env python3
"""
Visual Search Server — geographic prevalence search over street-level imagery.

Two search modes:
  1. Keyword (TF-IDF): stemmed token lookup in pre-aggregated caption index
  2. Semantic (FAISS):  embed query → top-K nearest images → aggregate by geography

Endpoints:
  GET  /                       - Choropleth map UI
  POST /api/search             - Unified search (mode=keyword|semantic)
  GET  /api/geojson/{level}    - GeoJSON boundaries (state|msa)
  GET  /api/stats              - Dataset statistics

Usage:
  uvicorn server.app:app --host 0.0.0.0 --port 8080
  uvicorn server.app:app --reload  # development
"""

import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="Visual Search", version="1.0")

# ── Global state loaded on startup ───────────────────────────────

keyword_indexes = {}   # {level: {"index": {term: {geo: count}}, "geo_totals": {...}}}
tfidf_models = {}      # {level: {"vectorizer", "matrix", "geo_names"}}
faiss_index = None
faiss_geo_lookup = None  # DataFrame: position-aligned state_name, cbsa_name
geo_stats = {}

# Stemmer (must match pipeline/03_build_tfidf_index.py)
import re

import nltk
from nltk.stem.snowball import SnowballStemmer

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(
    {
        "image", "photograph", "photo", "panoramic", "panorama",
        "equirectangular", "captured", "camera", "degree", "360",
        "mounted", "vehicle", "roof",
    }
)
_stemmer = SnowballStemmer("english")


def stem_tokenize(text):
    tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return [_stemmer.stem(t) for t in tokens if t not in STOP_WORDS]


# ── Embedding model (lazy-loaded for semantic search) ────────────

_embed_model = None


def get_embedder():
    """Lazy-load EmbeddingGemma for query embedding.

    Falls back to sentence-transformers if EmbeddingGemma not available.
    """
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    # Try sentence-transformers first (easier to install)
    try:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("google/embeddinggemma-300m")
        print("Loaded EmbeddingGemma via sentence-transformers")
        return _embed_model
    except Exception:
        pass

    # Fallback: TFLite model (matching pipeline/generate_visual_embeddings.py)
    try:
        import sentencepiece as spm

        model_dir = Path("/private/tmp/zephrpoint/here-poi-pipeline/models")
        tflite_path = model_dir / "embeddinggemma-300M-seq256.tflite"
        sp_path = model_dir / "embeddinggemma-tokenizer.spm"

        if tflite_path.exists():
            try:
                from ai_edge_litert import interpreter as tfl_interp
                interp = tfl_interp.Interpreter(model_path=str(tflite_path))
            except ImportError:
                import tensorflow as tf
                interp = tf.lite.Interpreter(model_path=str(tflite_path))

            interp.allocate_tensors()
            sp = spm.SentencePieceProcessor()
            sp.Load(str(sp_path))
            seq_len = interp.get_input_details()[0]["shape"][1]

            class TFLiteEmbedder:
                def __init__(self, interp, sp, seq_len):
                    self.interp = interp
                    self.sp = sp
                    self.seq_len = seq_len

                def encode(self, texts, **kwargs):
                    results = []
                    for text in texts:
                        prefixed = f"task: sentence similarity | query: {text}"
                        ids = [2] + self.sp.EncodeAsIds(prefixed) + [1]
                        if len(ids) > self.seq_len:
                            ids = ids[: self.seq_len]
                        else:
                            ids = ids + [0] * (self.seq_len - len(ids))
                        input_ids = np.array([ids], dtype=np.int32)
                        self.interp.set_tensor(
                            self.interp.get_input_details()[0]["index"], input_ids
                        )
                        self.interp.invoke()
                        emb = self.interp.get_tensor(
                            self.interp.get_output_details()[0]["index"]
                        )[0]
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                        results.append(emb.astype(np.float32))
                    return np.stack(results)

            _embed_model = TFLiteEmbedder(interp, sp, seq_len)
            print("Loaded EmbeddingGemma via TFLite")
            return _embed_model
    except Exception as e:
        print(f"TFLite fallback failed: {e}")

    raise RuntimeError(
        "No embedding model available. Install sentence-transformers or "
        "ensure TFLite model is at /private/tmp/zephrpoint/here-poi-pipeline/models/"
    )


def embed_query(text):
    """Embed a query string to a 768-dim normalized vector."""
    model = get_embedder()
    vec = model.encode([text])[0].astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ── Startup ──────────────────────────────────────────────────────

@app.on_event("startup")
def load_indexes():
    global keyword_indexes, tfidf_models, faiss_index, faiss_geo_lookup, geo_stats

    # Keyword indexes
    for level in ("state", "msa"):
        kw_path = DATA_DIR / f"keyword_index_{level}.pkl"
        if kw_path.exists():
            with open(kw_path, "rb") as f:
                keyword_indexes[level] = pickle.load(f)
            n_terms = len(keyword_indexes[level]["index"])
            n_geos = len(keyword_indexes[level]["geo_totals"])
            print(f"Loaded keyword index ({level}): {n_terms:,} terms, {n_geos} geos")

    # TF-IDF models
    for level in ("state", "msa"):
        tfidf_path = DATA_DIR / f"tfidf_model_{level}.pkl"
        if tfidf_path.exists():
            with open(tfidf_path, "rb") as f:
                tfidf_models[level] = pickle.load(f)
            shape = tfidf_models[level]["matrix"].shape
            print(f"Loaded TF-IDF model ({level}): {shape}")

    # FAISS index
    faiss_path = DATA_DIR / "faiss_index.ivfpq"
    if faiss_path.exists():
        faiss_index = faiss.read_index(str(faiss_path))
        faiss_index.nprobe = 64
        print(f"Loaded FAISS index: {faiss_index.ntotal:,} vectors")

        geo_path = DATA_DIR / "faiss_geo.parquet"
        if geo_path.exists():
            faiss_geo_lookup = pd.read_parquet(geo_path)
            print(f"Loaded FAISS geo lookup: {len(faiss_geo_lookup):,} rows")

    # Geo stats
    stats_path = DATA_DIR / "geography_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            geo_stats = json.load(f)

    print("Server ready.")


# ── API Models ───────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    mode: str = "keyword"       # "keyword" or "semantic"
    geo_level: str = "state"    # "state" or "msa"
    top_k: int = 100_000        # FAISS top-K for semantic mode


class SearchResult(BaseModel):
    query: str
    mode: str
    geo_level: str
    elapsed_ms: float
    results: list  # [{geo_name, prevalence, count, total_images}, ...]


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (TEMPLATE_DIR / "index.html").read_text()


@app.post("/api/search", response_model=SearchResult)
def search(req: SearchRequest):
    t0 = time.time()

    if req.mode == "keyword":
        results = keyword_search(req.query, req.geo_level)
    elif req.mode == "semantic":
        results = semantic_search(req.query, req.geo_level, req.top_k)
    else:
        raise HTTPException(400, f"Unknown mode: {req.mode}")

    elapsed_ms = (time.time() - t0) * 1000

    return SearchResult(
        query=req.query,
        mode=req.mode,
        geo_level=req.geo_level,
        elapsed_ms=round(elapsed_ms, 1),
        results=results,
    )


@app.get("/api/geojson/{level}")
def geojson(level: str):
    """Serve GeoJSON boundaries for choropleth rendering."""
    geo_path = DATA_DIR / "shapefiles" / f"{level}_boundaries.geojson"
    if not geo_path.exists():
        raise HTTPException(404, f"GeoJSON not found for level: {level}")
    return FileResponse(geo_path, media_type="application/geo+json")


@app.get("/api/stats")
def stats():
    return {
        "keyword_indexes": {k: len(v["geo_totals"]) for k, v in keyword_indexes.items()},
        "faiss_vectors": faiss_index.ntotal if faiss_index else 0,
        "geo_stats": geo_stats,
    }


# ── Search implementations ───────────────────────────────────────

def keyword_search(query, geo_level):
    """Stemmed keyword prevalence search."""
    level = geo_level
    if level not in keyword_indexes:
        raise HTTPException(400, f"No keyword index for level: {level}")

    kw = keyword_indexes[level]
    index = kw["index"]
    geo_totals = kw["geo_totals"]

    tokens = stem_tokenize(query)
    if not tokens:
        return []

    # Sum image counts per geography across all query tokens
    geo_counts = {}
    for token in tokens:
        if token in index:
            for geo, count in index[token].items():
                geo_counts[geo] = geo_counts.get(geo, 0) + count

    # Build results with prevalence
    results = []
    for geo, count in geo_counts.items():
        total = geo_totals.get(geo, 1)
        results.append(
            {
                "geo_name": geo,
                "count": count,
                "total_images": total,
                "prevalence": round(count / total, 6),
            }
        )

    results.sort(key=lambda r: r["prevalence"], reverse=True)
    return results


def semantic_search(query, geo_level, top_k):
    """FAISS semantic search with geographic aggregation."""
    if faiss_index is None:
        raise HTTPException(503, "FAISS index not loaded")
    if faiss_geo_lookup is None:
        raise HTTPException(503, "FAISS geo lookup not loaded")

    geo_col = "state_name" if geo_level == "state" else "cbsa_name"

    # Embed query
    query_vec = embed_query(query).reshape(1, -1)

    # Search FAISS
    actual_k = min(top_k, faiss_index.ntotal)
    distances, indices = faiss_index.search(query_vec, actual_k)

    # Aggregate by geography
    valid = indices[0] >= 0
    result_indices = indices[0][valid]

    geo_names = faiss_geo_lookup[geo_col].values
    geo_totals_col = faiss_geo_lookup[geo_col].value_counts().to_dict()

    # Count results per geography
    geo_counts = {}
    for idx in result_indices:
        geo = geo_names[idx]
        if pd.notna(geo):
            geo_counts[geo] = geo_counts.get(geo, 0) + 1

    results = []
    for geo, count in geo_counts.items():
        total = geo_totals_col.get(geo, 1)
        results.append(
            {
                "geo_name": geo,
                "count": count,
                "total_images": total,
                "prevalence": round(count / total, 6),
            }
        )

    results.sort(key=lambda r: r["prevalence"], reverse=True)
    return results
