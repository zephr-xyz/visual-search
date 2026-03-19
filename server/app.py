#!/usr/bin/env python3
"""
Visual Search Server — geographic prevalence search over street-level imagery.

Two search modes:
  1. Keyword (TF-IDF): stemmed token lookup in pre-aggregated caption index
  2. Semantic (FAISS):  embed query → top-K nearest images → aggregate by geography

Supports drill-down from state/MSA → z8/z10 grid → z12/z14 → individual images.

Endpoints:
  GET  /                       - Choropleth map UI
  POST /api/search             - Unified search (mode=keyword|semantic)
  POST /api/search/grid        - Grid-level search (z8/z10 pre-computed, z12/z14 on-demand)
  POST /api/search/images      - Image-level search within a bounding box
  GET  /api/geojson/{level}    - GeoJSON boundaries (state|msa)
  GET  /api/stats              - Dataset statistics

Usage:
  uvicorn server.app:app --host 0.0.0.0 --port 8080
  uvicorn server.app:app --reload  # development
"""

import io
import json
import logging
import math
import os
import pickle
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import boto3
import faiss
import numpy as np
import pandas as pd
from botocore.config import Config
from scipy import sparse
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger("visual-search")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="Visual Search", version="1.0")

# ── Global state loaded on startup ───────────────────────────────

keyword_indexes = {}   # {level: {"index": {term: {geo: count}}, "geo_totals": {...}}}
tfidf_models = {}      # {level: {"vocabulary", "matrix", "geo_names"}}
faiss_index = None
faiss_geo_lookup = None  # DataFrame: position-aligned state_name, cbsa_name
geo_stats = {}
caption_tile_dir = None  # Path to data/captions_by_tile/ for on-demand z12/z14 queries

# Mapillary thumbnail proxy
MAPILLARY_CLIENT_TOKEN = os.environ.get("MAPILLARY_CLIENT_TOKEN", "")
_thumb_cache = OrderedDict()  # image_id -> thumb_url, max 2000 entries
_THUMB_CACHE_MAX = 2000
_http_client = None  # lazy-init httpx.AsyncClient

# ── DINOv2 visual re-ranking ────────────────────────────────────
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").lower() in ("true", "1", "yes")
RERANK_S3_BUCKET = "zephr-mapillary-computed-data"
RERANK_S3_WORKERS = 50
RERANK_TEXT_CAP = 500       # max text matches to re-rank
RERANK_MIN_EMBEDDINGS = 10  # skip re-ranking below this

_emb_cache = OrderedDict()          # image_id -> np.ndarray (1024,), max 10K
_EMB_CACHE_MAX = 10_000
_rerank_result_cache = OrderedDict() # (query, tile_key) -> list of image dicts, max 50
_RERANK_RESULT_CACHE_MAX = 50
_rerank_s3_local = threading.local()
_rerank_s3_config = None

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


def raw_tokenize(text):
    """Return non-stopword tokens without stemming (for regex caption matching)."""
    tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


# ── Tile utilities ────────────────────────────────────────────────

def lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to Web Mercator tile (x, y) at given zoom level."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_to_bbox(x, y, zoom):
    """Convert tile (x, y, zoom) to (west, south, east, north) bounding box."""
    n = 2 ** zoom

    def tile_lat(ty):
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
        return math.degrees(lat_rad)

    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = tile_lat(y)
    south = tile_lat(y + 1)
    return west, south, east, north


def tiles_in_bbox(west, south, east, north, zoom):
    """Return all tile keys at given zoom that overlap a bounding box."""
    x_min, y_max = lat_lon_to_tile(south, west, zoom)
    x_max, y_min = lat_lon_to_tile(north, east, zoom)
    # Handle potential wrap-around
    if x_min > x_max:
        x_max, x_min = x_min, x_max
    if y_min > y_max:
        y_max, y_min = y_min, y_max
    tiles = []
    for tx in range(x_min, x_max + 1):
        for ty in range(y_min, y_max + 1):
            tiles.append(f"z{zoom}/{tx}/{ty}")
    return tiles


_tile_cache = {}  # z10_key -> DataFrame (cached for fast drill-down)
_tile_cache_max = 20  # Keep up to 20 z10 tiles in memory (~200MB)


def load_tile_partition(z10_key, columns=None):
    """Load a caption partition file for a z10 tile with LRU caching."""
    if caption_tile_dir is None:
        return None

    # Check cache (full DataFrame, then filter columns if needed)
    if z10_key in _tile_cache:
        df = _tile_cache[z10_key]
        if columns:
            available = [c for c in columns if c in df.columns]
            return df[available].copy()
        return df

    safe_name = z10_key.replace("/", "_")
    path = caption_tile_dir / f"{safe_name}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)

    # Evict oldest if cache is full
    if len(_tile_cache) >= _tile_cache_max:
        oldest = next(iter(_tile_cache))
        del _tile_cache[oldest]
    _tile_cache[z10_key] = df

    if columns:
        available = [c for c in columns if c in df.columns]
        return df[available].copy()
    return df


# ── DINOv2 S3 fetching (thread-local clients, same pattern as pipeline) ──

def _init_rerank_s3():
    """Initialize S3 config for re-ranking workers. Call once before parallel fetches."""
    global _rerank_s3_config
    if _rerank_s3_config is None:
        _rerank_s3_config = Config(
            max_pool_connections=RERANK_S3_WORKERS + 10,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )


def _get_rerank_s3():
    """Get thread-local S3 client for re-ranking."""
    if not hasattr(_rerank_s3_local, "s3"):
        _init_rerank_s3()
        _rerank_s3_local.s3 = boto3.client(
            "s3", region_name="us-east-2", config=_rerank_s3_config,
        )
    return _rerank_s3_local.s3


def _fetch_cls_embedding(sequence_id, image_id):
    """Fetch DINOv2 CLS embedding for one image from S3.

    Returns (image_id, embedding) or (image_id, None) on failure.
    Caches successful fetches in _emb_cache.
    """
    # Check cache first
    if image_id in _emb_cache:
        _emb_cache.move_to_end(image_id)
        return (image_id, _emb_cache[image_id])

    try:
        s3 = _get_rerank_s3()
        key = f"{sequence_id}/{image_id}/cls_embedding.npz"
        obj = s3.get_object(Bucket=RERANK_S3_BUCKET, Key=key)
        data = np.load(io.BytesIO(obj["Body"].read()))
        emb = data["data"].astype(np.float32)

        # L2-normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Cache
        _emb_cache[image_id] = emb
        if len(_emb_cache) > _EMB_CACHE_MAX:
            _emb_cache.popitem(last=False)

        return (image_id, emb)
    except Exception:
        return (image_id, None)


def _fetch_embeddings_parallel(image_rows):
    """Fetch DINOv2 embeddings for a list of image dicts in parallel.

    Args:
        image_rows: list of dicts with at least 'image_id' and 'sequence_id'

    Returns:
        dict of {image_id: np.ndarray} for successfully fetched embeddings
    """
    _init_rerank_s3()

    # Split into cached and uncached
    results = {}
    to_fetch = []
    for row in image_rows:
        img_id = row["image_id"]
        if img_id in _emb_cache:
            _emb_cache.move_to_end(img_id)
            results[img_id] = _emb_cache[img_id]
        else:
            to_fetch.append(row)

    if not to_fetch:
        return results

    with ThreadPoolExecutor(max_workers=RERANK_S3_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_cls_embedding, row["sequence_id"], row["image_id"]): row["image_id"]
            for row in to_fetch
        }
        for fut in as_completed(futures):
            try:
                img_id, emb = fut.result()
                if emb is not None:
                    results[img_id] = emb
            except Exception:
                pass

    return results


# ── DINOv2 re-ranking algorithm ──────────────────────────────────

def _rerank_by_visual_consensus(image_dicts, embeddings):
    """Re-rank images by visual consensus using iterative centroid refinement.

    True positives for a visual query cluster in DINOv2 embedding space (real graffiti
    looks similar). False positives (hallucinated captions) scatter randomly. By computing
    a consensus centroid and scoring similarity to it, true positives rise to the top.

    Args:
        image_dicts: list of image dicts (each must have 'image_id')
        embeddings: dict of {image_id: np.ndarray (1024,)}

    Returns:
        Re-ordered list of image dicts with 'visual_score' added.
    """
    # Separate images with/without embeddings
    with_emb = []
    without_emb = []
    for img in image_dicts:
        if img["image_id"] in embeddings:
            with_emb.append(img)
        else:
            without_emb.append(img)

    if len(with_emb) < RERANK_MIN_EMBEDDINGS:
        # Not enough embeddings to form meaningful consensus
        for img in image_dicts:
            img["visual_score"] = 0.0
        return image_dicts

    # Stack embeddings into matrix (N, 1024) — already L2-normalized
    ids = [img["image_id"] for img in with_emb]
    matrix = np.stack([embeddings[img_id] for img_id in ids])  # (N, 1024)

    # Iterative centroid refinement
    centroid = matrix.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    for _ in range(3):
        # Cosine similarity = dot product (vectors are L2-normalized)
        sims = matrix @ centroid
        # Keep top 60%, minimum 10
        keep_n = max(RERANK_MIN_EMBEDDINGS, int(len(sims) * 0.6))
        top_indices = np.argsort(sims)[-keep_n:]
        centroid = matrix[top_indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

    # Final scoring: cosine similarity of ALL embedded images to refined centroid
    final_scores = matrix @ centroid

    for i, img in enumerate(with_emb):
        img["visual_score"] = round(float(final_scores[i]), 4)

    # Sort by visual score descending
    with_emb.sort(key=lambda x: x["visual_score"], reverse=True)

    # Append images without embeddings at end
    for img in without_emb:
        img["visual_score"] = 0.0

    return with_emb + without_emb


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
    global caption_tile_dir

    # Keyword indexes (state, msa, z8, z10)
    for level in ("state", "msa", "z8", "z10"):
        kw_path = DATA_DIR / f"keyword_index_{level}.pkl"
        if kw_path.exists():
            with open(kw_path, "rb") as f:
                keyword_indexes[level] = pickle.load(f)
            n_terms = len(keyword_indexes[level]["index"])
            n_geos = len(keyword_indexes[level]["geo_totals"])
            print(f"Loaded keyword index ({level}): {n_terms:,} terms, {n_geos} geos")

    # TF-IDF models
    for level in ("state", "msa", "z8", "z10"):
        tfidf_path = DATA_DIR / f"tfidf_model_{level}.pkl"
        if tfidf_path.exists():
            with open(tfidf_path, "rb") as f:
                tfidf_models[level] = pickle.load(f)
            shape = tfidf_models[level]["matrix"].shape
            print(f"Loaded TF-IDF model ({level}): {shape}")

    # Caption tile partitions for on-demand z12/z14 queries
    tile_dir = DATA_DIR / "captions_by_tile"
    if tile_dir.exists():
        n_parts = len(list(tile_dir.glob("*.parquet")))
        caption_tile_dir = tile_dir
        print(f"Caption tile partitions: {n_parts:,} z10 tiles")

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
    mode: str = "combined"      # "keyword", "combined", or "semantic"
    geo_level: str = "state"    # "state" or "msa"
    top_k: int = 100_000        # FAISS top-K for semantic mode


class SearchResult(BaseModel):
    query: str
    mode: str
    geo_level: str
    elapsed_ms: float
    results: list  # [{geo_name, prevalence, count, total_images, signed_chi, tfidf_score, score}, ...]


class GridSearchRequest(BaseModel):
    query: str
    zoom: int = 10           # 8 or 10 (pre-computed), 12 or 14 (on-demand)
    bbox: list = None        # [west, south, east, north] — required for z12/z14
    parent_tile: str = None  # e.g. "z10/512/345" — alternative to bbox for drill-down


class GridSearchResult(BaseModel):
    query: str
    zoom: int
    elapsed_ms: float
    tiles: list  # [{tile_key, x, y, west, south, east, north, count, total, prevalence}, ...]


class ImageSearchRequest(BaseModel):
    query: str
    bbox: list = None        # [west, south, east, north]
    tile_key: str = None     # e.g. "z14/8192/5461"
    limit: int = 40
    offset: int = 0


class ImageSearchResult(BaseModel):
    query: str
    elapsed_ms: float
    images: list  # [{image_id, sequence_id, lat, lng, caption, mapillary_url, visual_score}, ...]
    total_in_area: int
    total_matches: int
    has_more: bool
    reranked: bool = False


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (TEMPLATE_DIR / "index.html").read_text()


@app.post("/api/search", response_model=SearchResult)
def search(req: SearchRequest):
    t0 = time.time()

    if req.mode == "combined":
        results = combined_search(req.query, req.geo_level)
    elif req.mode == "keyword":
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


@app.get("/api/mapillary/thumb/{image_id}")
async def mapillary_thumb(image_id: str):
    """Resolve Mapillary thumbnail URL and redirect to CDN."""
    global _http_client

    if not MAPILLARY_CLIENT_TOKEN:
        # No token: redirect to Mapillary viewer as fallback
        return RedirectResponse(
            url=f"https://www.mapillary.com/app/?pKey={image_id}",
            status_code=302,
        )

    # Check cache
    if image_id in _thumb_cache:
        _thumb_cache.move_to_end(image_id)
        return RedirectResponse(url=_thumb_cache[image_id], status_code=302)

    # Lazy-init httpx client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=10.0)

    try:
        resp = await _http_client.get(
            f"https://graph.mapillary.com/{image_id}",
            params={"access_token": MAPILLARY_CLIENT_TOKEN, "fields": "thumb_1024_url"},
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"Mapillary API returned {resp.status_code}")

        data = resp.json()
        thumb_url = data.get("thumb_1024_url")
        if not thumb_url:
            raise HTTPException(502, "No thumbnail URL in Mapillary response")

        # Cache the URL
        _thumb_cache[image_id] = thumb_url
        if len(_thumb_cache) > _THUMB_CACHE_MAX:
            _thumb_cache.popitem(last=False)

        return RedirectResponse(url=thumb_url, status_code=302)

    except httpx.HTTPError as e:
        raise HTTPException(502, f"Failed to fetch from Mapillary: {e}")


@app.post("/api/search/grid", response_model=GridSearchResult)
def search_grid(req: GridSearchRequest):
    """Grid-level keyword search. z8/z10 use pre-computed indexes; z12/z14 use on-demand queries."""
    t0 = time.time()
    tokens = stem_tokenize(req.query)
    raw_tokens = raw_tokenize(req.query)
    if not tokens:
        return GridSearchResult(query=req.query, zoom=req.zoom, elapsed_ms=0, tiles=[])

    if req.zoom in (8, 10):
        tiles = _grid_search_precomputed(tokens, req.zoom, req.bbox)
    elif req.zoom in (12, 14):
        tiles = _grid_search_ondemand(tokens, raw_tokens, req.zoom, req.bbox, req.parent_tile)
    else:
        raise HTTPException(400, f"Unsupported zoom level: {req.zoom}. Use 8, 10, 12, or 14.")

    elapsed_ms = (time.time() - t0) * 1000
    return GridSearchResult(
        query=req.query, zoom=req.zoom,
        elapsed_ms=round(elapsed_ms, 1), tiles=tiles,
    )


@app.post("/api/search/images", response_model=ImageSearchResult)
def search_images(req: ImageSearchRequest):
    """Image-level search within a bounding box or tile. Returns individual images matching query."""
    t0 = time.time()
    tokens = stem_tokenize(req.query)
    raw_tokens = raw_tokenize(req.query)
    if not tokens:
        return ImageSearchResult(
            query=req.query, elapsed_ms=0, images=[], total_in_area=0,
            total_matches=0, has_more=False,
        )

    images, total, total_matches, reranked = _image_search(
        raw_tokens, req.bbox, req.tile_key, req.limit, req.offset, req.query,
    )
    elapsed_ms = (time.time() - t0) * 1000

    return ImageSearchResult(
        query=req.query,
        elapsed_ms=round(elapsed_ms, 1),
        images=images,
        total_in_area=total,
        total_matches=total_matches,
        has_more=total_matches > req.offset + req.limit,
        reranked=reranked,
    )


# ── Search implementations ───────────────────────────────────────

def _compute_keyword_counts(tokens, level):
    """Get per-geography keyword match counts and totals."""
    kw = keyword_indexes[level]
    index = kw["index"]
    geo_totals = kw["geo_totals"]

    geo_counts = {}
    for token in tokens:
        if token in index:
            for geo, count in index[token].items():
                geo_counts[geo] = geo_counts.get(geo, 0) + count

    return geo_counts, geo_totals


def _compute_signed_chi(geo_counts, geo_totals):
    """Compute signed chi-squared for each geography.

    Measures whether a term appears more (positive) or less (negative)
    than expected under uniform distribution across geographies.

    signed_chi = sign(O - E) * (O - E)^2 / E

    where O = observed count, E = expected count based on global rate.
    """
    total_observed = sum(geo_counts.values())
    total_images = sum(geo_totals.values())

    if total_images == 0 or total_observed == 0:
        return {}

    global_rate = total_observed / total_images

    chi_scores = {}
    for geo, total in geo_totals.items():
        observed = geo_counts.get(geo, 0)
        expected = global_rate * total

        if expected < 0.001:
            continue

        diff = observed - expected
        chi_val = (diff * diff) / expected
        # Sign it: positive = over-represented, negative = under-represented
        signed = chi_val if diff >= 0 else -chi_val
        chi_scores[geo] = signed

    return chi_scores


def _compute_tfidf_scores(tokens, level):
    """Score geographies by TF-IDF cosine similarity to the query.

    Uses the pre-built TF-IDF vocabulary and matrix to create a sparse
    query vector, then computes dot product against each geography row.
    """
    if level not in tfidf_models:
        return {}

    model = tfidf_models[level]
    vocabulary = model["vocabulary"]
    matrix = model["matrix"]       # sparse CSR, shape (n_geos, n_terms)
    geo_names = model["geo_names"]

    # Build sparse query vector from stemmed tokens
    query_indices = []
    for token in tokens:
        if token in vocabulary:
            query_indices.append(vocabulary[token])

    if not query_indices:
        return {}

    # Create sparse query vector (1 x n_terms)
    n_terms = matrix.shape[1]
    data = np.ones(len(query_indices), dtype=np.float64)
    cols = np.array(query_indices, dtype=np.int32)
    rows = np.zeros(len(query_indices), dtype=np.int32)
    query_vec = sparse.csr_matrix((data, (rows, cols)), shape=(1, n_terms))

    # L2 normalize query
    q_norm = sparse.linalg.norm(query_vec)
    if q_norm > 0:
        query_vec = query_vec / q_norm

    # Cosine similarity = dot product (both matrix rows and query are L2 normalized)
    similarities = (matrix @ query_vec.T).toarray().flatten()

    return {geo_names[i]: float(similarities[i]) for i in range(len(geo_names))}


def keyword_search(query, geo_level):
    """Stemmed keyword prevalence search with signed chi-squared."""
    if geo_level not in keyword_indexes:
        raise HTTPException(400, f"No keyword index for level: {geo_level}")

    tokens = stem_tokenize(query)
    if not tokens:
        return []

    geo_counts, geo_totals = _compute_keyword_counts(tokens, geo_level)
    chi_scores = _compute_signed_chi(geo_counts, geo_totals)

    results = []
    for geo, total in geo_totals.items():
        count = geo_counts.get(geo, 0)
        if count == 0 and geo not in chi_scores:
            continue
        results.append({
            "geo_name": geo,
            "count": count,
            "total_images": total,
            "prevalence": round(count / total, 6) if total > 0 else 0,
            "signed_chi": round(chi_scores.get(geo, 0), 2),
            "tfidf_score": 0,
            "score": round(chi_scores.get(geo, 0), 2),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def combined_search(query, geo_level):
    """Combined keyword + TF-IDF search with signed chi-squared ranking.

    Scoring formula:
      score = signed_chi_norm * 0.7 + tfidf_score_norm * 0.3

    The signed chi-squared captures statistical over/under-representation
    (penalizing small samples), while TF-IDF captures softer topical
    relevance (bigrams, partial matches) that exact stemming misses.
    """
    if geo_level not in keyword_indexes:
        raise HTTPException(400, f"No keyword index for level: {geo_level}")

    tokens = stem_tokenize(query)
    if not tokens:
        return []

    geo_counts, geo_totals = _compute_keyword_counts(tokens, geo_level)
    chi_scores = _compute_signed_chi(geo_counts, geo_totals)
    tfidf_scores = _compute_tfidf_scores(tokens, geo_level)

    # Collect all geographies that have any signal
    all_geos = set(geo_counts.keys()) | set(tfidf_scores.keys())
    if not all_geos:
        return []

    # Normalize signed chi scores to [-1, 1] range for blending
    chi_vals = [chi_scores.get(g, 0) for g in all_geos]
    chi_abs_max = max(abs(v) for v in chi_vals) if chi_vals else 1.0
    if chi_abs_max == 0:
        chi_abs_max = 1.0

    # TF-IDF scores are already [0, 1] from cosine similarity
    tfidf_max = max(tfidf_scores.values()) if tfidf_scores else 1.0
    if tfidf_max == 0:
        tfidf_max = 1.0

    CHI_WEIGHT = 0.7
    TFIDF_WEIGHT = 0.3

    results = []
    for geo in all_geos:
        total = geo_totals.get(geo, 0)
        count = geo_counts.get(geo, 0)
        chi = chi_scores.get(geo, 0)
        tfidf = tfidf_scores.get(geo, 0)

        chi_norm = chi / chi_abs_max       # [-1, 1]
        tfidf_norm = tfidf / tfidf_max     # [0, 1]

        score = CHI_WEIGHT * chi_norm + TFIDF_WEIGHT * tfidf_norm

        results.append({
            "geo_name": geo,
            "count": count,
            "total_images": total,
            "prevalence": round(count / total, 6) if total > 0 else 0,
            "signed_chi": round(chi, 2),
            "tfidf_score": round(tfidf, 4),
            "score": round(score, 4),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ── Grid search implementations ──────────────────────────────────

def _grid_search_precomputed(tokens, zoom, bbox):
    """Search pre-computed z8 or z10 keyword index. Returns tile-level prevalence."""
    level = f"z{zoom}"
    if level not in keyword_indexes:
        raise HTTPException(400, f"No keyword index for level: {level}")

    kw = keyword_indexes[level]
    index = kw["index"]
    geo_totals = kw["geo_totals"]

    # Sum image counts per tile across query tokens
    tile_counts = {}
    for token in tokens:
        if token in index:
            for tile_key, count in index[token].items():
                tile_counts[tile_key] = tile_counts.get(tile_key, 0) + count

    # If bbox filter provided, only include tiles that overlap
    if bbox and len(bbox) == 4:
        west, south, east, north = bbox
        valid_tiles = set(tiles_in_bbox(west, south, east, north, zoom))
        tile_counts = {k: v for k, v in tile_counts.items() if k in valid_tiles}

    # Build results with tile geometry
    results = []
    for tile_key, count in tile_counts.items():
        total = geo_totals.get(tile_key, 1)
        parts = tile_key.split("/")
        if len(parts) != 3:
            continue
        x, y = int(parts[1]), int(parts[2])
        w, s, e, n = tile_to_bbox(x, y, zoom)

        results.append({
            "tile_key": tile_key,
            "x": x, "y": y,
            "west": round(w, 6), "south": round(s, 6),
            "east": round(e, 6), "north": round(n, 6),
            "count": count,
            "total": total,
            "prevalence": round(count / total, 6),
        })

    results.sort(key=lambda r: r["prevalence"], reverse=True)
    return results


def _fast_caption_match(captions, raw_tokens):
    """Fast vectorized caption matching using regex instead of per-row stemming.

    Uses raw (unstemmed) query tokens as prefixes so that e.g. "sunny" matches
    "sunny", "sunniest", etc. in the raw caption text.
    """
    patterns = [r'\b' + re.escape(t) + r'\w*' for t in raw_tokens]
    combined = '|'.join(patterns)
    return captions.str.contains(combined, case=False, na=False, regex=True)


def _grid_search_ondemand(tokens, raw_tokens, zoom, bbox, parent_tile):
    """On-demand search at z12 or z14 by loading z10 caption partitions."""
    if caption_tile_dir is None:
        raise HTTPException(503, "Caption tile partitions not available")

    z10_keys = []
    if parent_tile:
        parts = parent_tile.split("/")
        parent_zoom = int(parts[0][1:])
        px, py = int(parts[1]), int(parts[2])
        pw, ps, pe, pn = tile_to_bbox(px, py, parent_zoom)
        z10_keys = tiles_in_bbox(pw, ps, pe, pn, 10)
    elif bbox and len(bbox) == 4:
        west, south, east, north = bbox
        z10_keys = tiles_in_bbox(west, south, east, north, 10)
    else:
        raise HTTPException(400, "bbox or parent_tile required for z12/z14 queries")

    if len(z10_keys) > 100:
        raise HTTPException(400, f"Query area too large: {len(z10_keys)} z10 tiles. Zoom in further.")

    # Only load columns needed for grid aggregation
    tile_col = f"z{zoom}_tile"
    load_cols = ["lat", "lng", "caption"]
    if tile_col in ("z12_tile", "z14_tile"):
        load_cols.append(tile_col)

    dfs = []
    for z10_key in z10_keys:
        part_df = load_tile_partition(z10_key, columns=load_cols)
        if part_df is not None:
            # Apply bbox filter per-partition to reduce concat size
            if bbox and len(bbox) == 4:
                west, south, east, north = bbox
                part_df = part_df[
                    (part_df["lat"] >= south) & (part_df["lat"] <= north) &
                    (part_df["lng"] >= west) & (part_df["lng"] <= east)
                ]
            if len(part_df) > 0:
                dfs.append(part_df)

    if not dfs:
        return []

    df = pd.concat(dfs, ignore_index=True)

    if tile_col not in df.columns:
        lat_rad = np.radians(df["lat"].values)
        n_tiles = 2 ** zoom
        x = ((df["lng"].values + 180.0) / 360.0 * n_tiles).astype(np.int32)
        y = ((1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n_tiles).astype(np.int32)
        x = np.clip(x, 0, n_tiles - 1)
        y = np.clip(y, 0, n_tiles - 1)
        df[tile_col] = [f"z{zoom}/{xi}/{yi}" for xi, yi in zip(x, y)]

    # Fast vectorized matching using raw (unstemmed) tokens as prefixes
    matches = _fast_caption_match(df["caption"], raw_tokens)
    matched_df = df[matches]

    tile_totals = df.groupby(tile_col).size().to_dict()
    tile_match_counts = matched_df.groupby(tile_col).size().to_dict()

    results = []
    for tile_key, count in tile_match_counts.items():
        total = tile_totals.get(tile_key, 1)
        parts = tile_key.split("/")
        if len(parts) != 3:
            continue
        tx, ty = int(parts[1]), int(parts[2])
        w, s, e, n_lat = tile_to_bbox(tx, ty, zoom)

        results.append({
            "tile_key": tile_key,
            "x": tx, "y": ty,
            "west": round(w, 6), "south": round(s, 6),
            "east": round(e, 6), "north": round(n_lat, 6),
            "count": count,
            "total": total,
            "prevalence": round(count / total, 6),
        })

    results.sort(key=lambda r: r["prevalence"], reverse=True)
    return results


def _image_search(raw_tokens, bbox, tile_key, limit, offset=0, query=""):
    """Search for individual images matching query tokens, with DINOv2 visual re-ranking.

    After text matching returns candidates, fetches DINOv2 CLS embeddings from S3
    and re-ranks by visual consensus — true positives cluster visually, false positives
    (hallucinated captions) scatter. This pushes genuine matches to the top.

    Full re-ranked lists are cached by (query, tile_key) so pagination doesn't re-fetch.
    """
    if caption_tile_dir is None:
        raise HTTPException(503, "Caption tile partitions not available")

    # Cache key for re-ranked results (makes pagination instant)
    cache_key = (query, tile_key or str(bbox))

    # Check re-rank result cache — serves paginated requests without re-fetching
    if cache_key in _rerank_result_cache:
        _rerank_result_cache.move_to_end(cache_key)
        cached = _rerank_result_cache[cache_key]
        page = cached["images"][offset:offset + limit]
        return page, cached["total_in_area"], cached["total_matches"], cached["reranked"]

    # Resolve bounding box from tile_key or bbox parameter
    if tile_key:
        parts = tile_key.split("/")
        tz = int(parts[0][1:])
        tx, ty = int(parts[1]), int(parts[2])
        w, s, e, n = tile_to_bbox(tx, ty, tz)
        z10_keys = tiles_in_bbox(w, s, e, n, 10)
    elif bbox and len(bbox) == 4:
        west, south, east, north = bbox
        z10_keys = tiles_in_bbox(west, south, east, north, 10)
    else:
        raise HTTPException(400, "bbox or tile_key required for image search")

    if len(z10_keys) > 50:
        raise HTTPException(400, f"Query area too large: {len(z10_keys)} z10 tiles. Zoom in further.")

    # Load tile partitions
    dfs = []
    for z10_key in z10_keys:
        part_df = load_tile_partition(z10_key)
        if part_df is not None:
            dfs.append(part_df)

    if not dfs:
        return [], 0, 0, False

    df = pd.concat(dfs, ignore_index=True)

    # Spatial filter
    if tile_key:
        df = df[(df["lat"] >= s) & (df["lat"] <= n) &
                (df["lng"] >= w) & (df["lng"] <= e)]
    elif bbox:
        west, south, east, north = bbox
        df = df[(df["lat"] >= south) & (df["lat"] <= north) &
                (df["lng"] >= west) & (df["lng"] <= east)]

    total_in_area = len(df)

    # Text matching — all existing logic preserved
    matches = _fast_caption_match(df["caption"], raw_tokens)
    matched_all = df[matches]
    total_matches = len(matched_all)

    if total_matches == 0:
        return [], total_in_area, 0, False

    # Build image dicts for ALL text matches (up to RERANK_TEXT_CAP)
    matched_capped = matched_all.head(RERANK_TEXT_CAP)
    all_images = []
    for _, row in matched_capped.iterrows():
        all_images.append({
            "image_id": str(row["image_id"]),
            "sequence_id": str(row.get("sequence_id", "")),
            "lat": round(float(row["lat"]), 6),
            "lng": round(float(row["lng"]), 6),
            "caption": str(row["caption"]),
            "mapillary_url": f"https://www.mapillary.com/app/?pKey={row['image_id']}",
        })

    # Visual re-ranking via DINOv2 embeddings
    reranked = False
    if RERANK_ENABLED and len(all_images) >= RERANK_MIN_EMBEDDINGS:
        try:
            embeddings = _fetch_embeddings_parallel(all_images)
            if len(embeddings) >= RERANK_MIN_EMBEDDINGS:
                all_images = _rerank_by_visual_consensus(all_images, embeddings)
                reranked = True
                logger.info(
                    "Re-ranked %d images (%d embeddings) for query=%r tile=%s",
                    len(all_images), len(embeddings), query, tile_key,
                )
        except Exception as e:
            logger.warning("Re-ranking failed, using text-match order: %s", e)

    # Cache full re-ranked list for pagination
    _rerank_result_cache[cache_key] = {
        "images": all_images,
        "total_in_area": total_in_area,
        "total_matches": total_matches,
        "reranked": reranked,
    }
    if len(_rerank_result_cache) > _RERANK_RESULT_CACHE_MAX:
        _rerank_result_cache.popitem(last=False)

    # Paginate
    page = all_images[offset:offset + limit]
    return page, total_in_area, total_matches, reranked


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
