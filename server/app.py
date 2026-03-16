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

import json
import math
import pickle
import time
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
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
caption_tile_dir = None  # Path to data/captions_by_tile/ for on-demand z12/z14 queries

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


def load_tile_partition(z10_key):
    """Load a caption partition file for a z10 tile. Returns DataFrame or None."""
    if caption_tile_dir is None:
        return None
    safe_name = z10_key.replace("/", "_")
    path = caption_tile_dir / f"{safe_name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


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
    mode: str = "keyword"       # "keyword" or "semantic"
    geo_level: str = "state"    # "state" or "msa"
    top_k: int = 100_000        # FAISS top-K for semantic mode


class SearchResult(BaseModel):
    query: str
    mode: str
    geo_level: str
    elapsed_ms: float
    results: list  # [{geo_name, prevalence, count, total_images}, ...]


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
    limit: int = 200


class ImageSearchResult(BaseModel):
    query: str
    elapsed_ms: float
    images: list  # [{image_id, sequence_id, lat, lng, caption, mapillary_url}, ...]
    total_in_area: int


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


@app.post("/api/search/grid", response_model=GridSearchResult)
def search_grid(req: GridSearchRequest):
    """Grid-level keyword search. z8/z10 use pre-computed indexes; z12/z14 use on-demand queries."""
    t0 = time.time()
    tokens = stem_tokenize(req.query)
    if not tokens:
        return GridSearchResult(query=req.query, zoom=req.zoom, elapsed_ms=0, tiles=[])

    if req.zoom in (8, 10):
        tiles = _grid_search_precomputed(tokens, req.zoom, req.bbox)
    elif req.zoom in (12, 14):
        tiles = _grid_search_ondemand(tokens, req.zoom, req.bbox, req.parent_tile)
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
    if not tokens:
        return ImageSearchResult(
            query=req.query, elapsed_ms=0, images=[], total_in_area=0,
        )

    images, total = _image_search(tokens, req.bbox, req.tile_key, req.limit)
    elapsed_ms = (time.time() - t0) * 1000

    return ImageSearchResult(
        query=req.query,
        elapsed_ms=round(elapsed_ms, 1),
        images=images,
        total_in_area=total,
    )


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


def _grid_search_ondemand(tokens, zoom, bbox, parent_tile):
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

    dfs = []
    for z10_key in z10_keys:
        part_df = load_tile_partition(z10_key)
        if part_df is not None:
            dfs.append(part_df)

    if not dfs:
        return []

    df = pd.concat(dfs, ignore_index=True)

    if bbox and len(bbox) == 4:
        west, south, east, north = bbox
        df = df[(df["lat"] >= south) & (df["lat"] <= north) &
                (df["lng"] >= west) & (df["lng"] <= east)]

    tile_col = f"z{zoom}_tile"
    if tile_col not in df.columns:
        lat_rad = np.radians(df["lat"].values)
        n_tiles = 2 ** zoom
        x = ((df["lng"].values + 180.0) / 360.0 * n_tiles).astype(np.int32)
        y = ((1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n_tiles).astype(np.int32)
        x = np.clip(x, 0, n_tiles - 1)
        y = np.clip(y, 0, n_tiles - 1)
        df[tile_col] = [f"z{zoom}/{xi}/{yi}" for xi, yi in zip(x, y)]

    token_set = set(tokens)
    matches = df["caption"].apply(
        lambda cap: bool(token_set & set(stem_tokenize(cap))) if pd.notna(cap) else False
    )
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


def _image_search(tokens, bbox, tile_key, limit):
    """Search for individual images matching query tokens in a geographic area."""
    if caption_tile_dir is None:
        raise HTTPException(503, "Caption tile partitions not available")

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

    dfs = []
    for z10_key in z10_keys:
        part_df = load_tile_partition(z10_key)
        if part_df is not None:
            dfs.append(part_df)

    if not dfs:
        return [], 0

    df = pd.concat(dfs, ignore_index=True)

    if tile_key:
        df = df[(df["lat"] >= s) & (df["lat"] <= n) &
                (df["lng"] >= w) & (df["lng"] <= e)]
    elif bbox:
        west, south, east, north = bbox
        df = df[(df["lat"] >= south) & (df["lat"] <= north) &
                (df["lng"] >= west) & (df["lng"] <= east)]

    total_in_area = len(df)

    token_set = set(tokens)
    matches = df["caption"].apply(
        lambda cap: bool(token_set & set(stem_tokenize(cap))) if pd.notna(cap) else False
    )
    matched = df[matches].head(limit)

    images = []
    for _, row in matched.iterrows():
        images.append({
            "image_id": str(row["image_id"]),
            "sequence_id": str(row.get("sequence_id", "")),
            "lat": round(float(row["lat"]), 6),
            "lng": round(float(row["lng"]), 6),
            "caption": str(row["caption"]),
            "mapillary_url": f"https://www.mapillary.com/app/?pKey={row['image_id']}",
        })

    return images, total_in_area


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
