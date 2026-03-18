#!/usr/bin/env python3
"""
Step 3: Build stemmed keyword prevalence indexes from captions.

For each (stemmed_term, geography) pair, stores the count of images whose
caption contains that term.  At query time, prevalence = count / total_images.

Also builds a TF-IDF matrix where each "document" is a geography (all
captions concatenated), enabling TF-IDF-weighted ranking of geographies.

Supports multiple geo levels:
  - state / msa: pre-existing admin boundary levels
  - z8 / z10:    Web Mercator tile grid levels (for drill-down)

Also creates per-z10-tile caption partitions (data/captions_by_tile/)
for on-demand z12/z14 queries at runtime.

Input:  data/image_geography.parquet
Output: data/keyword_index_{level}.pkl - {term: {geo: count}} dict
        data/tfidf_model_{level}.pkl   - fitted TfidfVectorizer + matrix
        data/geography_stats.json      - image counts per geography
        data/captions_by_tile/*.parquet - per-z10-tile caption partitions

Usage:
  python pipeline/03_build_tfidf_index.py
  python pipeline/03_build_tfidf_index.py --geo-level state
  python pipeline/03_build_tfidf_index.py --geo-level z10
  python pipeline/03_build_tfidf_index.py --geo-level all    # default
"""

import argparse
import json
import multiprocessing as mp
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure NLTK data is available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
# Add domain-specific stop words common in street-level image captions
STOP_WORDS.update(
    {
        "image",
        "photograph",
        "photo",
        "panoramic",
        "panorama",
        "equirectangular",
        "captured",
        "camera",
        "degree",
        "360",
        "mounted",
        "vehicle",
        "roof",
    }
)

stemmer = SnowballStemmer("english")


def stem_tokenize(text):
    """Tokenize, lowercase, remove non-alpha, stem, remove stop words."""
    tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return [stemmer.stem(t) for t in tokens if t not in STOP_WORDS]


def _tokenize_chunk(args):
    """Tokenize a chunk of (geo_name, caption) pairs. Used by multiprocessing."""
    chunk_geos, chunk_captions = args
    local_index = defaultdict(lambda: defaultdict(int))
    local_totals = defaultdict(int)
    for geo_name, caption in zip(chunk_geos, chunk_captions):
        if pd.isna(geo_name) or not caption:
            continue
        local_totals[geo_name] += 1
        seen = set()
        for token in stem_tokenize(caption):
            if token not in seen:
                local_index[token][geo_name] += 1
                seen.add(token)
    return dict(local_index), dict(local_totals)


def build_keyword_index_chunked(geo_path, geo_col, batch_size=500_000, n_workers=None):
    """Build inverted index: stemmed_term -> {geography: image_count}.

    Reads data in chunks via PyArrow to avoid loading all 14M captions at once.
    Each chunk is processed with multiprocessing, then results are merged.
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    import pyarrow.parquet as pq

    print(f"  Building keyword index for {geo_col} ({n_workers} workers)...")
    t0 = time.time()

    index = defaultdict(lambda: defaultdict(int))
    geo_totals = defaultdict(int)
    total_rows = 0

    pf = pq.ParquetFile(geo_path)
    for batch_i, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=[geo_col, "caption"])):
        df_chunk = batch.to_pandas()
        geos = df_chunk[geo_col].tolist()
        captions = df_chunk["caption"].tolist()
        n = len(geos)
        total_rows += n
        del df_chunk

        # Split into sub-chunks for parallel processing
        sub_size = max(1, n // n_workers)
        chunks = []
        for i in range(0, n, sub_size):
            chunks.append((geos[i:i + sub_size], captions[i:i + sub_size]))
        del geos, captions

        with mp.Pool(n_workers) as pool:
            results = pool.map(_tokenize_chunk, chunks)
        del chunks

        for chunk_index, chunk_totals in results:
            for term, geo_counts in chunk_index.items():
                for geo, count in geo_counts.items():
                    index[term][geo] += count
            for geo, count in chunk_totals.items():
                geo_totals[geo] += count
        del results

        if (batch_i + 1) % 5 == 0:
            print(f"    batch {batch_i+1}: {total_rows:,} rows processed", flush=True)

    index = {term: dict(geos) for term, geos in index.items()}
    geo_totals = dict(geo_totals)

    elapsed = time.time() - t0
    print(f"    {len(index):,} unique terms, {len(geo_totals)} geographies ({elapsed:.1f}s)")
    return index, geo_totals


def _stem_doc(text):
    """Stem a single concatenated document. Used by multiprocessing."""
    return " ".join(stem_tokenize(text))


def build_tfidf_matrix_chunked(geo_path, geo_col, batch_size=500_000, n_workers=None):
    """Build TF-IDF matrix where each row is a geography.

    Reads data in chunks to build per-geography stemmed token lists,
    then fits a TfidfVectorizer.
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    import pyarrow.parquet as pq

    print(f"  Building TF-IDF matrix for {geo_col} ({n_workers} workers)...")
    t0 = time.time()

    # Phase 1: Collect stemmed tokens per geography in chunks
    # Store as pre-stemmed token strings to avoid holding raw captions
    geo_tokens = defaultdict(list)  # geo -> list of stemmed token strings

    pf = pq.ParquetFile(geo_path)
    total_rows = 0
    for batch_i, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=[geo_col, "caption"])):
        df_chunk = batch.to_pandas()
        total_rows += len(df_chunk)

        # Group captions by geo within this chunk
        chunk_geo_captions = defaultdict(list)
        for geo_name, caption in zip(df_chunk[geo_col], df_chunk["caption"]):
            if pd.isna(geo_name) or not caption:
                continue
            chunk_geo_captions[geo_name].append(caption)
        del df_chunk

        # Stem concatenated captions per geo in this chunk using multiprocessing
        geo_names_chunk = list(chunk_geo_captions.keys())
        raw_docs_chunk = [" ".join(chunk_geo_captions[g]) for g in geo_names_chunk]
        del chunk_geo_captions

        with mp.Pool(n_workers) as pool:
            stemmed_chunk = pool.map(_stem_doc, raw_docs_chunk)
        del raw_docs_chunk

        for geo_name, stemmed in zip(geo_names_chunk, stemmed_chunk):
            geo_tokens[geo_name].append(stemmed)
        del stemmed_chunk, geo_names_chunk

        if (batch_i + 1) % 5 == 0:
            print(f"    batch {batch_i+1}: {total_rows:,} rows, {len(geo_tokens)} geos", flush=True)

    # Phase 2: Build final documents and fit TF-IDF
    geo_names = sorted(geo_tokens.keys())
    documents = [" ".join(geo_tokens[g]) for g in geo_names]
    del geo_tokens

    print(f"    Fitting TF-IDF on {len(geo_names)} geographies...", flush=True)
    vectorizer = TfidfVectorizer(
        token_pattern=r"[a-z]{2,}",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=100_000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    del documents

    elapsed = time.time() - t0
    vocab_size = len(vectorizer.vocabulary_)
    print(
        f"    Matrix: {tfidf_matrix.shape[0]} geographies x {vocab_size:,} terms "
        f"({elapsed:.1f}s)"
    )
    return vectorizer, tfidf_matrix, geo_names


def build_caption_partitions_chunked(geo_path, partition_col, output_dir, batch_size=500_000):
    """Partition image captions by tile key for on-demand z12/z14 queries.

    Reads data in chunks to avoid loading all 14M rows at once.
    Accumulates per-tile DataFrames and writes them at the end.
    """
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Building caption partitions by {partition_col}...")
    t0 = time.time()

    desired_cols = ["image_id", "sequence_id", "lat", "lng", "caption",
                    "z12_tile", "z14_tile", partition_col]
    available_cols = pq.read_schema(geo_path).names
    load_cols = [c for c in desired_cols if c in available_cols]

    # Accumulate rows per tile
    tile_rows = defaultdict(list)
    total_rows = 0

    pf = pq.ParquetFile(geo_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=load_cols):
        df_chunk = batch.to_pandas()
        df_chunk = df_chunk.dropna(subset=[partition_col, "caption"])
        total_rows += len(df_chunk)

        for tile_key, group in df_chunk.groupby(partition_col):
            tile_rows[tile_key].append(group.drop(columns=[partition_col]))
        del df_chunk

    # Write partition files
    n_partitions = 0
    for tile_key, dfs in tile_rows.items():
        safe_name = tile_key.replace("/", "_")
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(output_dir / f"{safe_name}.parquet", index=False)
        n_partitions += 1
    del tile_rows

    elapsed = time.time() - t0
    print(f"    {n_partitions:,} partition files from {total_rows:,} rows ({elapsed:.1f}s)")
    return n_partitions


def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF keyword index")
    parser.add_argument(
        "--geo-level",
        choices=["state", "msa", "z8", "z10", "both", "all"],
        default="all",
    )
    parser.add_argument(
        "--skip-partitions",
        action="store_true",
        help="Skip building per-z10-tile caption partitions",
    )
    args = parser.parse_args()

    geo_path = DATA_DIR / "image_geography.parquet"
    if not geo_path.exists():
        print(f"ERROR: {geo_path} not found. Run 02_geocode.py first.")
        raise SystemExit(1)

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(geo_path)
    available_cols = pf.schema.names
    total_rows = pf.metadata.num_rows
    print(f"Geocoded data: {total_rows:,} images, columns: {available_cols}")

    results = {}

    # Define all available levels and their column names
    all_levels = {
        "state": "state_name",
        "msa": "cbsa_name",
        "z8": "z8_tile",
        "z10": "z10_tile",
    }

    # Determine which levels to build
    if args.geo_level == "all":
        levels = list(all_levels.items())
    elif args.geo_level == "both":
        levels = [("state", "state_name"), ("msa", "cbsa_name")]
    else:
        levels = [(args.geo_level, all_levels[args.geo_level])]

    for level_name, geo_col in levels:
        if geo_col not in available_cols:
            print(f"\n=== {level_name.upper()} level === SKIPPED (column {geo_col} not found)")
            continue

        print(f"\n=== {level_name.upper()} level ===")

        kw_index, geo_totals = build_keyword_index_chunked(geo_path, geo_col)
        vectorizer, tfidf_matrix, geo_names = build_tfidf_matrix_chunked(geo_path, geo_col)

        # Save keyword index
        kw_path = DATA_DIR / f"keyword_index_{level_name}.pkl"
        with open(kw_path, "wb") as f:
            pickle.dump({"index": kw_index, "geo_totals": geo_totals}, f)
        print(f"  Saved keyword index: {kw_path}")
        del kw_index

        # Save TF-IDF model
        tfidf_path = DATA_DIR / f"tfidf_model_{level_name}.pkl"
        with open(tfidf_path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": vectorizer,
                    "matrix": tfidf_matrix,
                    "geo_names": geo_names,
                },
                f,
            )
        print(f"  Saved TF-IDF model: {tfidf_path}")
        del vectorizer, tfidf_matrix, geo_names

        results[level_name] = geo_totals

    # Build per-z10 caption partitions for on-demand z12/z14 queries
    if not args.skip_partitions and "z10_tile" in available_cols:
        print("\n=== Caption partitions (by z10 tile) ===")
        partition_dir = DATA_DIR / "captions_by_tile"
        build_caption_partitions_chunked(geo_path, "z10_tile", partition_dir)

    # Save geography stats
    stats_path = DATA_DIR / "geography_stats.json"
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved geography stats: {stats_path}")


if __name__ == "__main__":
    main()
