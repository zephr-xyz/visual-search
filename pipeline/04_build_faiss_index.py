#!/usr/bin/env python3
"""
Step 4: Build FAISS IVF-PQ index from 768-dim caption embeddings.

Loads embedding batches from data/extracts/emb_*.npy, trains an IVF-PQ
index, adds all vectors, and saves the index with aligned metadata for
geographic aggregation at query time.

Index: IVF4096,PQ96x8
  - 4096 Voronoi cells for coarse quantization
  - PQ96: 768 dims / 96 subquantizers = 8 dims each, 256 centroids (1 byte)
  - Storage: ~12.2M * 96 bytes ≈ 1.2 GB
  - Query: nprobe=64 searches ~1.5% of vectors (fast)

Input:  data/extracts/emb_*.npy           - embedding batches (N, 768) float32
        data/extracts/meta_*.parquet      - aligned metadata
        data/image_geography.parquet      - geography assignments
Output: data/faiss_index.ivfpq            - trained FAISS index
        data/faiss_ids.npy                - image_id array aligned with index
        data/faiss_geo.parquet            - image_id → state/MSA for aggregation

Usage:
  python pipeline/04_build_faiss_index.py
  python pipeline/04_build_faiss_index.py --train-sample 500000  # training sample size
  python pipeline/04_build_faiss_index.py --nlist 4096 --pq 96   # index params
"""

import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXTRACT_DIR = DATA_DIR / "extracts"

DIM = 768


def load_embeddings():
    """Load all embedding batches and aligned image IDs."""
    emb_files = sorted(EXTRACT_DIR.glob("emb_*.npy"))
    meta_files = sorted(EXTRACT_DIR.glob("meta_*.parquet"))

    if not emb_files:
        print("ERROR: No embedding files found. Run 01_extract_from_s3.py first.")
        raise SystemExit(1)

    print(f"Loading {len(emb_files)} embedding batches...")
    all_embeddings = []
    all_ids = []

    for emb_f, meta_f in zip(emb_files, meta_files):
        emb = np.load(emb_f)
        meta = pd.read_parquet(meta_f, columns=["image_id"])

        # Ensure alignment
        assert len(emb) == len(meta), (
            f"Mismatch: {emb_f.name} has {len(emb)} rows, "
            f"{meta_f.name} has {len(meta)} rows"
        )

        all_embeddings.append(emb)
        all_ids.extend(meta["image_id"].tolist())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    image_ids = np.array(all_ids)

    print(f"  Total: {len(embeddings):,} vectors, {DIM}-dim")
    return embeddings, image_ids


def normalize_embeddings(embeddings):
    """L2-normalize for cosine similarity via inner product."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return embeddings / norms


def build_index(embeddings, nlist, pq_subquantizers, train_sample_size):
    """Build and train a FAISS IVF-PQ index."""
    n = len(embeddings)
    print(f"\nBuilding FAISS IVF{nlist},PQ{pq_subquantizers}x8 index...")

    # Use inner product (equivalent to cosine sim on normalized vectors)
    quantizer = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFPQ(quantizer, DIM, nlist, pq_subquantizers, 8)

    # Train on a random sample
    if n > train_sample_size:
        print(f"  Training on {train_sample_size:,} random samples...")
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, train_sample_size, replace=False)
        train_data = embeddings[sample_idx]
    else:
        print(f"  Training on all {n:,} vectors...")
        train_data = embeddings

    t0 = time.time()
    index.train(train_data)
    print(f"  Training done ({time.time() - t0:.1f}s)")

    # Add all vectors
    print(f"  Adding {n:,} vectors...")
    t0 = time.time()

    # Add in chunks to show progress
    chunk_size = 500_000
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        index.add(embeddings[i:end])
        if end < n:
            print(f"    {end:,} / {n:,}")

    print(f"  Adding done ({time.time() - t0:.1f}s)")
    print(f"  Index size: {index.ntotal:,} vectors")

    return index


def build_geo_lookup(image_ids):
    """Build a lookup from FAISS index position to geography."""
    geo_path = DATA_DIR / "image_geography.parquet"
    if not geo_path.exists():
        print("WARNING: image_geography.parquet not found, skipping geo lookup.")
        return None

    print("Building geography lookup...")
    geo_df = pd.read_parquet(
        geo_path,
        columns=["image_id", "state_name", "state_abbr", "cbsa_name"],
    )

    # Create a DataFrame aligned with FAISS index order
    ids_df = pd.DataFrame({"image_id": image_ids})
    lookup = ids_df.merge(geo_df, on="image_id", how="left")

    in_state = lookup["state_name"].notna().sum()
    in_msa = lookup["cbsa_name"].notna().sum()
    print(f"  {in_state:,} in US states, {in_msa:,} in MSAs")

    return lookup


def main():
    parser = argparse.ArgumentParser(description="Build FAISS IVF-PQ index")
    parser.add_argument("--nlist", type=int, default=4096, help="IVF cells")
    parser.add_argument("--pq", type=int, default=96, help="PQ subquantizers")
    parser.add_argument("--train-sample", type=int, default=500_000)
    args = parser.parse_args()

    embeddings, image_ids = load_embeddings()

    print("Normalizing embeddings...")
    embeddings = normalize_embeddings(embeddings)

    index = build_index(embeddings, args.nlist, args.pq, args.train_sample)

    # Save index
    index_path = DATA_DIR / "faiss_index.ivfpq"
    print(f"\nSaving index to {index_path}...")
    faiss.write_index(index, str(index_path))
    size_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"  Index size: {size_mb:.0f} MB")

    # Save aligned image IDs
    ids_path = DATA_DIR / "faiss_ids.npy"
    np.save(ids_path, image_ids)
    print(f"  Saved image IDs: {ids_path}")

    # Build and save geo lookup
    geo_lookup = build_geo_lookup(image_ids)
    if geo_lookup is not None:
        geo_path = DATA_DIR / "faiss_geo.parquet"
        geo_lookup.to_parquet(geo_path, index=False)
        print(f"  Saved geo lookup: {geo_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
