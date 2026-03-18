#!/usr/bin/env python3
"""
Step 4: Build FAISS IVF-PQ index from 768-dim caption embeddings.

Streams embedding batches from data/extracts/emb_*.npy to avoid loading
all vectors into memory at once (14M x 768 = ~40GB).

Index: IVF4096,PQ96x8
  - 4096 Voronoi cells for coarse quantization
  - PQ96: 768 dims / 96 subquantizers = 8 dims each, 256 centroids (1 byte)
  - Storage: ~14M * 96 bytes ≈ 1.3 GB
  - Query: nprobe=64 searches ~1.5% of vectors (fast)

Input:  data/extracts/emb_*.npy           - embedding batches (N, 768) float32
        data/extracts/meta_*.parquet      - aligned metadata
        data/image_geography.parquet      - geography assignments
Output: data/faiss_index.ivfpq            - trained FAISS index
        data/faiss_ids.npy                - image_id array aligned with index
        data/faiss_geo.parquet            - image_id -> state/MSA for aggregation

Usage:
  python pipeline/04_build_faiss_index.py
  python pipeline/04_build_faiss_index.py --train-sample 500000
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


def get_file_pairs():
    """Get aligned (embedding, metadata) file pairs."""
    emb_files = sorted(EXTRACT_DIR.glob("emb_*.npy"))
    meta_files = sorted(EXTRACT_DIR.glob("meta_*.parquet"))

    if not emb_files:
        print("ERROR: No embedding files found.")
        raise SystemExit(1)

    # Match by name pattern (emb_s0_000000.npy <-> meta_s0_000000.parquet)
    meta_lookup = {f.stem.replace("meta_", ""): f for f in meta_files}
    pairs = []
    for emb_f in emb_files:
        key = emb_f.stem.replace("emb_", "")
        meta_f = meta_lookup.get(key)
        if meta_f:
            pairs.append((emb_f, meta_f))

    print(f"Found {len(pairs)} embedding/metadata pairs")
    return pairs


def load_training_sample(pairs, sample_size):
    """Load a random sample of embeddings for training. Memory-efficient."""
    # First pass: count total vectors
    total = 0
    file_sizes = []
    for emb_f, _ in pairs:
        # Read shape without loading data
        with open(emb_f, "rb") as f:
            version = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format._read_array_header(f, version)
        n = shape[0]
        file_sizes.append(n)
        total += n

    print(f"  Total vectors across {len(pairs)} files: {total:,}")

    if total <= sample_size:
        # Load everything (small dataset)
        all_emb = []
        for emb_f, _ in pairs:
            all_emb.append(np.load(emb_f))
        return np.vstack(all_emb).astype(np.float32), total

    # Proportionally sample from each file
    rng = np.random.default_rng(42)
    samples = []
    remaining = sample_size

    for i, (emb_f, _) in enumerate(pairs):
        n = file_sizes[i]
        # How many to sample from this file
        if i == len(pairs) - 1:
            k = remaining
        else:
            k = int(sample_size * n / total)
        k = min(k, n, remaining)
        if k <= 0:
            continue

        emb = np.load(emb_f)
        idx = rng.choice(n, k, replace=False)
        samples.append(emb[idx])
        remaining -= k
        del emb

        if remaining <= 0:
            break

    sample = np.vstack(samples).astype(np.float32)
    print(f"  Training sample: {len(sample):,} vectors")
    return sample, total


def normalize_batch(emb):
    """L2-normalize for cosine similarity via inner product."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return emb / norms


def main():
    parser = argparse.ArgumentParser(description="Build FAISS IVF-PQ index")
    parser.add_argument("--nlist", type=int, default=4096, help="IVF cells")
    parser.add_argument("--pq", type=int, default=96, help="PQ subquantizers")
    parser.add_argument("--train-sample", type=int, default=500_000)
    args = parser.parse_args()

    pairs = get_file_pairs()

    # Phase 1: Load training sample (memory-efficient)
    print("\nPhase 1: Loading training sample...")
    t0 = time.time()
    train_data, total_vectors = load_training_sample(pairs, args.train_sample)
    train_data = normalize_batch(train_data)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Phase 2: Train index
    print(f"\nPhase 2: Training FAISS IVF{args.nlist},PQ{args.pq}x8...")
    quantizer = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFPQ(quantizer, DIM, args.nlist, args.pq, 8)

    t0 = time.time()
    index.train(train_data)
    print(f"  Training done ({time.time() - t0:.1f}s)")
    del train_data

    # Phase 3: Stream-add all vectors
    print(f"\nPhase 3: Adding {total_vectors:,} vectors (streaming)...")
    t0 = time.time()
    all_ids = []
    added = 0

    for i, (emb_f, meta_f) in enumerate(pairs):
        emb = np.load(emb_f).astype(np.float32)
        emb = normalize_batch(emb)
        meta = pd.read_parquet(meta_f, columns=["image_id"])

        assert len(emb) == len(meta), (
            f"Mismatch: {emb_f.name}={len(emb)}, {meta_f.name}={len(meta)}"
        )

        index.add(emb)
        all_ids.extend(meta["image_id"].tolist())
        added += len(emb)
        del emb, meta

        if (i + 1) % 20 == 0 or i == len(pairs) - 1:
            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{len(pairs)}] {added:,} / {total_vectors:,} vectors | "
                f"{elapsed:.0f}s",
                flush=True,
            )

    print(f"  Adding done ({time.time() - t0:.1f}s)")
    print(f"  Index total: {index.ntotal:,} vectors")

    # Save index
    index_path = DATA_DIR / "faiss_index.ivfpq"
    print(f"\nSaving index to {index_path}...")
    faiss.write_index(index, str(index_path))
    size_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"  Index size: {size_mb:.0f} MB")
    del index

    # Save aligned image IDs
    image_ids = np.array(all_ids)
    ids_path = DATA_DIR / "faiss_ids.npy"
    np.save(ids_path, image_ids)
    print(f"  Saved {len(image_ids):,} image IDs")

    # Build geo lookup
    print("\nBuilding geography lookup...")
    geo_path = DATA_DIR / "image_geography.parquet"
    if geo_path.exists():
        geo_df = pd.read_parquet(
            geo_path,
            columns=["image_id", "state_name", "state_abbr", "cbsa_name"],
        )
        ids_df = pd.DataFrame({"image_id": image_ids})
        lookup = ids_df.merge(geo_df, on="image_id", how="left")
        del geo_df, ids_df

        in_state = lookup["state_name"].notna().sum()
        in_msa = lookup["cbsa_name"].notna().sum()
        print(f"  {in_state:,} in US states, {in_msa:,} in MSAs")

        faiss_geo_path = DATA_DIR / "faiss_geo.parquet"
        lookup.to_parquet(faiss_geo_path, index=False)
        print(f"  Saved geo lookup: {faiss_geo_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
