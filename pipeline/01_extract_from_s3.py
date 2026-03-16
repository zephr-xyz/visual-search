#!/usr/bin/env python3
"""
Step 1: Extract captions, metadata, and caption embeddings from S3.

Two-phase approach:
  Phase 1: List all images across sequences (32 parallel threads, ~2 min)
  Phase 2: Download caption + metadata + embedding per image (200 parallel threads)

Usage:
  python pipeline/01_extract_from_s3.py --shard 0/4 &
  python pipeline/01_extract_from_s3.py --shard 1/4 &
  python pipeline/01_extract_from_s3.py --shard 2/4 &
  python pipeline/01_extract_from_s3.py --shard 3/4 &
  python pipeline/01_extract_from_s3.py --limit-sequences 10   # test
"""

import argparse
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

BUCKET = "zephr-mapillary-computed-data"
BATCH_SIZE = 50_000
BASE_DIR = Path(__file__).resolve().parent.parent

_local = threading.local()
_s3_config = None


def get_s3():
    if not hasattr(_local, "s3"):
        _local.s3 = boto3.client("s3", region_name="us-east-2", config=_s3_config)
    return _local.s3


def init_s3(max_pool):
    global _s3_config
    _s3_config = Config(
        max_pool_connections=max_pool + 10,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    return boto3.client("s3", region_name="us-east-2", config=_s3_config)


def list_sequences(s3, limit=None):
    paginator = s3.get_paginator("list_objects_v2")
    seqs = []
    for page in paginator.paginate(Bucket=BUCKET, Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            prefix = p["Prefix"].rstrip("/")
            if not prefix.startswith("."):
                seqs.append(prefix)
                if limit and len(seqs) >= limit:
                    return seqs
    return seqs


def list_images_in_sequence(seq_id):
    """List image IDs in one sequence. Uses thread-local S3 client."""
    s3 = get_s3()
    images = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{seq_id}/", Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            img_id = p["Prefix"].rstrip("/").split("/")[-1]
            images.append(img_id)
    return seq_id, images


def download_one_image(seq_id, img_id, skip_embeddings):
    """Download metadata + caption + embedding for one image."""
    s3 = get_s3()
    prefix = f"{seq_id}/{img_id}/"

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{prefix}metadata.json")
        meta = json.loads(obj["Body"].read())
    except Exception:
        return None

    record = {
        "image_id": img_id,
        "sequence_id": seq_id,
        "lat": meta["geometry"]["lat"],
        "lng": meta["geometry"]["lng"],
        "compass_angle": meta.get("compass_angle", 0.0),
        "camera_type": meta.get("camera_type", ""),
        "is_pano": meta.get("is_pano", False),
        "captured_at": meta.get("captured_at", ""),
    }

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{prefix}caption.txt")
        record["caption"] = obj["Body"].read().decode("utf-8").strip()
    except Exception:
        record["caption"] = ""

    embedding = None
    if not skip_embeddings:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=f"{prefix}caption_embedding.npz")
            data = np.load(io.BytesIO(obj["Body"].read()))
            emb = data["data"].astype(np.float32)
            if emb.shape == (768,):
                embedding = emb
        except Exception:
            pass

    if not record.get("caption"):
        return None
    if not skip_embeddings and embedding is None:
        return None

    return record, embedding


def flush_batch(records, embeddings, batch_num, output_dir, skip_emb, tag):
    df = pd.DataFrame(records)
    df.to_parquet(output_dir / f"meta_{tag}_{batch_num:06d}.parquet", index=False)
    if not skip_emb and embeddings:
        np.save(output_dir / f"emb_{tag}_{batch_num:06d}.npy", np.stack(embeddings))
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-sequences", type=int, default=None)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--shard", type=str, default=None, help="N/M")
    parser.add_argument("--download-workers", type=int, default=200)
    parser.add_argument("--list-workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    shard_idx, shard_total = 0, 1
    tag = "s0"
    if args.shard:
        shard_idx, shard_total = int(args.shard.split("/")[0]), int(args.shard.split("/")[1])
        tag = f"s{shard_idx}"

    output_dir = BASE_DIR / "data" / "extracts"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = BASE_DIR / "data" / f"extract_checkpoint_{tag}.json"

    batch_num = 0
    total_images = 0
    skip_images = set()
    if args.resume and ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        batch_num = ckpt["next_batch_num"]
        total_images = ckpt["total_images"]
        print(f"[{tag}] Resuming: batch={batch_num}, images={total_images}", flush=True)

    s3 = init_s3(args.download_workers)

    # ── Phase 1: List all sequences and images ──────────────────
    print(f"[{tag}] Listing sequences...", flush=True)
    all_seqs = list_sequences(s3, limit=args.limit_sequences)
    sequences = [seq for i, seq in enumerate(all_seqs) if i % shard_total == shard_idx]
    print(f"[{tag}] {len(sequences)} sequences in this shard", flush=True)

    print(f"[{tag}] Phase 1: Listing images in all sequences ({args.list_workers} threads)...", flush=True)
    t0 = time.time()
    all_images = []  # [(seq_id, img_id), ...]
    done = 0

    with ThreadPoolExecutor(max_workers=args.list_workers) as pool:
        futures = {pool.submit(list_images_in_sequence, seq): seq for seq in sequences}
        for fut in as_completed(futures):
            try:
                seq_id, images = fut.result()
                for img_id in images:
                    all_images.append((seq_id, img_id))
            except Exception as e:
                print(f"[{tag}] WARN: listing failed: {e}", flush=True)
            done += 1
            if done % 1000 == 0:
                print(
                    f"[{tag}]   Listed {done}/{len(sequences)} sequences, "
                    f"{len(all_images):,} images so far ({time.time()-t0:.0f}s)",
                    flush=True,
                )

    listing_time = time.time() - t0
    print(
        f"[{tag}] Phase 1 done: {len(all_images):,} images across "
        f"{len(sequences)} sequences ({listing_time:.0f}s)",
        flush=True,
    )

    if not all_images:
        print(f"[{tag}] No images found.", flush=True)
        return

    # ── Phase 2: Download all images ────────────────────────────
    print(
        f"[{tag}] Phase 2: Downloading {len(all_images):,} images "
        f"({args.download_workers} threads)...",
        flush=True,
    )
    t_start = time.time()
    batch_records = []
    batch_embeddings = []
    completed = 0
    errors = 0
    t_last_log = t_start

    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures = {
            pool.submit(download_one_image, seq_id, img_id, args.skip_embeddings): (seq_id, img_id)
            for seq_id, img_id in all_images
        }

        for fut in as_completed(futures):
            completed += 1
            try:
                result = fut.result()
            except Exception:
                errors += 1
                continue

            if result is None:
                continue

            record, embedding = result
            batch_records.append(record)
            if embedding is not None:
                batch_embeddings.append(embedding)

            # Flush batch
            if len(batch_records) >= args.batch_size:
                n = flush_batch(
                    batch_records, batch_embeddings, batch_num,
                    output_dir, args.skip_embeddings, tag,
                )
                total_images += n
                elapsed = time.time() - t_start
                rate = total_images / elapsed if elapsed > 0 else 0
                pct = completed / len(all_images) * 100
                print(
                    f"[{tag}] Batch {batch_num}: {n:,} images | "
                    f"Total: {total_images:,} | "
                    f"{completed:,}/{len(all_images):,} ({pct:.1f}%) | "
                    f"{rate:.0f} img/s | {elapsed:.0f}s",
                    flush=True,
                )
                batch_records = []
                batch_embeddings = []
                batch_num += 1
                with open(ckpt_path, "w") as f:
                    json.dump({"next_batch_num": batch_num, "total_images": total_images}, f)

            # Progress log every 30 seconds
            now = time.time()
            if now - t_last_log >= 30:
                elapsed = now - t_start
                effective = total_images + len(batch_records)
                rate = effective / elapsed if elapsed > 0 else 0
                pct = completed / len(all_images) * 100
                est_remaining = (len(all_images) - completed) / (completed / elapsed) if completed > 0 else 0
                print(
                    f"[{tag}] {effective:,} images ({len(batch_records):,} pending) | "
                    f"{completed:,}/{len(all_images):,} ({pct:.1f}%) | "
                    f"~{rate:.0f} img/s | "
                    f"errors: {errors} | "
                    f"ETA: {est_remaining/3600:.1f}h | {elapsed:.0f}s",
                    flush=True,
                )
                t_last_log = now

    # Final flush
    if batch_records:
        n = flush_batch(
            batch_records, batch_embeddings, batch_num,
            output_dir, args.skip_embeddings, tag,
        )
        total_images += n
        batch_num += 1

    with open(ckpt_path, "w") as f:
        json.dump({"next_batch_num": batch_num, "total_images": total_images}, f)

    elapsed = time.time() - t_start
    print(f"\n[{tag}] Done in {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
    print(f"  Images extracted: {total_images:,}", flush=True)
    print(f"  Batches: {batch_num}", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"  Rate: {total_images / elapsed:.0f} img/s", flush=True)


if __name__ == "__main__":
    main()
