#!/usr/bin/env python3
"""
Step 1: Extract captions, metadata, and caption embeddings from S3.

Downloads from s3://zephr-mapillary-computed-data/:
  {sequence_id}/{image_id}/caption.txt
  {sequence_id}/{image_id}/metadata.json
  {sequence_id}/{image_id}/caption_embedding.npz

Outputs batched files to data/extracts/:
  meta_{batch:06d}.parquet  - image metadata + captions
  emb_{batch:06d}.npy       - caption embeddings (N, 768) float32

Resume-safe via data/extract_checkpoint.json.

Usage:
  python pipeline/01_extract_from_s3.py                          # Full run
  python pipeline/01_extract_from_s3.py --limit-sequences 10     # Test with 10 sequences
  python pipeline/01_extract_from_s3.py --resume                 # Resume interrupted run
  python pipeline/01_extract_from_s3.py --skip-embeddings        # Captions only (for TF-IDF)
"""

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

BUCKET = "zephr-mapillary-computed-data"
BATCH_SIZE = 50_000
MAX_WORKERS = 64
BASE_DIR = Path(__file__).resolve().parent.parent


def create_s3_client(max_workers):
    config = Config(
        max_pool_connections=max_workers + 10,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    return boto3.session.Session().client("s3", config=config)


def list_sequences(s3, limit=None):
    """List all Mapillary sequence prefixes."""
    paginator = s3.get_paginator("list_objects_v2")
    sequences = []
    for page in paginator.paginate(Bucket=BUCKET, Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            prefix = p["Prefix"].rstrip("/")
            if not prefix.startswith("."):
                sequences.append(prefix)
                if limit and len(sequences) >= limit:
                    return sequences
    return sequences


def list_images_in_sequence(s3, sequence_id):
    """List image IDs within a sequence."""
    paginator = s3.get_paginator("list_objects_v2")
    images = []
    for page in paginator.paginate(
        Bucket=BUCKET, Prefix=f"{sequence_id}/", Delimiter="/"
    ):
        for p in page.get("CommonPrefixes", []):
            image_id = p["Prefix"].rstrip("/").split("/")[-1]
            images.append(image_id)
    return images


def download_one_image(s3, sequence_id, image_id, skip_embeddings=False):
    """Download metadata + caption + embedding for one image.

    Returns (record_dict, embedding_array) or None on failure.
    """
    prefix = f"{sequence_id}/{image_id}/"

    # metadata.json — required
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{prefix}metadata.json")
        meta = json.loads(obj["Body"].read())
    except Exception:
        return None

    record = {
        "image_id": image_id,
        "sequence_id": sequence_id,
        "lat": meta["geometry"]["lat"],
        "lng": meta["geometry"]["lng"],
        "compass_angle": meta.get("compass_angle", 0.0),
        "camera_type": meta.get("camera_type", ""),
        "is_pano": meta.get("is_pano", False),
        "captured_at": meta.get("captured_at", ""),
    }

    # caption.txt
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{prefix}caption.txt")
        record["caption"] = obj["Body"].read().decode("utf-8").strip()
    except Exception:
        record["caption"] = ""

    # caption_embedding.npz
    embedding = None
    if not skip_embeddings:
        try:
            obj = s3.get_object(
                Bucket=BUCKET, Key=f"{prefix}caption_embedding.npz"
            )
            data = np.load(io.BytesIO(obj["Body"].read()))
            emb = data["data"].astype(np.float32)
            if emb.shape == (768,):
                embedding = emb
        except Exception:
            pass

    return record, embedding


def save_checkpoint(path, seq_idx, batch_num, total_images):
    with open(path, "w") as f:
        json.dump(
            {
                "next_sequence_idx": seq_idx,
                "next_batch_num": batch_num,
                "total_images": total_images,
            },
            f,
        )


def flush_batch(records, embeddings, batch_num, output_dir, skip_embeddings):
    """Write one batch of results to disk."""
    df = pd.DataFrame(records)
    df.to_parquet(output_dir / f"meta_{batch_num:06d}.parquet", index=False)

    if not skip_embeddings and embeddings:
        arr = np.stack(embeddings)
        np.save(output_dir / f"emb_{batch_num:06d}.npy", arr)

    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Extract image data from S3")
    parser.add_argument(
        "--limit-sequences",
        type=int,
        default=None,
        help="Max sequences to process (for testing)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip downloading embeddings (captions only)",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = BASE_DIR / "data" / "extracts"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = BASE_DIR / "data" / "extract_checkpoint.json"

    # Resume state
    start_seq_idx = 0
    batch_num = 0
    total_images = 0
    if args.resume and checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text())
        start_seq_idx = ckpt["next_sequence_idx"]
        batch_num = ckpt["next_batch_num"]
        total_images = ckpt["total_images"]
        print(f"Resuming: seq={start_seq_idx}, batch={batch_num}, images={total_images}")

    s3 = create_s3_client(args.workers)

    print("Listing sequences...")
    sequences = list_sequences(s3, limit=args.limit_sequences)
    print(f"  Total sequences: {len(sequences)}")

    if start_seq_idx >= len(sequences):
        print("All sequences already processed.")
        return

    # Buffers for current batch
    batch_records = []
    batch_embeddings = []
    t_start = time.time()
    images_since_log = 0

    for seq_idx in range(start_seq_idx, len(sequences)):
        seq_id = sequences[seq_idx]

        # List images in this sequence
        try:
            image_ids = list_images_in_sequence(s3, seq_id)
        except Exception as e:
            print(f"  WARN: Failed to list {seq_id}: {e}")
            continue

        if not image_ids:
            continue

        # Download all images in parallel
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for img_id in image_ids:
                fut = pool.submit(
                    download_one_image,
                    s3,
                    seq_id,
                    img_id,
                    args.skip_embeddings,
                )
                futures[fut] = img_id

            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    continue
                record, embedding = result

                # Only keep images that have captions
                if not record.get("caption"):
                    continue

                # For embedding mode, only keep if embedding exists
                if not args.skip_embeddings and embedding is None:
                    continue

                batch_records.append(record)
                if embedding is not None:
                    batch_embeddings.append(embedding)

        images_since_log += len(image_ids)

        # Flush batch if full
        if len(batch_records) >= args.batch_size:
            n = flush_batch(
                batch_records,
                batch_embeddings,
                batch_num,
                output_dir,
                args.skip_embeddings,
            )
            total_images += n
            elapsed = time.time() - t_start
            rate = total_images / elapsed if elapsed > 0 else 0
            print(
                f"  Batch {batch_num}: {n} images | "
                f"Total: {total_images:,} | "
                f"Seq {seq_idx + 1}/{len(sequences)} | "
                f"{rate:.0f} img/s | "
                f"{elapsed:.0f}s"
            )
            batch_records = []
            batch_embeddings = []
            batch_num += 1
            save_checkpoint(checkpoint_path, seq_idx + 1, batch_num, total_images)

        # Progress log every 100 sequences
        elif (seq_idx - start_seq_idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            pending = len(batch_records)
            print(
                f"  Seq {seq_idx + 1}/{len(sequences)} | "
                f"Pending: {pending} | "
                f"Total: {total_images:,} | "
                f"{elapsed:.0f}s"
            )

    # Flush remaining
    if batch_records:
        n = flush_batch(
            batch_records,
            batch_embeddings,
            batch_num,
            output_dir,
            args.skip_embeddings,
        )
        total_images += n
        batch_num += 1

    save_checkpoint(checkpoint_path, len(sequences), batch_num, total_images)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Total images extracted: {total_images:,}")
    print(f"  Batches written: {batch_num}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
