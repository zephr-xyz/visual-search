#!/usr/bin/env python3
"""
Step 1: Extract captions, metadata, and caption embeddings from S3.

Downloads from s3://zephr-mapillary-computed-data/:
  {sequence_id}/{image_id}/caption.txt
  {sequence_id}/{image_id}/metadata.json
  {sequence_id}/{image_id}/caption_embedding.npz

Uses a pipelined architecture: sequence listing and image downloads run
concurrently in a single thread pool for maximum throughput.

Outputs batched files to data/extracts/:
  meta_{shard}_{batch:06d}.parquet  - image metadata + captions
  emb_{shard}_{batch:06d}.npy       - caption embeddings (N, 768) float32

Usage:
  # Sharded parallel run (4 processes)
  python pipeline/01_extract_from_s3.py --shard 0/4 &
  python pipeline/01_extract_from_s3.py --shard 1/4 &
  python pipeline/01_extract_from_s3.py --shard 2/4 &
  python pipeline/01_extract_from_s3.py --shard 3/4 &

  # Test
  python pipeline/01_extract_from_s3.py --limit-sequences 10
"""

import argparse
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue, Empty

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

BUCKET = "zephr-mapillary-computed-data"
BATCH_SIZE = 50_000
MAX_WORKERS = 200
BASE_DIR = Path(__file__).resolve().parent.parent

# Thread-local S3 clients
_local = threading.local()
_s3_config = None


def get_s3():
    if not hasattr(_local, "s3"):
        _local.s3 = boto3.client("s3", region_name="us-east-2", config=_s3_config)
    return _local.s3


def init_s3(max_workers):
    global _s3_config
    _s3_config = Config(
        max_pool_connections=max_workers + 10,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    return boto3.client("s3", region_name="us-east-2", config=_s3_config)


def list_sequences(s3, limit=None):
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


def list_and_submit_images(seq_id, pool, skip_embeddings, result_queue, counter):
    """List images in a sequence and submit download tasks.

    This runs inside the thread pool — listing and downloads overlap.
    """
    s3 = get_s3()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=BUCKET, Prefix=f"{seq_id}/", Delimiter="/"
        ):
            for p in page.get("CommonPrefixes", []):
                img_id = p["Prefix"].rstrip("/").split("/")[-1]
                fut = pool.submit(download_one_image, seq_id, img_id, skip_embeddings)
                fut.add_done_callback(lambda f, q=result_queue, c=counter: _on_done(f, q, c))
    except Exception as e:
        print(f"  WARN: list failed for {seq_id}: {e}", flush=True)


def _on_done(future, result_queue, counter):
    """Callback when a download completes."""
    try:
        result = future.result()
        if result is not None:
            result_queue.put(result)
        counter["submitted"] += 1
    except Exception:
        counter["errors"] += 1


def download_one_image(sequence_id, image_id, skip_embeddings=False):
    """Download metadata + caption + embedding for one image."""
    s3 = get_s3()
    prefix = f"{sequence_id}/{image_id}/"

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

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{prefix}caption.txt")
        record["caption"] = obj["Body"].read().decode("utf-8").strip()
    except Exception:
        record["caption"] = ""

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

    if not record.get("caption"):
        return None
    if not skip_embeddings and embedding is None:
        return None

    return record, embedding


def flush_batch(records, embeddings, batch_num, output_dir, skip_embeddings, shard_tag):
    df = pd.DataFrame(records)
    df.to_parquet(output_dir / f"meta_{shard_tag}_{batch_num:06d}.parquet", index=False)
    if not skip_embeddings and embeddings:
        np.save(output_dir / f"emb_{shard_tag}_{batch_num:06d}.npy", np.stack(embeddings))
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Extract image data from S3")
    parser.add_argument("--limit-sequences", type=int, default=None)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--shard", type=str, default=None, help="N/M shard spec")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    shard_idx, shard_total = 0, 1
    shard_tag = "s0"
    if args.shard:
        shard_idx, shard_total = int(args.shard.split("/")[0]), int(args.shard.split("/")[1])
        shard_tag = f"s{shard_idx}"

    output_dir = BASE_DIR / "data" / "extracts"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = BASE_DIR / "data" / f"extract_checkpoint_{shard_tag}.json"

    # Resume
    batch_num = 0
    total_images = 0
    start_seq_idx = 0
    if args.resume and checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text())
        start_seq_idx = ckpt["next_sequence_idx"]
        batch_num = ckpt["next_batch_num"]
        total_images = ckpt["total_images"]
        print(f"Resuming: seq={start_seq_idx}, batch={batch_num}, images={total_images}", flush=True)

    s3 = init_s3(args.workers)

    print(f"[{shard_tag}] Listing sequences...", flush=True)
    all_sequences = list_sequences(s3, limit=args.limit_sequences)
    sequences = [seq for i, seq in enumerate(all_sequences) if i % shard_total == shard_idx]
    sequences = sequences[start_seq_idx:]
    print(f"[{shard_tag}] {len(sequences)} sequences to process", flush=True)

    if not sequences:
        print("Nothing to do.", flush=True)
        return

    # Result queue for processed images
    result_queue = Queue(maxsize=args.batch_size * 2)
    counter = {"submitted": 0, "errors": 0}

    batch_records = []
    batch_embeddings = []
    t_start = time.time()
    t_last_log = t_start

    # Use a single thread pool for both listing and downloading.
    # Submit sequence listings (which internally submit downloads).
    pool = ThreadPoolExecutor(max_workers=args.workers)

    # Submit all sequence listings — they'll run concurrently and
    # internally submit download tasks as images are discovered.
    listing_futures = []
    for seq_id in sequences:
        fut = pool.submit(
            list_and_submit_images,
            seq_id, pool, args.skip_embeddings, result_queue, counter,
        )
        listing_futures.append(fut)

    print(f"[{shard_tag}] Submitted {len(listing_futures)} sequence listings", flush=True)

    # Collect results from the queue
    done_listings = 0
    all_listings_done = False

    while True:
        # Check if all listing futures are done
        if not all_listings_done:
            done_count = sum(1 for f in listing_futures if f.done())
            if done_count == len(listing_futures):
                all_listings_done = True
                print(
                    f"[{shard_tag}] All {len(listing_futures)} sequence listings complete, "
                    f"draining downloads...",
                    flush=True,
                )

        # Drain the result queue
        got_any = False
        while True:
            try:
                result = result_queue.get(timeout=0.5)
                got_any = True
            except Empty:
                break

            record, embedding = result
            batch_records.append(record)
            if embedding is not None:
                batch_embeddings.append(embedding)

            # Flush batch if full
            if len(batch_records) >= args.batch_size:
                n = flush_batch(
                    batch_records, batch_embeddings, batch_num,
                    output_dir, args.skip_embeddings, shard_tag,
                )
                total_images += n
                elapsed = time.time() - t_start
                rate = total_images / elapsed if elapsed > 0 else 0
                print(
                    f"[{shard_tag}] Batch {batch_num}: {n:,} images | "
                    f"Total: {total_images:,} | "
                    f"{rate:.0f} img/s | {elapsed:.0f}s",
                    flush=True,
                )
                batch_records = []
                batch_embeddings = []
                batch_num += 1

                ckpt_seq = start_seq_idx + done_count if not all_listings_done else len(sequences)
                with open(checkpoint_path, "w") as f:
                    json.dump({
                        "next_sequence_idx": ckpt_seq,
                        "next_batch_num": batch_num,
                        "total_images": total_images,
                    }, f)

        # Progress logging every 30 seconds
        now = time.time()
        if now - t_last_log >= 30:
            pending = len(batch_records)
            elapsed = now - t_start
            effective = total_images + pending
            rate = effective / elapsed if elapsed > 0 else 0
            print(
                f"[{shard_tag}] Progress: {effective:,} images "
                f"({pending:,} pending) | "
                f"~{rate:.0f} img/s | "
                f"errors: {counter['errors']} | "
                f"{elapsed:.0f}s",
                flush=True,
            )
            t_last_log = now

        # Exit condition: all listings done AND queue is empty AND no pending downloads
        if all_listings_done and result_queue.empty() and not got_any:
            # Wait a bit more for stragglers
            time.sleep(2)
            if result_queue.empty():
                break

    pool.shutdown(wait=True)

    # Flush remaining
    if batch_records:
        n = flush_batch(
            batch_records, batch_embeddings, batch_num,
            output_dir, args.skip_embeddings, shard_tag,
        )
        total_images += n
        batch_num += 1

    with open(checkpoint_path, "w") as f:
        json.dump({
            "next_sequence_idx": start_seq_idx + len(sequences),
            "next_batch_num": batch_num,
            "total_images": total_images,
        }, f)

    elapsed = time.time() - t_start
    print(f"\n[{shard_tag}] Done in {elapsed:.0f}s", flush=True)
    print(f"  Total images: {total_images:,}", flush=True)
    print(f"  Batches: {batch_num}", flush=True)
    print(f"  Errors: {counter['errors']}", flush=True)
    print(f"  Rate: {total_images / elapsed:.0f} img/s", flush=True)


if __name__ == "__main__":
    main()
