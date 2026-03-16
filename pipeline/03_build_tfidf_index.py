#!/usr/bin/env python3
"""
Step 3: Build a stemmed keyword prevalence index from captions.

For each (stemmed_term, geography) pair, stores the count of images whose
caption contains that term.  At query time, prevalence = count / total_images.

Also builds a TF-IDF matrix where each "document" is a geography (all
captions concatenated), enabling TF-IDF-weighted ranking of geographies.

Input:  data/image_geography.parquet
Output: data/keyword_index.pkl        - {term: {geo: count}} dict
        data/tfidf_model.pkl          - fitted TfidfVectorizer + matrix
        data/geography_stats.json     - image counts per geography

Usage:
  python pipeline/03_build_tfidf_index.py
  python pipeline/03_build_tfidf_index.py --geo-level state   # states only
  python pipeline/03_build_tfidf_index.py --geo-level msa     # MSAs only
  python pipeline/03_build_tfidf_index.py --geo-level both    # default
"""

import argparse
import json
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


def build_keyword_index(df, geo_col):
    """Build inverted index: stemmed_term -> {geography: image_count}.

    Also returns per-geography total image counts.
    """
    print(f"  Building keyword index for {geo_col}...")
    t0 = time.time()

    # term -> geo -> count
    index = defaultdict(lambda: defaultdict(int))
    geo_totals = defaultdict(int)

    for geo_name, caption in zip(df[geo_col], df["caption"]):
        if pd.isna(geo_name) or not caption:
            continue
        geo_totals[geo_name] += 1
        # Deduplicate tokens per image (presence, not frequency)
        seen = set()
        for token in stem_tokenize(caption):
            if token not in seen:
                index[token][geo_name] += 1
                seen.add(token)

    # Convert to regular dicts for pickling
    index = {term: dict(geos) for term, geos in index.items()}
    geo_totals = dict(geo_totals)

    elapsed = time.time() - t0
    print(f"    {len(index):,} unique terms, {len(geo_totals)} geographies ({elapsed:.1f}s)")
    return index, geo_totals


def build_tfidf_matrix(df, geo_col):
    """Build TF-IDF matrix where each row is a geography.

    Concatenates all captions in each geography into one document,
    then fits a TfidfVectorizer with stemming and bigrams.
    """
    print(f"  Building TF-IDF matrix for {geo_col}...")
    t0 = time.time()

    # Concatenate captions per geography
    geo_docs = {}
    for geo_name, caption in zip(df[geo_col], df["caption"]):
        if pd.isna(geo_name) or not caption:
            continue
        if geo_name not in geo_docs:
            geo_docs[geo_name] = []
        geo_docs[geo_name].append(caption)

    geo_names = sorted(geo_docs.keys())
    # Pre-stem documents so vectorizer uses default (picklable) tokenizer
    documents = [" ".join(stem_tokenize(" ".join(geo_docs[g]))) for g in geo_names]

    vectorizer = TfidfVectorizer(
        token_pattern=r"[a-z]{2,}",  # match pre-stemmed tokens
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=100_000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    elapsed = time.time() - t0
    vocab_size = len(vectorizer.vocabulary_)
    print(
        f"    Matrix: {tfidf_matrix.shape[0]} geographies x {vocab_size:,} terms "
        f"({elapsed:.1f}s)"
    )
    return vectorizer, tfidf_matrix, geo_names


def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF keyword index")
    parser.add_argument(
        "--geo-level",
        choices=["state", "msa", "both"],
        default="both",
    )
    args = parser.parse_args()

    geo_path = DATA_DIR / "image_geography.parquet"
    if not geo_path.exists():
        print(f"ERROR: {geo_path} not found. Run 02_geocode.py first.")
        raise SystemExit(1)

    print("Loading geocoded data...")
    df = pd.read_parquet(geo_path, columns=["image_id", "caption", "state_name", "cbsa_name"])
    print(f"  {len(df):,} images")

    results = {}

    levels = []
    if args.geo_level in ("state", "both"):
        levels.append(("state", "state_name"))
    if args.geo_level in ("msa", "both"):
        levels.append(("msa", "cbsa_name"))

    for level_name, geo_col in levels:
        print(f"\n=== {level_name.upper()} level ===")

        kw_index, geo_totals = build_keyword_index(df, geo_col)
        vectorizer, tfidf_matrix, geo_names = build_tfidf_matrix(df, geo_col)

        # Save keyword index
        kw_path = DATA_DIR / f"keyword_index_{level_name}.pkl"
        with open(kw_path, "wb") as f:
            pickle.dump({"index": kw_index, "geo_totals": geo_totals}, f)
        print(f"  Saved keyword index: {kw_path}")

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

        results[level_name] = geo_totals

    # Save geography stats
    stats_path = DATA_DIR / "geography_stats.json"
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved geography stats: {stats_path}")


if __name__ == "__main__":
    main()
