#!/usr/bin/env python3
"""
Step 2: Reverse-geocode images to US states and MSAs (CBSAs).

Uses Census TIGER/Line shapefiles for point-in-polygon lookups.
Downloads shapefiles automatically if not present.

Input:  data/extracts/meta_*.parquet
Output: data/image_geography.parquet
          Columns: image_id, sequence_id, lat, lng, caption, captured_at,
                   state_fips, state_name, state_abbr,
                   cbsa_id, cbsa_name

Usage:
  python pipeline/02_geocode.py
  python pipeline/02_geocode.py --limit 100000   # Test with subset
"""

import argparse
import io
import os
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SHAPEFILE_DIR = DATA_DIR / "shapefiles"

# Census TIGER/Line 2023 generalized (20m) boundaries
TIGER_URLS = {
    "states": "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip",
    "cbsa": "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_cbsa_20m.zip",
}


def download_shapefile(name, url, dest_dir):
    """Download and extract a Census shapefile zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{name}.zip"
    extract_dir = dest_dir / name

    # Check if already extracted
    shp_files = list(extract_dir.glob("*.shp")) if extract_dir.exists() else []
    if shp_files:
        return shp_files[0]

    print(f"  Downloading {name} shapefile...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(resp.content)

    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()

    shp_files = list(extract_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found in {extract_dir}")
    print(f"  Extracted: {shp_files[0].name}")
    return shp_files[0]


def load_boundaries():
    """Load state and CBSA boundary GeoDataFrames."""
    print("Loading boundary shapefiles...")

    states_shp = download_shapefile("states", TIGER_URLS["states"], SHAPEFILE_DIR)
    states = gpd.read_file(states_shp)
    # Keep only relevant columns, rename for clarity
    states = states[["STATEFP", "NAME", "STUSPS", "geometry"]].rename(
        columns={"STATEFP": "state_fips", "NAME": "state_name", "STUSPS": "state_abbr"}
    )
    # Filter to 50 states + DC
    valid_fips = {
        "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
        "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
        "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
        "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
        "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56",
    }
    states = states[states["state_fips"].isin(valid_fips)].copy()
    print(f"  States: {len(states)} polygons")

    cbsa_shp = download_shapefile("cbsa", TIGER_URLS["cbsa"], SHAPEFILE_DIR)
    cbsa = gpd.read_file(cbsa_shp)
    # Metropolitan Statistical Areas only (not Micropolitan)
    cbsa = cbsa[cbsa["LSAD"] == "M1"][["CBSAFP", "NAME", "geometry"]].rename(
        columns={"CBSAFP": "cbsa_id", "NAME": "cbsa_name"}
    )
    print(f"  MSAs (Metropolitan): {len(cbsa)} polygons")

    # Reproject to EPSG:4326 to match image coordinates
    states = states.to_crs("EPSG:4326")
    cbsa = cbsa.to_crs("EPSG:4326")

    return states, cbsa


def load_extracts(limit=None):
    """Load all extracted metadata parquet files."""
    extract_dir = DATA_DIR / "extracts"
    parquet_files = sorted(extract_dir.glob("meta_*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found in {extract_dir}")
        raise SystemExit(1)

    print(f"Loading {len(parquet_files)} parquet files...")
    dfs = []
    total = 0
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if limit and total + len(df) > limit:
            df = df.head(limit - total)
        dfs.append(df)
        total += len(df)
        if limit and total >= limit:
            break

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(combined):,} images")
    return combined


def geocode(df, states_gdf, cbsa_gdf):
    """Assign each image to a state and MSA via spatial join."""
    print("Creating point geometries...")
    geometry = gpd.points_from_xy(df["lng"], df["lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Spatial join with states
    print("Joining with state boundaries...")
    gdf_states = gpd.sjoin(gdf, states_gdf, how="left", predicate="within")
    gdf_states = gdf_states.drop(columns=["index_right"], errors="ignore")

    # Spatial join with CBSAs
    print("Joining with MSA boundaries...")
    gdf_cbsa = gpd.sjoin(gdf, cbsa_gdf, how="left", predicate="within")
    gdf_cbsa = gdf_cbsa[["image_id", "cbsa_id", "cbsa_name"]].drop_duplicates(
        subset=["image_id"], keep="first"
    )

    # Merge CBSA results back
    result = gdf_states.merge(gdf_cbsa, on="image_id", how="left")

    # Drop geometry column for parquet output
    result = pd.DataFrame(result.drop(columns=["geometry"]))

    # Report coverage
    in_state = result["state_name"].notna().sum()
    in_msa = result["cbsa_name"].notna().sum()
    print(f"  In a US state: {in_state:,} / {len(result):,} ({100*in_state/len(result):.1f}%)")
    print(f"  In an MSA:     {in_msa:,} / {len(result):,} ({100*in_msa/len(result):.1f}%)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Geocode images to state/MSA")
    parser.add_argument("--limit", type=int, help="Limit images for testing")
    args = parser.parse_args()

    states_gdf, cbsa_gdf = load_boundaries()
    df = load_extracts(limit=args.limit)
    result = geocode(df, states_gdf, cbsa_gdf)

    out_path = DATA_DIR / "image_geography.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nSaved {len(result):,} rows to {out_path}")

    # Summary by state
    print("\nTop 10 states by image count:")
    top = result.groupby("state_name").size().sort_values(ascending=False).head(10)
    for state, count in top.items():
        print(f"  {state:20s} {count:>10,}")


if __name__ == "__main__":
    main()
