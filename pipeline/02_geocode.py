#!/usr/bin/env python3
"""
Step 2: Reverse-geocode images to US states and MSAs (CBSAs),
and assign Web Mercator tile coordinates at z8, z10, z12, z14.

Uses Census TIGER/Line shapefiles for point-in-polygon lookups.
Downloads shapefiles automatically if not present.

Input:  data/extracts/meta_*.parquet
Output: data/image_geography.parquet
          Columns: image_id, sequence_id, lat, lng, caption, captured_at,
                   state_fips, state_name, state_abbr,
                   cbsa_id, cbsa_name,
                   z8_tile, z10_tile, z12_tile, z14_tile

Usage:
  python pipeline/02_geocode.py
  python pipeline/02_geocode.py --limit 100000   # Test with subset
"""

import argparse
import io
import math
import os
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
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


def lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to Web Mercator tile (x, y) at given zoom level."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def compute_tile_keys(lats, lngs, zoom):
    """Vectorized tile key computation for arrays of lat/lng.

    Returns array of strings like 'z10/512/345'.
    """
    lat_rad = np.radians(lats)
    n = 2 ** zoom
    x = ((lngs + 180.0) / 360.0 * n).astype(np.int32)
    y = ((1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n).astype(np.int32)
    x = np.clip(x, 0, n - 1)
    y = np.clip(y, 0, n - 1)
    return pd.array([f"z{zoom}/{xi}/{yi}" for xi, yi in zip(x, y)], dtype="string")


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


def geocode_file(parquet_path, states_gdf, cbsa_gdf, output_path):
    """Geocode a single parquet file and write result to output_path.

    Loads only the columns needed for geocoding (no captions) to minimize
    memory usage. Captions are joined back from the original file at the end.
    """
    # Load lightweight columns for spatial join
    geo_cols = ["image_id", "sequence_id", "lat", "lng"]
    df_geo = pd.read_parquet(parquet_path, columns=geo_cols)

    # Compute tile coordinates
    lats = df_geo["lat"].values.astype(np.float64)
    lngs = df_geo["lng"].values.astype(np.float64)
    for zoom in (8, 10, 12, 14):
        df_geo[f"z{zoom}_tile"] = compute_tile_keys(lats, lngs, zoom)

    # Spatial join with states
    geometry = gpd.points_from_xy(df_geo["lng"], df_geo["lat"])
    gdf = gpd.GeoDataFrame(df_geo, geometry=geometry, crs="EPSG:4326")
    del df_geo

    gdf_states = gpd.sjoin(gdf, states_gdf, how="left", predicate="within")
    gdf_states = gdf_states.drop(columns=["index_right"], errors="ignore")

    # Spatial join with CBSAs
    gdf_cbsa = gpd.sjoin(gdf, cbsa_gdf, how="left", predicate="within")
    gdf_cbsa = gdf_cbsa[["image_id", "cbsa_id", "cbsa_name"]].drop_duplicates(
        subset=["image_id"], keep="first"
    )
    del gdf

    # Merge state + CBSA results
    result = pd.DataFrame(gdf_states.drop(columns=["geometry"]))
    result = result.merge(gdf_cbsa, on="image_id", how="left")
    del gdf_states, gdf_cbsa

    # Join back captions and other columns from original file
    extra_cols = ["image_id", "caption", "captured_at", "compass_angle",
                  "camera_type", "is_pano"]
    available = pd.read_parquet(parquet_path, columns=["image_id"]).columns.tolist()
    # Read available extra columns
    import pyarrow.parquet as pq
    schema_cols = pq.read_schema(parquet_path).names
    load_extra = [c for c in extra_cols if c in schema_cols]
    df_extra = pd.read_parquet(parquet_path, columns=load_extra)
    result = result.merge(df_extra, on="image_id", how="left")
    del df_extra

    result.to_parquet(output_path, index=False)
    return len(result)


def main():
    parser = argparse.ArgumentParser(description="Geocode images to state/MSA")
    parser.add_argument("--limit", type=int, help="Limit files for testing")
    args = parser.parse_args()

    states_gdf, cbsa_gdf = load_boundaries()

    extract_dir = DATA_DIR / "extracts"
    parquet_files = sorted(extract_dir.glob("meta_*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found in {extract_dir}")
        raise SystemExit(1)

    if args.limit:
        parquet_files = parquet_files[: args.limit]

    geo_dir = DATA_DIR / "geocoded"
    geo_dir.mkdir(parents=True, exist_ok=True)

    print(f"Geocoding {len(parquet_files)} parquet files...")
    import time
    t0 = time.time()
    total_images = 0

    for i, pf in enumerate(parquet_files):
        out_name = pf.name.replace("meta_", "geo_")
        out_path = geo_dir / out_name
        if out_path.exists():
            n = len(pd.read_parquet(out_path, columns=["image_id"]))
            total_images += n
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(parquet_files)}] {pf.name} -> skipped (exists, {n:,} rows)", flush=True)
            continue

        n = geocode_file(pf, states_gdf, cbsa_gdf, out_path)
        total_images += n
        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(parquet_files)}] {pf.name} -> {n:,} rows | "
                f"total: {total_images:,} | {elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\nGeocoded {total_images:,} images in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Merge into single output file
    print("Merging geocoded files...")
    geo_files = sorted(geo_dir.glob("geo_*.parquet"))
    out_path = DATA_DIR / "image_geography.parquet"

    # Stream-merge to avoid loading all into memory at once
    import pyarrow.parquet as pq
    writer = None
    for gf in geo_files:
        table = pq.read_table(gf)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
    if writer:
        writer.close()

    print(f"Saved {out_path}")

    # Summary by state (read just state column)
    states = pd.read_parquet(out_path, columns=["state_name"])
    print(f"\nTotal: {len(states):,} images")
    print("\nTop 10 states by image count:")
    top = states["state_name"].value_counts().head(10)
    for state, count in top.items():
        print(f"  {state:20s} {count:>10,}")


if __name__ == "__main__":
    main()
