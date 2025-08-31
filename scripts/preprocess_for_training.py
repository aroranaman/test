#!/usr/bin/env python3
# Manthan/scripts/preprocess_for_training.py
from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_DB_URL = os.getenv("MANTHAN_DB_URL", "sqlite:///data/manthan.db")

def _connect(db_url: str):
    return create_engine(db_url, future=True)

def _table_exists(engine, name: str) -> bool:
    with engine.connect() as c:
        r = c.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"), {"n": name}).fetchone()
    return r is not None

def export_species_catalog(engine, outdir: Path, region: Optional[str]) -> Path:
    """
    Exports a catalog of species with optional region-native flag and occurrence counts.
    """
    region = (region or "").strip().lower().replace(" ", "_") or None
    q = """
        SELECT
            s.species_key, s.canonical_name, s.scientific_name, s.family, s.genus,
            s.class_name, s.order_name, s.kingdom, s.phylum, s.iucn_category,
            COALESCE(o.num_occ, 0) AS occurrences_total,
            d.is_native AS is_native_in_region
        FROM species s
        LEFT JOIN (
            SELECT species_key, COUNT(*) AS num_occ
            FROM occurrences
            GROUP BY species_key
        ) o ON o.species_key = s.species_key
        LEFT JOIN (
            SELECT species_key, is_native
            FROM species_distribution
            {where_clause}
        ) d ON d.species_key = s.species_key
        ORDER BY occurrences_total DESC
    """
    where_clause = f"WHERE region_name = :region" if region else ""
    sql = text(q.format(where_clause=where_clause))
    params: Dict[str, Any] = {"region": region} if region else {}
    with engine.connect() as c:
        rows = c.execute(sql, params).mappings().all()
    df = pd.DataFrame(rows)
    outpath = outdir / ("species_catalog.csv" if not region else f"species_catalog_{region}.csv")
    df.to_csv(outpath, index=False)
    logging.info(f"Exported species catalog → {outpath}")
    return outpath

def export_training_examples(engine, outdir: Path, region: Optional[str]) -> Path:
    """
    Produces a simple supervised dataset:
      - feature columns from traits
      - label: is_native_in_region (1/0)
    """
    region = (region or "").strip().lower().replace(" ", "_") or None
    sql = text("""
        SELECT
            s.species_key, s.canonical_name, s.scientific_name,
            st_can.trait_value AS canopy_layer,
            st_succ.trait_value AS successional_role,
            st_dt.trait_value   AS drought_tolerance,
            st_soil.trait_value AS soil_type_preference,
            d.is_native AS is_native_in_region
        FROM species s
        LEFT JOIN species_traits st_can
               ON st_can.species_key = s.species_key AND st_can.trait_name = 'canopy_layer'
        LEFT JOIN species_traits st_succ
               ON st_succ.species_key = s.species_key AND st_succ.trait_name = 'successional_role'
        LEFT JOIN species_traits st_dt
               ON st_dt.species_key   = s.species_key AND st_dt.trait_name   = 'drought_tolerance'
        LEFT JOIN species_traits st_soil
               ON st_soil.species_key = s.species_key AND st_soil.trait_name = 'soil_type_preference'
        LEFT JOIN species_distribution d
               ON d.species_key = s.species_key {region_clause}
    """.format(region_clause="AND d.region_name = :region" if region else ""))
    params = {"region": region} if region else {}
    with engine.connect() as c:
        rows = c.execute(sql, params).mappings().all()
    df = pd.DataFrame(rows)

    # Encode label to int
    if "is_native_in_region" in df.columns:
        df["is_native_in_region"] = df["is_native_in_region"].fillna(0).astype(int)

    outpath = outdir / ("training_examples.csv" if not region else f"training_examples_{region}.csv")
    df.to_csv(outpath, index=False)
    logging.info(f"Exported training examples → {outpath}")
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Preprocess the local DB into CSVs for training (no cloud).")
    ap.add_argument("--db-url", default=DEFAULT_DB_URL, help="SQLAlchemy DB URL (sqlite:///...)")
    ap.add_argument("--out-dir", default="data/training", help="Output directory for CSVs")
    ap.add_argument("--region", default=None, help="Optional region label (e.g., 'uttar_pradesh')")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    engine = _connect(args.db_url)

    # Quick sanity
    for t in ["species", "occurrences", "species_traits"]:
        if not _table_exists(engine, t):
            logging.warning(f"Expected table '{t}' not found in DB at {args.db_url}")

    export_species_catalog(engine, outdir, args.region)
    export_training_examples(engine, outdir, args.region)

if __name__ == "__main__":
    main()
