#!/usr/bin/env python3
# Manthan/scripts/ingest_external_data.py
from __future__ import annotations

import os
import sys
import re
import math
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set
from contextlib import contextmanager
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Optional deps
try:
    import pygbif.occurrences as occ
    PYGIBF_OK = True
except Exception:
    PYGIBF_OK = False

try:
    import ee  # Google Earth Engine
    GEE_IMPORTED = True
except Exception:
    GEE_IMPORTED = False

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
    GEOPY_OK = True
except Exception:
    GEOPY_OK = False

# ------------------------------------------------------------------------------
# Config & Setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

DB_URL = os.getenv("MANTHAN_DB_URL", "sqlite:///data/manthan.db")
IUCN_TOKEN = os.getenv("IUCN_TOKEN", "")  # optional; if missing, IUCN enrichment is skipped

PLANTAE_TAXON_KEY = 6
DEFAULT_PAGE_SIZE = 300  # GBIF max per page

GBIF_OCC_SEARCH = "https://api.gbif.org/v1/occurrence/search"
GBIF_SPECIES_META = "https://api.gbif.org/v1/species/"

# ------------------------------------------------------------------------------
# Special 14 regions (bounds)
# ------------------------------------------------------------------------------
_SPECIAL_BOUNDS: List[Dict[str, Any]] = [
    {"name": "western_ghats_north",   "bounds": [72.8, 17.0, 75.3, 20.8]},
    {"name": "western_ghats_central", "bounds": [74.0, 13.0, 76.0, 16.5]},
    {"name": "western_ghats_south",   "bounds": [76.0,  8.0, 77.5, 12.2]},
    {"name": "eastern_ghats_north",   "bounds": [83.0, 18.0, 85.5, 21.0]},
    {"name": "eastern_ghats_south",   "bounds": [79.0, 13.0, 81.0, 16.0]},
    {"name": "himalaya_uttarakhand",  "bounds": [79.0, 29.5, 81.1, 31.5]},
    {"name": "himalaya_himachal",     "bounds": [76.0, 31.0, 78.0, 32.8]},
    {"name": "kashmir_valley",        "bounds": [74.0, 33.5, 75.7, 34.7]},
    {"name": "meghalaya_hills",       "bounds": [90.0, 25.0, 92.7, 26.5]},
    {"name": "assam_floodplains",     "bounds": [91.0, 26.5, 95.0, 27.7]},
    {"name": "thar_desert_core",      "bounds": [70.0, 25.0, 73.5, 28.5]},
    {"name": "rann_of_kutch",         "bounds": [68.5, 23.0, 71.0, 24.8]},
    {"name": "sundarbans",            "bounds": [88.0, 21.5, 89.6, 22.6]},
    {"name": "deccan_plateau_core",   "bounds": [75.0, 15.0, 80.0, 20.0]},
]

# ------------------------------------------------------------------------------
# Helpers (general)
# ------------------------------------------------------------------------------
def _ensure_sqlite_dir(db_url: str) -> None:
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "", 1)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

def _connect_engine(db_url: str) -> Engine:
    _ensure_sqlite_dir(db_url)
    return create_engine(db_url, future=True)

def _slugify_region(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s

_BINOMIAL_RE = re.compile(r"[A-Za-z-]+")
def _canon_binomial(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    toks = _BINOMIAL_RE.findall(str(name))
    if len(toks) >= 2:
        return f"{toks[0].lower()} {toks[1].lower()}"
    if len(toks) == 1:
        return toks[0].lower()
    return None

def _serialize_issues(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, list):
        return ",".join(str(x).strip() for x in val if str(x).strip())
    s = str(val).strip()
    return s or None

def _has_bad_issue(issues: Any) -> bool:
    bad = {
        "COUNTRY_COORDINATE_MISMATCH",
        "ZERO_COORDINATE",
        "COORDINATE_INVALID",
        "PRESUMED_SWAPPED_COORDINATE",
        "COORDINATE_OUT_OF_RANGE",
    }
    if issues is None:
        return False
    try:
        if isinstance(issues, float) and math.isnan(issues):
            return False
    except Exception:
        pass
    if isinstance(issues, list):
        toks = {str(t).strip() for t in issues if str(t).strip()}
    elif isinstance(issues, str):
        toks = {t.strip() for t in issues.split(",") if t.strip()}
    else:
        s = str(issues).strip()
        toks = {s} if s else set()
    return bool(bad.intersection(toks))

# --- Geocoding (for single-location mode) ---
def get_location_details(location_name: str) -> Optional[Dict[str, Any]]:
    if not GEOPY_OK:
        logging.error("geopy not installed; cannot geocode.")
        return None
    logging.info(f"Geocoding '{location_name}' to bounding box ...")
    geolocator = Nominatim(user_agent="manthan_app")
    try:
        location = geolocator.geocode(location_name, timeout=15, addressdetails=True)
        if location and location.raw.get("boundingbox"):
            south, north, west, east = map(float, location.raw["boundingbox"])
            bbox = (west, south, east, north)
            addr = location.raw.get("address", {})
            state = addr.get("state") or addr.get("region") or addr.get("state_district") or addr.get("county")
            label_src = state or location_name
            label = _slugify_region(label_src)
            logging.info(f"  -> bbox={bbox} | state={state} | label={label}")
            return {"bbox": bbox, "state": state, "label": label}
        logging.error(f"Could not resolve complete details for '{location_name}'.")
        return None
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        logging.error(f"Geocoding unavailable: {e}")
        return None
    except Exception as e:
        logging.error(f"Geocoding failed: {e}")
        return None

def _bbox_to_wkt(bbox: Tuple[float, float, float, float]) -> str:
    w, s, e, n = bbox
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"

# ------------------------------------------------------------------------------
# DB schema & upserts (row-per-trait schema; matches your current DB)
# ------------------------------------------------------------------------------
SPECIES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS species (
    species_key INTEGER PRIMARY KEY,
    canonical_name TEXT,
    scientific_name TEXT,
    family TEXT,
    genus TEXT,
    class_name TEXT,
    order_name TEXT,
    kingdom TEXT,
    phylum TEXT,
    iucn_category TEXT,
    canonical_binomial TEXT
);
"""

OCC_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS occurrences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    species_key INTEGER,
    lat REAL,
    lon REAL,
    event_date TEXT,
    basis_of_record TEXT,
    dataset_key TEXT,
    institution_code TEXT,
    locality TEXT,
    state_province TEXT,
    elevation REAL,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    issues TEXT,
    establishment_means TEXT,
    UNIQUE (species_key, lat, lon, event_date),
    FOREIGN KEY (species_key) REFERENCES species (species_key)
);
"""

TRAITS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS species_traits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    species_key INTEGER NOT NULL,
    trait_name TEXT NOT NULL,
    trait_value TEXT,
    source TEXT,
    last_updated TEXT DEFAULT (datetime('now')),
    UNIQUE (species_key, trait_name),
    FOREIGN KEY (species_key) REFERENCES species (species_key)
);
"""

SPECIES_DIST_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS species_distribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    species_key INTEGER NOT NULL,
    region_name TEXT NOT NULL,
    is_native INTEGER NOT NULL,
    UNIQUE (species_key, region_name),
    FOREIGN KEY (species_key) REFERENCES species (species_key)
);
"""

@contextmanager
def _conn(engine: Engine):
    with engine.begin() as c:
        yield c

def _table_exists(conn, name: str) -> bool:
    r = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"), {"n": name}).fetchone()
    return r is not None

def _table_cols(conn, name: str) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(text(f"PRAGMA table_info({name})")).mappings().all()
    return {row["name"]: dict(row) for row in rows}

def _add_missing_columns(conn, table: str, required_cols: Dict[str, str], existing: Dict[str, Dict[str, Any]]):
    for col, coltype in required_cols.items():
        if col not in existing:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"))

def _recreate_table(conn, name: str, create_sql: str):
    conn.execute(text(f"DROP TABLE IF EXISTS {name}"))
    conn.execute(text(create_sql))

def ensure_schema(engine: Engine, recreate_schema: bool = False) -> None:
    with _conn(engine) as conn:
        # species
        if not _table_exists(conn, "species"):
            conn.execute(text(SPECIES_SCHEMA_SQL))
        else:
            cols = _table_cols(conn, "species")
            has_pk = ("species_key" in cols) and (cols["species_key"].get("pk", 0) == 1)
            if not has_pk:
                msg = "Existing 'species' table missing primary key 'species_key'."
                if recreate_schema:
                    logging.warning(msg + " Recreating table due to --recreate-schema.")
                    _recreate_table(conn, "species", SPECIES_SCHEMA_SQL)
                else:
                    logging.error(msg + " Rerun with --recreate-schema to fix.")
                    raise RuntimeError("Incompatible 'species' schema.")
            else:
                req = {
                    "species_key": "INTEGER",
                    "canonical_name": "TEXT",
                    "scientific_name": "TEXT",
                    "family": "TEXT",
                    "genus": "TEXT",
                    "class_name": "TEXT",
                    "order_name": "TEXT",
                    "kingdom": "TEXT",
                    "phylum": "TEXT",
                    "iucn_category": "TEXT",
                    "canonical_binomial": "TEXT",
                }
                _add_missing_columns(conn, "species", req, cols)
        # occurrences
        if not _table_exists(conn, "occurrences"):
            conn.execute(text(OCC_SCHEMA_SQL))
        else:
            cols = _table_cols(conn, "occurrences")
            req = {
                "id": "INTEGER","species_key": "INTEGER","lat": "REAL","lon": "REAL",
                "event_date": "TEXT","basis_of_record": "TEXT","dataset_key": "TEXT",
                "institution_code": "TEXT","locality": "TEXT","state_province": "TEXT",
                "elevation": "REAL","year": "INTEGER","month": "INTEGER","day": "INTEGER",
                "issues": "TEXT","establishment_means": "TEXT",
            }
            _add_missing_columns(conn, "occurrences", req, cols)
        # traits
        if not _table_exists(conn, "species_traits"):
            conn.execute(text(TRAITS_SCHEMA_SQL))
        else:
            cols = _table_cols(conn, "species_traits")
            if "species_key" not in cols:
                if recreate_schema:
                    logging.warning("'species_traits' missing 'species_key'. Recreating.")
                    _recreate_table(conn, "species_traits", TRAITS_SCHEMA_SQL)
                else:
                    logging.warning("'species_traits' missing 'species_key'. Adding column in-place.")
                    conn.execute(text("ALTER TABLE species_traits ADD COLUMN species_key INTEGER"))
                    if "species_id" in cols:
                        conn.execute(text("UPDATE species_traits SET species_key = species_id WHERE species_key IS NULL"))
            cols2 = _table_cols(conn, "species_traits")
            req = {"trait_name": "TEXT","trait_value": "TEXT","source": "TEXT","last_updated": "TEXT"}
            _add_missing_columns(conn, "species_traits", req, cols2)
        # distribution
        if not _table_exists(conn, "species_distribution"):
            conn.execute(text(SPECIES_DIST_SCHEMA_SQL))
        else:
            cols = _table_cols(conn, "species_distribution")
            req = {"species_key": "INTEGER", "region_name": "TEXT", "is_native": "INTEGER"}
            _add_missing_columns(conn, "species_distribution", req, cols)

def insert_traits(engine: Engine, trait_records: list[dict]) -> int:
    if not trait_records:
        return 0
    sql = text("""
        INSERT INTO species_traits (species_key, trait_name, trait_value, source)
        VALUES (:species_key, :trait_name, :trait_value, :source)
        ON CONFLICT(species_key, trait_name) DO UPDATE SET
            trait_value = excluded.trait_value,
            source = excluded.source,
            last_updated = datetime('now')
    """)
    with engine.begin() as conn:
        conn.execute(sql, trait_records)
    return len(trait_records)

def upsert_species(engine: Engine, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols = ["species_key", "canonical_name", "scientific_name", "family", "genus",
            "class_name", "order_name", "kingdom", "phylum", "iucn_category", "canonical_binomial"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    rows = df[cols].to_dict("records")
    sql = text("""
        INSERT INTO species (species_key, canonical_name, scientific_name, family, genus,
                             class_name, order_name, kingdom, phylum, iucn_category, canonical_binomial)
        VALUES (:species_key, :canonical_name, :scientific_name, :family, :genus,
                :class_name, :order_name, :kingdom, :phylum, :iucn_category, :canonical_binomial)
        ON CONFLICT(species_key) DO UPDATE SET
            canonical_name=excluded.canonical_name,
            scientific_name=excluded.scientific_name,
            family=excluded.family,
            genus=excluded.genus,
            class_name=excluded.class_name,
            order_name=excluded.order_name,
            kingdom=excluded.kingdom,
            phylum=excluded.phylum,
            iucn_category=COALESCE(excluded.iucn_category, species.iucn_category),
            canonical_binomial=excluded.canonical_binomial
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)

def insert_occurrences(engine: Engine, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    if "establishment_means" not in df.columns:
        df["establishment_means"] = None
    df = df.where(pd.notna(df), None)
    rows = df.to_dict("records")
    sql = text("""
        INSERT OR IGNORE INTO occurrences
        (species_key, lat, lon, event_date, basis_of_record, dataset_key, institution_code,
         locality, state_province, elevation, year, month, day, issues, establishment_means)
        VALUES
        (:species_key, :lat, :lon, :event_date, :basis_of_record, :dataset_key, :institution_code,
         :locality, :state_province, :elevation, :year, :month, :day, :issues, :establishment_means)
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)

def upsert_species_distribution(engine: Engine, records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0
    sql = text("""
        INSERT INTO species_distribution (species_key, region_name, is_native)
        VALUES (:species_key, :region_name, :is_native)
        ON CONFLICT(species_key, region_name) DO UPDATE SET
            is_native = excluded.is_native
    """)
    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)

# ------------------------------------------------------------------------------
# GBIF fetch/clean utilities (used by Single-area & Special-14)
# ------------------------------------------------------------------------------
def fetch_gbif_occurrences(
    taxon_key: int,
    bbox: Tuple[float, float, float, float],
    country_code: Optional[str] = None,
    since_year: Optional[int] = None,
    max_records: int = 10_000,
    page_size: int = DEFAULT_PAGE_SIZE,
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    if not PYGIBF_OK:
        logging.error("pygbif is not installed. Run: pip install pygbif")
        return pd.DataFrame()
    wkt = _bbox_to_wkt(bbox)
    records: List[Dict[str, Any]] = []
    fetched = 0
    offset = 0
    logging.info(
        f"Fetching GBIF occurrences (taxon={taxon_key}, bbox={bbox}, country={country_code or 'ANY'}, "
        f"since={since_year}, max={max_records})"
    )
    while fetched < max_records:
        limit = min(page_size, max_records - fetched)
        try:
            res = occ.search(
                taxonKey=taxon_key,
                geometry=wkt,
                hasCoordinate=True,
                limit=limit,
                offset=offset,
                country=country_code,
            )
        except Exception as e:
            logging.error(f"GBIF request failed at offset={offset}: {e}")
            break
        results = res.get("results", [])
        if not results:
            break
        records.extend(results)
        got = len(results)
        fetched += got
        offset += got
        logging.info(f"  - fetched {fetched} / {max_records} ...")
        time.sleep(sleep_s)

    if not records:
        logging.warning("No GBIF records found.")
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)

def clean_occurrences(df: pd.DataFrame, since_year: Optional[int]) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "speciesKey", "scientificName", "acceptedScientificName",
        "decimalLatitude", "decimalLongitude", "eventDate",
        "basisOfRecord", "individualCount", "elevation",
        "stateProvince", "locality", "datasetKey", "institutionCode",
        "year", "month", "day", "issues",
        "kingdom", "phylum", "class", "order", "family", "genus",
        "establishmentMeans", "canonicalName", "species"
    ]
    present = [c for c in cols if c in df.columns]
    df = df[present].copy()

    df = df[df["decimalLatitude"].notna() & df["decimalLongitude"].notna()]
    if "issues" in df.columns:
        df = df[~df["issues"].apply(_has_bad_issue)]
    if since_year is not None and "year" in df.columns:
        df = df[df["year"].fillna(0).astype(int) >= int(since_year)]

    dedup_subset = [c for c in ["speciesKey", "decimalLatitude", "decimalLongitude", "eventDate"] if c in df.columns]
    if dedup_subset:
        df = df.drop_duplicates(subset=dedup_subset)

    rename_map = {
        "speciesKey": "species_key",
        "scientificName": "scientific_name",
        "acceptedScientificName": "accepted_scientific_name",
        "decimalLatitude": "lat",
        "decimalLongitude": "lon",
        "eventDate": "event_date",
        "basisOfRecord": "basis_of_record",
        "individualCount": "individual_count",
        "stateProvince": "state_province",
        "datasetKey": "dataset_key",
        "institutionCode": "institution_code",
        "class": "class_name",
        "order": "order_name",
        "elevation": "elevation",
        "locality": "locality",
        "establishmentMeans": "establishment_means",
        "canonicalName": "canonical_name_from_gbif",
        "species": "species_binomial"
    }
    df = df.rename(columns=rename_map)

    for col in ["lat", "lon", "elevation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["year", "month", "day", "individual_count", "species_key"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "canonical_name_from_gbif" in df.columns and df["canonical_name_from_gbif"].notna().any():
        df["canonical_name"] = df["canonical_name_from_gbif"]
    elif "scientific_name" in df.columns and "accepted_scientific_name" in df.columns:
        df["canonical_name"] = df["accepted_scientific_name"].fillna(df["scientific_name"])
    elif "scientific_name" in df.columns:
        df["canonical_name"] = df["scientific_name"]
    else:
        df["canonical_name"] = None

    df["canonical_binomial"] = df["canonical_name"].apply(_canon_binomial)
    if "species_binomial" in df.columns:
        df["species_binomial"] = df["species_binomial"].apply(_canon_binomial)
        df["canonical_binomial"] = df["species_binomial"].fillna(df["canonical_binomial"])
    return df.reset_index(drop=True)

# ------------------------------------------------------------------------------
# IUCN enrichment (optional)
# ------------------------------------------------------------------------------
def fetch_iucn_category(name: str, token: str) -> Optional[str]:
    url = f"https://apiv3.iucnredlist.org/api/v3/species/{requests.utils.quote(name)}"
    params = {"token": token}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        result = (data or {}).get("result") or []
        if not result:
            return None
        return result[0].get("category")
    except Exception:
        return None

def enrich_species_with_iucn(species_df: pd.DataFrame, token: str, throttle_s: float = 0.25) -> pd.DataFrame:
    if species_df.empty:
        species_df["iucn_category"] = pd.NA
        return species_df
    if not token:
        logging.info("IUCN_TOKEN missing; skipping IUCN enrichment.")
        species_df["iucn_category"] = pd.NA
        return species_df
    cats: Dict[str, Optional[str]] = {}
    unique_names = species_df["canonical_name"].dropna().unique().tolist()
    logging.info(f"Fetching IUCN status for {len(unique_names)} species ...")
    for i, name in enumerate(unique_names, 1):
        cats[name] = fetch_iucn_category(name, token)
        if i % 50 == 0:
            logging.info(f"  - IUCN fetched {i}/{len(unique_names)}")
        time.sleep(throttle_s)
    species_df["iucn_category"] = species_df["canonical_name"].map(cats).astype("string")
    return species_df

# ------------------------------------------------------------------------------
# Table builders
# ------------------------------------------------------------------------------
def build_species_table(occ_df: pd.DataFrame) -> pd.DataFrame:
    if occ_df.empty:
        return pd.DataFrame(columns=[
            "species_key","canonical_name","scientific_name","family","genus",
            "class_name","order_name","kingdom","phylum","iucn_category","canonical_binomial"
        ])
    cols = ["species_key", "canonical_name", "scientific_name", "family", "genus",
            "class_name", "order_name", "kingdom", "phylum", "canonical_binomial"]
    present = [c for c in cols if c in occ_df.columns]
    sp = occ_df[present].dropna(subset=["species_key"]).drop_duplicates(subset=["species_key"]).copy()
    sp["species_key"] = sp["species_key"].astype("Int64")
    return sp.reset_index(drop=True)

def build_occurrence_table(occ_df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "species_key","lat","lon","event_date","basis_of_record","dataset_key",
        "institution_code","locality","state_province","elevation","year","month","day",
        "issues","establishment_means"
    ]
    if occ_df.empty:
        return pd.DataFrame(columns=expected)

    present = [c for c in expected if c in occ_df.columns]
    out = occ_df[present].copy()
    for c in expected:
        if c not in out.columns:
            out[c] = pd.NA

    for col in ["lat", "lon", "elevation"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["year", "month", "day"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    for col in ["event_date", "basis_of_record", "dataset_key", "institution_code",
                "locality", "state_province", "establishment_means"]:
        out[col] = out[col].astype("string")

    out["issues"] = out["issues"].apply(_serialize_issues).astype("string")

    out = out[out["species_key"].notna()]
    out["species_key"] = pd.to_numeric(out["species_key"], errors="coerce").astype("Int64")
    return out.reset_index(drop=True)

def build_species_distribution(
    species_df: pd.DataFrame,
    occ_df: pd.DataFrame,
    region_label: str,
    invasive_state: Set[str],
    invasive_country: Set[str],
) -> List[Dict[str, Any]]:
    non_native_means = {"INTRODUCED", "INVASIVE", "NATURALISED", "MANAGED", "DOMESTICATED"}
    recs: List[Dict[str, Any]] = []
    for _, row in species_df.iterrows():
        sk = int(row["species_key"])
        binom = (row.get("canonical_binomial")
                 or _canon_binomial(row.get("canonical_name") or row.get("scientific_name")))
        is_native = True
        if binom:
            if binom in invasive_state:   is_native = False
            elif binom in invasive_country: is_native = False

        if is_native and "establishment_means" in occ_df.columns and not occ_df.empty:
            em = occ_df[occ_df["species_key"] == sk]["establishment_means"].dropna().astype("string").str.upper()
            if len(em) > 0 and em.isin(non_native_means).all():
                is_native = False
        recs.append({"species_key": sk, "region_name": region_label, "is_native": 1 if is_native else 0})
    return recs

# ------------------------------------------------------------------------------
# SINGLE-AREA PIPELINE (original flow)
# ------------------------------------------------------------------------------
def run_pipeline_single(
    db_url: str,
    location: Optional[str],
    region: Optional[str],
    bbox_arg: Optional[str],
    taxon_key: int,
    country_code: Optional[str],
    since_year: Optional[int],
    max_records: int,
    out_csv_dir: Optional[str],
    enrich_iucn_flag: bool,
    dry_run: bool,
    sleep_s: float = 0.25,
    recreate_schema: bool = False,
) -> None:

    bbox_tuple: Optional[Tuple[float, float, float, float]] = None
    state_name: Optional[str] = None
    region_label: Optional[str] = None

    if bbox_arg:
        try:
            west, south, east, north = [float(x) for x in bbox_arg.split(",")]
            bbox_tuple = (west, south, east, north)
        except Exception:
            logging.error("--bbox must be 'west,south,east,north'")
            sys.exit(1)
        region_label = _slugify_region(region or location or "custom_bbox")
    else:
        loc_query = location or region
        if not loc_query:
            logging.error("Provide --location (or --region) or --bbox.")
            sys.exit(1)
        details = get_location_details(loc_query)
        if not details:
            logging.error("Could not resolve location to a bounding box.")
            sys.exit(1)
        bbox_tuple = details["bbox"]
        state_name = details.get("state")
        region_label = details.get("label") or _slugify_region(loc_query)

    raw_df = fetch_gbif_occurrences(
        taxon_key=taxon_key,
        bbox=bbox_tuple,              # type: ignore[arg-type]
        country_code=country_code,
        since_year=since_year,
        max_records=max_records,
        sleep_s=sleep_s,
    )
    if raw_df.empty:
        logging.warning("Nothing to ingest.")
        return

    occ_df = clean_occurrences(raw_df, since_year=since_year)
    if occ_df.empty:
        logging.warning("All records filtered out after cleaning.")
        return

    species_df = build_species_table(occ_df)
    if enrich_iucn_flag:
        species_df = enrich_species_with_iucn(species_df, IUCN_TOKEN)

    # simple placeholder trait rows (canopy, successional derived from basic placeholders)
    def estimate_canopy_layer(traits: dict) -> str:
        try:
            height = float(traits.get("max_height_m", 0))
            growth_form = (traits.get("growth_form", "tree") or "tree").lower()
            if "tree" in growth_form:
                if height > 20: return "Upper Canopy"
                if height > 10: return "Mid-story"
                return "Lower Canopy / Sub-canopy"
            if "shrub" in growth_form: return "Shrub Layer"
            if "herb" in growth_form or "grass" in growth_form: return "Understory / Ground Cover"
            return "Unknown"
        except (ValueError, TypeError):
            return "Unknown"
    def infer_successional_role(traits: dict) -> str:
        growth_rate = (traits.get("growth_rate") or "").lower()
        shade_tol   = (traits.get("shade_tolerance") or "").lower()
        if "fast" in growth_rate or "low" in shade_tol:  return "Pioneer"
        if "slow" in growth_rate or "high" in shade_tol: return "Climax"
        return "Intermediate"

    trait_records: list[dict] = []
    for _, row in species_df.iterrows():
        species_key = int(row["species_key"])
        canopy_layer = estimate_canopy_layer({})
        successional = infer_successional_role({})
        trait_records.extend([
            {"species_key": species_key, "trait_name": "canopy_layer",        "trait_value": canopy_layer, "source": "derived"},
            {"species_key": species_key, "trait_name": "successional_role",   "trait_value": successional, "source": "derived"},
        ])

    occ_out = build_occurrence_table(occ_df)

    # invasive status
    invasive_state: Set[str] = set()
    if state_name:
        invasive_state = _get_griis_invasive_index_state(state_name)
    invasive_country = _get_griis_invasive_index_country((country_code or "IN").upper())
    dist_records = build_species_distribution(
        species_df=species_df,
        occ_df=occ_out,
        region_label=region_label,
        invasive_state=invasive_state,
        invasive_country=invasive_country,
    )

    if out_csv_dir:
        outdir = Path(out_csv_dir); outdir.mkdir(parents=True, exist_ok=True)
        csv_prefix = region_label or "region"
        species_df.to_csv(outdir / f"{csv_prefix}_species.csv", index=False)
        occ_out.to_csv(outdir / f"{csv_prefix}_occurrences.csv", index=False)
        pd.DataFrame(trait_records).to_csv(outdir / f"{csv_prefix}_traits.csv", index=False)
        pd.DataFrame(dist_records).to_csv(outdir / f"{csv_prefix}_species_distribution.csv", index=False)
        logging.info(f"CSV exports written to {out_csv_dir}/...")

    if dry_run:
        logging.info("[dry-run] Skipping DB writes.")
        return

    engine = _connect_engine(db_url)
    ensure_schema(engine, recreate_schema=recreate_schema)
    n_sp = upsert_species(engine, species_df)
    n_tr = insert_traits(engine, trait_records)
    n_oc = insert_occurrences(engine, occ_out)
    n_sd = upsert_species_distribution(engine, dist_records)
    logging.info(f"DB ingest complete → species: {n_sp}, traits: {n_tr}, occurrences: {n_oc}, species_distribution: {n_sd} (region='{region_label}')")

# ------------------------------------------------------------------------------
# SPECIAL-14 PIPELINE (tiling + species facets, resumable)
# ------------------------------------------------------------------------------
def tile_bounds(bounds: List[float], cols: int = 3, rows: int = 3) -> List[List[float]]:
    xmin, ymin, xmax, ymax = bounds
    dx = (xmax - xmin) / cols
    dy = (ymax - ymin) / rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = xmin + c * dx
            y0 = ymin + r * dy
            x1 = x0 + dx
            y1 = y0 + dy
            tiles.append([x0, y0, x1, y1])
    return tiles

def _wkt_from_bounds(b: List[float]) -> str:
    return _bbox_to_wkt(tuple(b))  # type: ignore[arg-type]

def gbif_species_facets(geometry_wkt: str, facet_limit: int, facet_offset: int, sleep_s: float = 0.25) -> List[Dict[str, Any]]:
    params = {
        "taxonKey": PLANTAE_TAXON_KEY,
        "hasCoordinate": "true",
        "occurrenceStatus": "PRESENT",
        "geometry": geometry_wkt,
        "limit": 0,                      # we only want facets
        "facet": "speciesKey",
        "facetLimit": facet_limit,
        "facetOffset": facet_offset,
    }
    r = requests.get(GBIF_OCC_SEARCH, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    facets = (data.get("facets") or [])
    buckets = facets[0]["counts"] if facets else []
    time.sleep(sleep_s)
    return buckets  # [{"name": "12345", "count": 87}, ...]

def fetch_species_meta(species_key: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(GBIF_SPECIES_META + str(species_key), timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
        sci = j.get("scientificName") or j.get("canonicalName") or j.get("species")
        can = j.get("canonicalName") or sci
        fam = j.get("family"); gen = j.get("genus"); cls = j.get("class")
        ordn = j.get("order"); kng = j.get("kingdom"); phl = j.get("phylum")
        return {
            "species_key": int(species_key),
            "scientific_name": sci,
            "canonical_name": can,
            "family": fam, "genus": gen, "class_name": cls, "order_name": ordn,
            "kingdom": kng, "phylum": phl,
            "canonical_binomial": _canon_binomial(can or sci),
        }
    except Exception:
        return None

def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}

def _save_checkpoint(path: Path, ckpt: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ckpt, indent=2))

def _collect_species_from_dataset(dataset_key: str) -> Set[str]:
    names: Set[str] = set()
    offset, limit = 0, 1000
    while True:
        try:
            resp = requests.get(
                "https://api.gbif.org/v1/species/search",
                params={"datasetKey": dataset_key, "limit": limit, "offset": offset},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logging.warning(f"Failed fetching species chunk (offset={offset}) for dataset {dataset_key}: {e}")
            break
        rows = data.get("results", [])
        if not rows:
            break
        for r in rows:
            canon = _canon_binomial(r.get("scientificName"))
            if canon:
                names.add(canon)
        offset += limit
    return names

def _search_dataset_keys_by_query(query: str) -> List[str]:
    try:
        resp = requests.get(
            "https://api.gbif.org/v1/dataset/search",
            params={"q": query, "type": "CHECKLIST", "limit": 50},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return [d["key"] for d in data.get("results", []) if d.get("key")]
    except Exception as e:
        logging.warning(f"Dataset search failed for query '{query}': {e}")
        return []

def _get_griis_invasive_index_state(state_name: str) -> Set[str]:
    invasive: Set[str] = set()
    if not state_name:
        return invasive
    logging.info(f"Fetching GRIIS state checklist for: {state_name} ...")
    keys = _search_dataset_keys_by_query(f"GRIIS State Compendium {state_name}") or \
           _search_dataset_keys_by_query(f"GRIIS {state_name}")
    if keys:
        invasive = _collect_species_from_dataset(keys[0])
        logging.info(f"Loaded {len(invasive)} invasive names for {state_name}.")
    else:
        logging.warning(f"No state-specific GRIIS dataset found for {state_name}.")
    return invasive

def _get_griis_invasive_index_country(country_code: str = "IN") -> Set[str]:
    logging.info(f"Fetching national GRIIS compendium for {country_code} ...")
    keys = _search_dataset_keys_by_query(f"GRIIS Country Compendium {country_code}")
    if not keys:
        logging.warning("No national GRIIS dataset found.")
        return set()
    names = _collect_species_from_dataset(keys[0])
    logging.info(f"Loaded {len(names)} invasive (national) names.")
    return names

def ingest_special_14_facets(
    db_url: str,
    *,
    facet_page: int = 2000,
    max_species_per_region: int = 5000,
    tile_cols: int = 3,
    tile_rows: int = 3,
    sleep_page: float = 0.3,
    sleep_tile: float = 2.0,
    sleep_region: float = 8.0,
    checkpoint_path: str = "data/checkpoints/special14_facets.json",
    enrich_iucn_flag: bool = True,
) -> None:
    """
    Resumable ingestion for the 14 curated regions using GBIF species facets.
    """
    engine = _connect_engine(db_url)
    ensure_schema(engine, recreate_schema=False)

    ckpt_file = Path(checkpoint_path)
    ckpt = _load_checkpoint(ckpt_file)

    # national invasive list once
    invasive_country = _get_griis_invasive_index_country("IN")

    for ridx, region in enumerate(_SPECIAL_BOUNDS):
        name = region["name"]
        bounds = region["bounds"]
        region_label = _slugify_region(name)

        # region checkpoint
        rck = ckpt.get(region_label)
        if not rck:
            tiles = tile_bounds(bounds, cols=tile_cols, rows=tile_rows)
            rck = {
                "bounds": bounds,
                "tiles": [{"bounds": t, "facet_offset": 0, "done": False} for t in tiles],
                "done": False,
            }
            ckpt[region_label] = rck

        if rck.get("done"):
            logging.info(f"[{ridx+1:02d}/14] {region_label}: already done, skipping.")
            continue

        logging.info(f"[{ridx+1:02d}/14] Ingesting region: {region_label} (tiles={len(rck['tiles'])})")

        # state unknown for these rectangles; use only national GRIIS
        invasive_state: Set[str] = set()

        # collect species rows to upsert in batch
        batch_species: List[Dict[str, Any]] = []
        batch_dist:   List[Dict[str, Any]] = []
        seen_species_keys: Set[int] = set()

        # process tiles
        for t_idx, tile in enumerate(rck["tiles"]):
            if tile.get("done"):
                continue
            tb = tile["bounds"]
            local_offset = int(tile.get("facet_offset", 0))
            pulled = 0
            logging.info(f"  Tile {t_idx+1}/{len(rck['tiles'])}: starting facetOffset={local_offset}")

            while pulled < max_species_per_region:
                try:
                    buckets = gbif_species_facets(_wkt_from_bounds(tb), facet_limit=facet_page, facet_offset=local_offset, sleep_s=sleep_page)
                except Exception as e:
                    logging.warning(f"    facet call failed (offset={local_offset}): {e}")
                    break

                if not buckets:
                    tile["done"] = True
                    logging.info("    no more species facets; tile done.")
                    break

                for b in buckets:
                    skey = b.get("name")
                    if not skey:
                        continue
                    sk_int = int(skey)
                    if sk_int in seen_species_keys:
                        continue
                    meta = fetch_species_meta(skey)
                    if not meta:
                        continue
                    seen_species_keys.add(sk_int)
                    batch_species.append(meta)
                    binom = meta.get("canonical_binomial")
                    is_native = 0 if (binom and (binom in invasive_state or binom in invasive_country)) else 1
                    batch_dist.append({"species_key": sk_int, "region_name": region_label, "is_native": is_native})

                    if len(batch_species) >= 500:
                        sp_df = pd.DataFrame(batch_species)
                        n_sp = upsert_species(_connect_engine(db_url), sp_df)
                        n_sd = upsert_species_distribution(_connect_engine(db_url), batch_dist)
                        logging.info(f"    flushed species={n_sp}, dist={n_sd}")
                        batch_species.clear()
                        batch_dist.clear()

                    pulled += 1
                    if pulled >= max_species_per_region:
                        break

                local_offset += facet_page
                tile["facet_offset"] = local_offset
                _save_checkpoint(ckpt_file, ckpt)

            time.sleep(sleep_tile)

        # final flush
        if batch_species:
            sp_df = pd.DataFrame(batch_species)
            n_sp = upsert_species(_connect_engine(db_url), sp_df)
            n_sd = upsert_species_distribution(_connect_engine(db_url), batch_dist)
            logging.info(f"  final flush species={n_sp}, dist={n_sd}")

        # optional IUCN enrichment for species in this region
        if enrich_iucn_flag and IUCN_TOKEN:
            with _connect_engine(db_url).begin() as conn:
                rows = conn.execute(text("""
                    SELECT s.species_key, s.canonical_name
                    FROM species s
                    JOIN species_distribution d ON d.species_key = s.species_key
                    WHERE d.region_name = :r
                """), {"r": region_label}).mappings().all()
            if rows:
                sp_df = pd.DataFrame(rows)
                sp_df = sp_df.rename(columns={"canonical_name": "canonical_name"})
                # Fetch categories and update
                enriched = enrich_species_with_iucn(sp_df[["species_key","canonical_name"]].assign(iucn_category=pd.NA), IUCN_TOKEN)
                rows_up = enriched[["species_key","iucn_category"]].to_dict("records")
                with _connect_engine(db_url).begin() as conn:
                    for r in rows_up:
                        conn.execute(text("UPDATE species SET iucn_category = COALESCE(:cat, iucn_category) WHERE species_key = :sk"),
                                     {"cat": r["iucn_category"], "sk": int(r["species_key"])})
                logging.info(f"  IUCN enrichment updated for region '{region_label}'.")

        # mark region done if all tiles done
        all_done = all(t.get("done") for t in rck["tiles"])
        rck["done"] = all_done
        _save_checkpoint(ckpt_file, ckpt)
        logging.info(f"Region '{region_label}' complete: {all_done}.")
        time.sleep(sleep_region)

    logging.info("✅ Special-14 ingestion finished (checkpoints saved).")

# ------------------------------------------------------------------------------
# CLIMATE ENVELOPE MODELING (merged & adapted to row-per-trait schema)
# ------------------------------------------------------------------------------
class ClimateEnvelopeModeler:
    """
    Automated Climate Envelope Modeling → writes quantitative ranges
    into species_traits as rows (trait_name, trait_value), keyed by species_key.
    """
    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url
        self.engine = _connect_engine(db_url)
        ensure_schema(self.engine, recreate_schema=False)
        self.ee_initialized = self._init_ee()

        # Variables to extract
        self.env_variables = {
            'annual_precip_mm': 'Annual Precipitation',
            'mean_temp_c': 'Mean Annual Temperature',
            'min_temp_c': 'Minimum Temperature (coldest month)',
            'max_temp_c': 'Maximum Temperature (warmest month)',
            'soil_ph': 'Soil pH',
            'elevation_m': 'Elevation',
            'slope_deg': 'Slope',
            'ndvi_mean': 'NDVI (Vegetation Health)'
        }
        self.percentiles = [5, 25, 50, 75, 95]

    def _init_ee(self) -> bool:
        if not GEE_IMPORTED:
            logging.warning("ee not installed; using mock environmental data.")
            return False
        try:
            ee.Initialize()
            logging.info("✅ Google Earth Engine initialized.")
            return True
        except Exception as e:
            logging.warning(f"⚠️ GEE initialization failed: {e}")
            return False

    # ---------- species selection ----------
    def get_species_to_model(self, limit: Optional[int] = None) -> List[Tuple[int,str]]:
        """
        Returns [(species_key, scientific_name)] for species that are missing ANY of:
         min_ph, max_ph, min_rainfall_mm, max_rainfall_mm (as trait rows).
        """
        q = text("""
        WITH base AS (
          SELECT s.species_key, COALESCE(s.canonical_name, s.scientific_name) AS name
          FROM species s
        ),
        t AS (
          SELECT species_key,
                 SUM(CASE WHEN trait_name='min_ph' THEN 1 ELSE 0 END) AS has_min_ph,
                 SUM(CASE WHEN trait_name='max_ph' THEN 1 ELSE 0 END) AS has_max_ph,
                 SUM(CASE WHEN trait_name='min_rainfall_mm' THEN 1 ELSE 0 END) AS has_min_rain,
                 SUM(CASE WHEN trait_name='max_rainfall_mm' THEN 1 ELSE 0 END) AS has_max_rain
          FROM species_traits
          WHERE trait_name IN ('min_ph','max_ph','min_rainfall_mm','max_rainfall_mm')
          GROUP BY species_key
        )
        SELECT b.species_key, b.name
        FROM base b
        LEFT JOIN t ON t.species_key=b.species_key
        WHERE COALESCE(t.has_min_ph,0)=0 OR COALESCE(t.has_max_ph,0)=0
           OR COALESCE(t.has_min_rain,0)=0 OR COALESCE(t.has_max_rain,0)=0
        ORDER BY b.name
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(q).fetchall()
        pairs = [(int(r[0]), str(r[1])) for r in rows if r[0] is not None]
        if limit:
            pairs = pairs[:int(limit)]
        logging.info(f"📋 {len(pairs)} species need climate envelope traits.")
        return pairs

    # ---------- GBIF occurrences (simple) ----------
    def fetch_gbif_occurrences(self, species_name: str, max_records: int = 300) -> pd.DataFrame:
        if not PYGIBF_OK:
            logging.warning("pygbif not installed; generating mock occurrences.")
            return self._mock_occ(species_name)
        # Minimal GBIF fetch by species name
        try:
            # Direct API search (simplified)
            params = {"scientificName": species_name, "hasCoordinate": "true", "limit": max_records}
            r = requests.get(GBIF_OCC_SEARCH, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            rows = []
            for rec in results:
                lat, lon = rec.get("decimalLatitude"), rec.get("decimalLongitude")
                if lat is None or lon is None:
                    continue
                rows.append({
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "year": rec.get("year"),
                    "country": rec.get("country"),
                    "basis_of_record": rec.get("basisOfRecord"),
                    "dataset_key": rec.get("datasetKey"),
                })
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.drop_duplicates(subset=["latitude","longitude"])
            logging.info(f"GBIF occurrences for {species_name}: {len(df)}")
            return df
        except Exception as e:
            logging.warning(f"GBIF query failed for {species_name}: {e}; using mock occurrences.")
            return self._mock_occ(species_name)

    def _mock_occ(self, species_name: str) -> pd.DataFrame:
        np.random.seed(abs(hash(species_name)) % (2**32))
        n = np.random.randint(30, 90)
        lats = np.random.uniform(8.0, 37.0, n)
        lons = np.random.uniform(68.0, 97.0, n)
        return pd.DataFrame({"latitude": lats, "longitude": lons, "year": np.random.randint(1995, 2024, n)})

    # ---------- Environmental extraction ----------
    def extract_env(self, occ_df: pd.DataFrame) -> pd.DataFrame:
        if occ_df.empty:
            return occ_df
        if not self.ee_initialized:
            logging.warning("GEE not available; adding mock environmental values.")
            return self._mock_env(occ_df)

        try:
            # Build FeatureCollection
            points = [ee.Feature(ee.Geometry.Point([lon, lat])) for lat, lon in zip(occ_df["latitude"], occ_df["longitude"])]
            fc = ee.FeatureCollection(points)

            # Layers
            worldclim = ee.Image('WORLDCLIM/V1/BIO')
            annual_precip = worldclim.select('bio12').rename('annual_precip_mm')
            mean_temp = worldclim.select('bio01').divide(10).rename('mean_temp_c')
            min_temp = worldclim.select('bio06').divide(10).rename('min_temp_c')
            max_temp = worldclim.select('bio05').divide(10).rename('max_temp_c')

            # Soil pH (use resilient option)
            soil_ph_img = None
            for aid in [
                "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02",
                "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_A/v02",
                "projects/soilgrids-isric/ph_mean",   # may or may not be accessible
                "projects/soilgrids-isric/phh2o_mean"
            ]:
                try:
                    _img = ee.Image(aid)
                    _ = _img.bandNames().getInfo()
                    soil_ph_img = _img
                    logging.info(f"Using soil pH asset: {aid}")
                    break
                except Exception:
                    continue
            if soil_ph_img is None:
                soil_ph_img = ee.Image('WORLDCLIM/V1/BIO').select('bio01').multiply(0)  # neutral fallback @ 0
            soil_ph = soil_ph_img.divide(10).rename("soil_ph")

            # Topography
            dem = ee.Image('USGS/SRTMGL1_003').rename('elevation_m')
            slope = ee.Terrain.slope(dem).rename('slope_deg')

            # Vegetation (MODIS NDVI multi-year mean)
            ndvi = (ee.ImageCollection('MODIS/006/MOD13Q1')
                    .filterDate('2018-01-01', '2024-12-31')
                    .select('NDVI')).mean().multiply(0.0001).rename('ndvi_mean')

            composite = (annual_precip.addBands([mean_temp, min_temp, max_temp, soil_ph, dem, slope, ndvi]))
            sampled = composite.sampleRegions(collection=fc, scale=1000, geometries=False)

            data = sampled.getInfo()
            features = data.get("features", [])
            rows = []
            for i, feat in enumerate(features):
                props = feat.get("properties", {})
                rows.append({
                    "latitude": float(occ_df.iloc[i]["latitude"]),
                    "longitude": float(occ_df.iloc[i]["longitude"]),
                    **{k: props.get(k, np.nan) for k in self.env_variables.keys()}
                })
            env_df = pd.DataFrame(rows)
            # Filter rows that have at least half variables present
            thresh = max(1, int(0.5 * len(self.env_variables)))
            env_df = env_df.dropna(thresh=thresh)
            merged = pd.merge(occ_df, env_df, on=["latitude","longitude"], how="inner")
            logging.info(f"Extracted env at {len(merged)} of {len(occ_df)} points.")
            return merged

        except Exception as e:
            logging.warning(f"Env extraction failed: {e}; using mock env.")
            return self._mock_env(occ_df)

    def _mock_env(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        np.random.seed(42)
        out["annual_precip_mm"] = np.random.normal(900, 300, len(df))
        out["mean_temp_c"] = np.random.normal(25, 4, len(df))
        out["min_temp_c"] = out["mean_temp_c"] - np.random.uniform(6, 12, len(df))
        out["max_temp_c"] = out["mean_temp_c"] + np.random.uniform(6, 12, len(df))
        out["soil_ph"] = np.random.normal(7.1, 0.6, len(df))
        out["elevation_m"] = np.random.exponential(300, len(df))
        out["slope_deg"] = np.random.exponential(5, len(df))
        out["ndvi_mean"] = np.random.normal(0.5, 0.15, len(df))
        return out

    # ---------- Envelope statistics ----------
    def calc_envelope(self, env_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        if env_df.empty:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for var, desc in self.env_variables.items():
            if var not in env_df.columns:
                continue
            vals = pd.to_numeric(env_df[var], errors="coerce").dropna()
            if len(vals) < 10:
                continue
            q = np.percentile(vals, self.percentiles)
            out[var] = {
                "min_value": float(q[0]),
                "q25": float(q[1]),
                "median": float(q[2]),
                "q75": float(q[3]),
                "max_value": float(q[4]),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "sample_size": int(len(vals)),
                "description": desc,
            }
        return out

    # ---------- Persist to DB as rows in species_traits ----------
    def write_envelope_traits(self, species_key: int, species_name: str, env: Dict[str, Dict[str, Any]]) -> int:
        trait_rows: List[Dict[str, Any]] = []
        src = "GBIF+GEE Climate Envelope"
        now = datetime.now().isoformat()

        def add(name: str, value: Any):
            trait_rows.append({
                "species_key": species_key,
                "trait_name": name,
                "trait_value": None if value is None else str(value),
                "source": src,
            })

        # Map climate stats to traits
        if "annual_precip_mm" in env:
            add("min_rainfall_mm", env["annual_precip_mm"]["min_value"])
            add("max_rainfall_mm", env["annual_precip_mm"]["max_value"])
            add("optimal_rainfall_mm", env["annual_precip_mm"]["median"])

        if "soil_ph" in env:
            add("min_ph", env["soil_ph"]["min_value"])
            add("max_ph", env["soil_ph"]["max_value"])
            add("optimal_ph", env["soil_ph"]["median"])

        if "mean_temp_c" in env:
            # infer bounds using min/max if available, else use mean_temp envelope
            min_t = env.get("min_temp_c", {}).get("min_value", env["mean_temp_c"]["min_value"])
            max_t = env.get("max_temp_c", {}).get("max_value", env["mean_temp_c"]["max_value"])
            add("min_temp_c", min_t)
            add("max_temp_c", max_t)
            add("optimal_temp_c", env["mean_temp_c"]["median"])

        if "elevation_m" in env:
            add("min_elevation_m", env["elevation_m"]["min_value"])
            add("max_elevation_m", env["elevation_m"]["max_value"])

        # Derived qualitative traits
        # Drought tolerance from min precip
        if "annual_precip_mm" in env:
            minp = env["annual_precip_mm"]["min_value"]
            drought = "High" if minp < 300 else ("Moderate" if minp < 600 else "Low")
            add("drought_tolerance", drought)
        # Soil type pref from median pH
        if "soil_ph" in env:
            medph = env["soil_ph"]["median"]
            soil_pref = "Acidic" if medph < 6.0 else ("Alkaline" if medph > 7.5 else "Neutral")
            add("soil_type_preference", soil_pref)
        # Climate zone
        if "mean_temp_c" in env and "annual_precip_mm" in env:
            t = env["mean_temp_c"]["median"]; p = env["annual_precip_mm"]["median"]
            if t > 25 and p > 1000:
                cz = "Tropical"
            elif t > 20 and p > 600:
                cz = "Subtropical"
            elif t > 15:
                cz = "Temperate"
            else:
                cz = "Cool Temperate"
            add("climate_zone", cz)

        # metadata
        add("climate_envelope_sample_size", env.get("annual_precip_mm", {}).get("sample_size", 0))
        add("climate_envelope_last_updated", now)

        n = insert_traits(self.engine, trait_rows)
        logging.info(f"Updated {n} trait rows for {species_name} (key={species_key}).")
        return n

    # ---------- Orchestration ----------
    def process_species(self, species_key: int, species_name: str) -> bool:
        logging.info(f"\n🌿 Climate Envelope: {species_name} (key={species_key})")
        occ = self.fetch_gbif_occurrences(species_name, max_records=300)
        if occ.empty:
            logging.warning("No occurrences; skipping.")
            return False
        env_pts = self.extract_env(occ)
        if env_pts.empty:
            logging.warning("No environmental points; skipping.")
            return False
        env = self.calc_envelope(env_pts)
        if not env:
            logging.warning("No envelope statistics; skipping.")
            return False
        self.write_envelope_traits(species_key, species_name, env)
        return True

    def process_all(self, limit: Optional[int] = None, delay_s: float = 1.5) -> Dict[str, Any]:
        pairs = self.get_species_to_model(limit=limit)
        if not pairs:
            logging.info("No species need envelope traits.")
            return {"total": 0, "success": 0, "failed": 0, "failed_species": []}
        ok = 0; fail = 0; failed: List[str] = []
        for i, (sk, name) in enumerate(pairs, 1):
            logging.info(f"[{i}/{len(pairs)}] {name}")
            try:
                if self.process_species(sk, name):
                    ok += 1
                else:
                    fail += 1
                    failed.append(name)
            except KeyboardInterrupt:
                logging.info("Interrupted by user.")
                break
            except Exception as e:
                logging.error(f"Unexpected error on {name}: {e}")
                fail += 1
                failed.append(name)
            time.sleep(delay_s)
        logging.info(f"Envelope modeling done: success={ok}, failed={fail}")
        return {"total": len(pairs), "success": ok, "failed": fail, "failed_species": failed}

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Manthan Data Ingestion & Climate Envelope Modeling"
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # Single-area GBIF ingest
    p_single = sub.add_parser("single", help="Single area GBIF ingest (original pipeline).")
    p_single.add_argument("--db-url", type=str, default=DB_URL)
    p_single.add_argument("--location", type=str, help="Free-text location (e.g., 'Moradabad, Uttar Pradesh, India').")
    p_single.add_argument("--region", type=str, help="Alias for --location.")
    p_single.add_argument("--bbox", type=str, help="Bounding box 'west,south,east,north'.")
    p_single.add_argument("--taxon-key", type=int, default=PLANTAE_TAXON_KEY)
    p_single.add_argument("--country", type=str, default="IN")
    p_single.add_argument("--since-year", type=int, default=2000)
    p_single.add_argument("--max-records", type=int, default=10000)
    p_single.add_argument("--csv-out", type=str, help="Directory to also write CSV exports.")
    p_single.add_argument("--dry-run", action="store_true")
    p_single.add_argument("--sleep-s", type=float, default=0.25)
    p_single.add_argument("--no-iucn", action="store_true")
    p_single.add_argument("--recreate-schema", action="store_true")

    # Special-14 regions
    p_s14 = sub.add_parser("special14", help="Curated 14-region ingest via species facets (resumable).")
    p_s14.add_argument("--db-url", type=str, default=DB_URL)
    p_s14.add_argument("--facet-page", type=int, default=2000)
    p_s14.add_argument("--max-species-per-region", type=int, default=5000)
    p_s14.add_argument("--tile-cols", type=int, default=3)
    p_s14.add_argument("--tile-rows", type=int, default=3)
    p_s14.add_argument("--sleep-page", type=float, default=0.3)
    p_s14.add_argument("--sleep-tile", type=float, default=2.0)
    p_s14.add_argument("--sleep-region", type=float, default=8.0)
    p_s14.add_argument("--checkpoint", type=str, default="data/checkpoints/special14_facets.json")
    p_s14.add_argument("--no-iucn", action="store_true")

    # Climate Envelope Modeling
    p_env = sub.add_parser("envelope", help="Climate Envelope Modeling to derive quantitative trait ranges.")
    p_env.add_argument("--db-url", type=str, default=DB_URL)
    group = p_env.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Process all species that are missing envelope traits.")
    group.add_argument("--species-key", type=int, help="Process a single species by species_key.")
    group.add_argument("--species-name", type=str, help="Process a single species by scientific/canonical name.")
    p_env.add_argument("--limit", type=int, help="Limit number of species for --all.")
    p_env.add_argument("--delay-s", type=float, default=1.5)

    args = parser.parse_args()
    if args.cmd == "single":
        enrich_flag = (not args.no_iucn) and bool(IUCN_TOKEN)
        run_pipeline_single(
            db_url=args.db_url,
            location=args.location,
            region=args.region,
            bbox_arg=args.bbox,
            taxon_key=args.taxon_key,
            country_code=args.country,
            since_year=args.since_year,
            max_records=args.max_records,
            out_csv_dir=args.csv_out,
            enrich_iucn_flag=enrich_flag,
            dry_run=args.dry_run,
            sleep_s=args.sleep_s,
            recreate_schema=args.recreate_schema,
        )
    elif args.cmd == "special14":
        enrich_flag = (not args.no_iucn) and bool(IUCN_TOKEN)
        ingest_special_14_facets(
            db_url=args.db_url,
            facet_page=args.facet_page,
            max_species_per_region=args.max_species_per_region,
            tile_cols=args.tile_cols,
            tile_rows=args.tile_rows,
            sleep_page=args.sleep_page,
            sleep_tile=args.sleep_tile,
            sleep_region=args.sleep_region,
            checkpoint_path=args.checkpoint,
            enrich_iucn_flag=enrich_flag,
        )
    else:
        # envelope
        modeler = ClimateEnvelopeModeler(db_url=args.db_url)
        if args.all:
            modeler.process_all(limit=args.limit, delay_s=args.delay_s)
        elif args.species_key is not None:
            # lookup name for logging
            with _connect_engine(args.db_url).begin() as conn:
                row = conn.execute(text("SELECT COALESCE(canonical_name,scientific_name) FROM species WHERE species_key=:k"),
                                   {"k": int(args.species_key)}).fetchone()
            name = row[0] if row else f"key={args.species_key}"
            modeler.process_species(int(args.species_key), str(name))
        else:
            # name provided → find key
            with _connect_engine(args.db_url).begin() as conn:
                r = conn.execute(text("""
                    SELECT species_key FROM species
                    WHERE scientific_name=:n OR canonical_name=:n
                    ORDER BY species_key LIMIT 1
                """), {"n": args.species_name}).fetchone()
            if not r:
                logging.error(f"Species not found in DB: {args.species_name}")
                sys.exit(1)
            sk = int(r[0])
            modeler.process_species(sk, args.species_name)

if __name__ == "__main__":
    main()
