#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Optional: Google Earth Engine (for fast, per-region sampling)
GEE_AVAILABLE = True
try:
    import ee
except Exception:
    GEE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DB_URL = os.getenv("MANTHAN_DB_URL", "sqlite:///data/manthan.db")

# ---------------------------------------------------------------------
# Region bounds (the curated “Special 14” rectangles)
# ---------------------------------------------------------------------
SPECIAL_BOUNDS: Dict[str, List[float]] = {
    "western_ghats_north":   [72.8, 17.0, 75.3, 20.8],
    "western_ghats_central": [74.0, 13.0, 76.0, 16.5],
    "western_ghats_south":   [76.0,  8.0, 77.5, 12.2],
    "eastern_ghats_north":   [83.0, 18.0, 85.5, 21.0],
    "eastern_ghats_south":   [79.0, 13.0, 81.0, 16.0],
    "himalaya_uttarakhand":  [79.0, 29.5, 81.1, 31.5],
    "himalaya_himachal":     [76.0, 31.0, 78.0, 32.8],
    "kashmir_valley":        [74.0, 33.5, 75.7, 34.7],
    "meghalaya_hills":       [90.0, 25.0, 92.7, 26.5],
    "assam_floodplains":     [91.0, 26.5, 95.0, 27.7],
    "thar_desert_core":      [70.0, 25.0, 73.5, 28.5],
    "rann_of_kutch":         [68.5, 23.0, 71.0, 24.8],
    "sundarbans":            [88.0, 21.5, 89.6, 22.6],
    "deccan_plateau_core":   [75.0, 15.0, 80.0, 20.0],
}

# ---------------------------------------------------------------------
# Fallback per-region envelopes (when GEE is unavailable)
# Values are conservative “good-enough” ranges to get the recommender working.
# ---------------------------------------------------------------------
FALLBACK_ENV_BY_REGION: Dict[str, Dict[str, float]] = {
    # mins / maxes are interpreted as the 5th / 95th percentile proxies
    "western_ghats_north":   {"min_rain":1200,"max_rain":3000,"opt_rain":2000,"min_ph":5.0,"max_ph":6.8,"opt_ph":6.0,"min_t":18,"max_t":28,"opt_t":23},
    "western_ghats_central": {"min_rain":1500,"max_rain":3500,"opt_rain":2300,"min_ph":5.0,"max_ph":6.8,"opt_ph":6.0,"min_t":19,"max_t":28,"opt_t":24},
    "western_ghats_south":   {"min_rain":1600,"max_rain":3800,"opt_rain":2500,"min_ph":4.8,"max_ph":6.5,"opt_ph":5.8,"min_t":20,"max_t":29,"opt_t":25},
    "eastern_ghats_north":   {"min_rain":900, "max_rain":1800,"opt_rain":1300,"min_ph":5.5,"max_ph":7.2,"opt_ph":6.5,"min_t":20,"max_t":29,"opt_t":25},
    "eastern_ghats_south":   {"min_rain":700, "max_rain":1500,"opt_rain":1100,"min_ph":6.0,"max_ph":7.6,"opt_ph":6.8,"min_t":21,"max_t":31,"opt_t":26},
    "himalaya_uttarakhand":  {"min_rain":800, "max_rain":2500,"opt_rain":1400,"min_ph":4.8,"max_ph":6.8,"opt_ph":5.8,"min_t":6, "max_t":20,"opt_t":14},
    "himalaya_himachal":     {"min_rain":700, "max_rain":2200,"opt_rain":1300,"min_ph":4.8,"max_ph":6.8,"opt_ph":5.8,"min_t":5, "max_t":18,"opt_t":12},
    "kashmir_valley":        {"min_rain":500, "max_rain":1200,"opt_rain":800, "min_ph":6.2,"max_ph":7.8,"opt_ph":7.0,"min_t":3, "max_t":18,"opt_t":10},
    "meghalaya_hills":       {"min_rain":2000,"max_rain":5000,"opt_rain":3000,"min_ph":4.5,"max_ph":6.2,"opt_ph":5.4,"min_t":15,"max_t":25,"opt_t":20},
    "assam_floodplains":     {"min_rain":1800,"max_rain":3500,"opt_rain":2600,"min_ph":5.0,"max_ph":6.8,"opt_ph":6.0,"min_t":18,"max_t":29,"opt_t":24},
    "thar_desert_core":      {"min_rain":80,  "max_rain":350, "opt_rain":200, "min_ph":7.5,"max_ph":8.5,"opt_ph":8.0,"min_t":18,"max_t":34,"opt_t":27},
    "rann_of_kutch":         {"min_rain":200, "max_rain":600, "opt_rain":350, "min_ph":7.8,"max_ph":8.6,"opt_ph":8.2,"min_t":20,"max_t":34,"opt_t":28},
    "sundarbans":            {"min_rain":1600,"max_rain":3000,"opt_rain":2200,"min_ph":6.2,"max_ph":8.0,"opt_ph":7.2,"min_t":20,"max_t":30,"opt_t":26},
    "deccan_plateau_core":   {"min_rain":500, "max_rain":1200,"opt_rain":800, "min_ph":6.5,"max_ph":8.0,"opt_ph":7.2,"min_t":20,"max_t":33,"opt_t":26},
}

# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------
def engine() -> Engine:
    return create_engine(DB_URL, future=True)

def fetch_regions_in_db(conn) -> List[str]:
    rows = conn.execute(text("SELECT DISTINCT region_name FROM species_distribution")).fetchall()
    return [r[0] for r in rows]

def species_by_region(conn, region: str) -> List[int]:
    rows = conn.execute(
        text("SELECT species_key FROM species_distribution WHERE region_name=:r AND is_native=1"),
        {"r": region},
    ).fetchall()
    return [int(r[0]) for r in rows]

def upsert_traits(conn, species_key: int, traits: Dict[str, Any], source: str):
    rows = [
        {"species_key": species_key, "trait_name": k, "trait_value": str(v), "source": source}
        for k, v in traits.items()
        if v is not None
    ]
    if not rows:
        return
    conn.execute(text("""
        INSERT INTO species_traits (species_key, trait_name, trait_value, source)
        VALUES (:species_key, :trait_name, :trait_value, :source)
        ON CONFLICT(species_key, trait_name) DO UPDATE SET
            trait_value = excluded.trait_value,
            source = excluded.source,
            last_updated = datetime('now')
    """), rows)

# ---------------------------------------------------------------------
# Derived traits from envelopes
# ---------------------------------------------------------------------
def infer_drought_tolerance(min_rain: float) -> str:
    try:
        mr = float(min_rain)
    except Exception:
        return "Unknown"
    if mr < 300:  return "High"
    if mr < 600:  return "Moderate"
    return "Low"

def infer_soil_pref(opt_ph: float) -> str:
    try:
        p = float(opt_ph)
    except Exception:
        return "Unknown"
    if p < 6.0:   return "Acidic"
    if p > 7.5:   return "Alkaline"
    return "Neutral"

def infer_climate_zone(opt_t: float, opt_rain: float) -> str:
    try:
        t = float(opt_t); r = float(opt_rain)
    except Exception:
        return "Unknown"
    if t > 25 and r > 1000: return "Tropical"
    if t > 20 and r > 600:  return "Subtropical"
    if t > 15:              return "Temperate"
    return "Cool Temperate"

# ---------------------------------------------------------------------
# GEE envelopes per region (fast: 1 call per region)
# ---------------------------------------------------------------------
def init_gee():
    global GEE_AVAILABLE
    if not GEE_AVAILABLE:
        return
    try:
        ee.Initialize()
    except Exception:
        try:
            ee.Initialize(project=os.getenv("EE_PROJECT_ID", None))
        except Exception as e:
            logging.warning(f"GEE init failed: {e}")
            GEE_AVAILABLE = False

def env_envelope_from_gee(bounds: List[float]) -> Optional[Dict[str, float]]:
    """
    Compute 5th/95th percentile envelopes from public rasters quickly for a bbox.
    """
    if not GEE_AVAILABLE:
        return None
    w, s, e, n = bounds
    aoi = ee.Geometry.Rectangle([w, s, e, n], proj="EPSG:4326", geodesic=False)

    # WorldClim
    worldclim = ee.Image("WORLDCLIM/V1/BIO")
    annual_precip = worldclim.select("bio12").rename("annual_precip_mm")
    mean_temp = worldclim.select("bio01").divide(10).rename("mean_temp_c")
    min_temp = worldclim.select("bio06").divide(10).rename("min_temp_c")
    max_temp = worldclim.select("bio05").divide(10).rename("max_temp_c")

    # Soil pH (fallback chain)
    ph_candidates = [
        "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02",
        "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_A/v02",
        "projects/soilgrids-isric/ph_mean",
        "projects/soilgrids-isric/phh2o_mean",
    ]
    ph_img = None
    for aid in ph_candidates:
        try:
            img = ee.Image(aid)
            _ = img.bandNames().getInfo()
            ph_img = img
            break
        except Exception:
            continue
    if ph_img is None:
        # last resort: neutral-ish
        logging.warning("No accessible pH asset; using fallback neutral pH.")
        return None
    soil_ph = ph_img.select(0).divide(10).rename("soil_ph")

    stack = annual_precip.addBands([mean_temp, min_temp, max_temp, soil_ph])

    # Percentiles (5th, 50th, 95th)
    reducer = ee.Reducer.percentile([5, 50, 95])
    stats = stack.reduceRegion(reducer, aoi, scale=1000, bestEffort=True).getInfo() or {}

    def _g(key): 
        v = stats.get(f"{key}_p5"), stats.get(f"{key}_p50"), stats.get(f"{key}_p95")
        return v

    p5, p50, p95 = _g("annual_precip_mm")
    t5, t50, t95 = _g("mean_temp_c")
    ph5, ph50, ph95 = _g("soil_ph")

    # Fallback if missing any
    if any(v is None for v in [p5, p50, p95, t5, t50, t95, ph5, ph50, ph95]):
        return None

    return {
        "min_rain": float(p5),  "opt_rain": float(p50), "max_rain": float(p95),
        "min_t": float(t5),     "opt_t": float(t50),    "max_t": float(t95),
        "min_ph": float(ph5),   "opt_ph": float(ph50),  "max_ph": float(ph95),
    }

# ---------------------------------------------------------------------
# Merge envelopes for species native to multiple regions
# ---------------------------------------------------------------------
def merge_env_ranges(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    mins = {"min_rain","min_t","min_ph"}
    maxs = {"max_rain","max_t","max_ph"}
    meds = {"opt_rain","opt_t","opt_ph"}

    def _vals(k): return [x[k] for x in items if x.get(k) is not None]

    for k in mins:
        vs = _vals(k)
        merged[k] = min(vs) if vs else None
    for k in maxs:
        vs = _vals(k)
        merged[k] = max(vs) if vs else None
    for k in meds:
        vs = _vals(k)
        merged[k] = sum(vs)/len(vs) if vs else None

    return merged

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    init_gee()
    eng = engine()

    with eng.begin() as conn:
        # Ensure required tables exist (will NOOP if already there)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS species_traits (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          species_key INTEGER NOT NULL,
          trait_name TEXT NOT NULL,
          trait_value TEXT,
          source TEXT,
          last_updated TEXT DEFAULT (datetime('now')),
          UNIQUE (species_key, trait_name),
          FOREIGN KEY (species_key) REFERENCES species (species_key)
        )"""))

        regions = fetch_regions_in_db(conn)
        if not regions:
            logging.error("No regions found in species_distribution. Nothing to patch.")
            sys.exit(1)

        # Compute (or fallback) envelopes per region we know bounds for
        region_env: Dict[str, Dict[str, float]] = {}
        for r in regions:
            rkey = r.strip().lower()
            if rkey not in SPECIAL_BOUNDS:
                logging.warning(f"Region '{r}' not in SPECIAL_BOUNDS; skipping envelope (no bounds).")
                continue

            logging.info(f"Computing envelope for region: {r}")
            env = env_envelope_from_gee(SPECIAL_BOUNDS[rkey]) if GEE_AVAILABLE else None
            if env is None:
                env = FALLBACK_ENV_BY_REGION.get(rkey)
                if env is None:
                    logging.warning(f"No envelope (GEE+fallback missing) for '{r}'; skipping.")
                    continue
            region_env[r] = env

        if not region_env:
            logging.error("No region envelopes computed. Aborting.")
            sys.exit(1)

        # For each species, collect all region envelopes it is native to, then merge & write traits
        # Get unique species across covered regions
        covered_regions = tuple(region_env.keys())
        rows = conn.execute(text(f"""
            SELECT DISTINCT sd.species_key, sd.region_name
            FROM species_distribution sd
            WHERE sd.region_name IN ({",".join([f":r{i}" for i in range(len(covered_regions))])})
              AND sd.is_native = 1
        """), {f"r{i}": covered_regions[i] for i in range(len(covered_regions))}).mappings().all()

        species_to_regions: Dict[int, List[str]] = {}
        for row in rows:
            sk = int(row["species_key"]); reg = row["region_name"]
            species_to_regions.setdefault(sk, []).append(reg)

        logging.info(f"Species with covered regions: {len(species_to_regions)}")

        n_updated = 0
        for sk, regs in species_to_regions.items():
            envs = [region_env[r] for r in regs if r in region_env]
            if not envs:
                continue
            merged = merge_env_ranges(envs)

            # Build trait payload
            traits = {
                "min_rainfall_mm": round(merged.get("min_rain"), 1) if merged.get("min_rain") is not None else None,
                "max_rainfall_mm": round(merged.get("max_rain"), 1) if merged.get("max_rain") is not None else None,
                "optimal_rainfall_mm": round(merged.get("opt_rain"), 1) if merged.get("opt_rain") is not None else None,
                "min_ph": round(merged.get("min_ph"), 2) if merged.get("min_ph") is not None else None,
                "max_ph": round(merged.get("max_ph"), 2) if merged.get("max_ph") is not None else None,
                "optimal_ph": round(merged.get("opt_ph"), 2) if merged.get("opt_ph") is not None else None,
                "min_temp_c": round(merged.get("min_t"), 1) if merged.get("min_t") is not None else None,
                "max_temp_c": round(merged.get("max_t"), 1) if merged.get("max_t") is not None else None,
                "optimal_temp_c": round(merged.get("opt_t"), 1) if merged.get("opt_t") is not None else None,
            }

            # Derived qualitative traits (these are what your UI/rules already use)
            dtol = infer_drought_tolerance(traits.get("min_rainfall_mm"))
            soilp = infer_soil_pref(traits.get("optimal_ph"))
            cz = infer_climate_zone(traits.get("optimal_temp_c"), traits.get("optimal_rainfall_mm"))

            traits.update({
                "drought_tolerance": dtol,
                "soil_type_preference": soilp,
                "climate_zone": cz,
            })

            upsert_traits(conn, sk, traits, source="region-envelope (fast patch)")
            n_updated += 1

        logging.info(f"✅ Patched quantitative traits for {n_updated} species.")
        logging.info("You can now re-run the dashboard; the recommender will stop saying 'no species found'.")
        logging.info("Later, you can supersede these with per-occurrence climate envelopes; UPSERTs keep this safe.")

if __name__ == "__main__":
    main()
