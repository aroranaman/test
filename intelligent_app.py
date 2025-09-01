# intelligent_app.py
from __future__ import annotations

# --- macOS OpenMP workaround (put FIRST; dev only) ---
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import os
import sys
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Requests is optional (used for direct GEE download URL)
try:
    import requests
except Exception:
    requests = None

# Google Earth Engine
import ee

# ---------------- Paths & Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------- Config ----------------
st.set_page_config(page_title="Manthan: Living Forest Intelligence", layout="wide")
st.title("🌱 Manthan: Living Forest Intelligence")

DB_URL = os.getenv("MANTHAN_DB_URL", "sqlite:///data/manthan.db")
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
RESNET_MODEL_PATH = REPO_ROOT / "models" / "artifacts" / "resnet34_pan_india_weights.pth"
UNET_MODEL_PATH   = REPO_ROOT / "models" / "artifacts" / "unet_pan_india_final_v1.pth"

# Placeholder images for UI if inference is skipped/unavailable
PLACEHOLDER_RESNET_IMG = REPO_ROOT / "assets" / "placeholder_resnet_map.png"
PLACEHOLDER_UNET_IMG   = REPO_ROOT / "assets" / "placeholder_unet_map.png"

# AOI raster (download target) + quicklook paths
INFER_DIR = REPO_ROOT / "data" / "inference"
AOI_TIF_PATH = INFER_DIR / "current_aoi.tif"
RESNET_PNG = INFER_DIR / "resnet_quicklook.png"
UNET_PNG   = INFER_DIR / "unet_quicklook.png"

# ---------------- Core Manthan Imports ----------------
from manthan_core.recommender.intelligent_recommender import IntelligentRecommender
from manthan_core.utils.model_runner import run_resnet_inference, run_unet_inference
from manthan_core.site_assessment.gee_pipeline import (
    get_site_fingerprint,
    download_aoi_as_geotiff,  # placeholder-friendly fallback
)
from manthan_core.utils.db_connector import KnowledgeBase

# ---------------- EE init (quiet) ----------------
def _init_ee() -> str:
    project = os.getenv("EE_PROJECT_ID", "manthan-466509")
    try:
        ee.Initialize(project=project)
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=project)
        except Exception as e:
            logging.warning(f"Earth Engine init skipped (auth not available now): {e}")
    return project

ee_project = _init_ee()

# ---------------- Health checks ----------------
def _sqlite_path_from_url(db_url: str) -> Path:
    return Path(db_url.replace("sqlite:///", "", 1)) if db_url.startswith("sqlite:///") else Path(db_url)

db_ok     = _sqlite_path_from_url(DB_URL).exists()
index_ok  = all((EMBEDDINGS_DIR / f).exists() for f in ("faiss.index", "embeddings.npy", "items.json"))
resnet_ok = RESNET_MODEL_PATH.exists()
unet_ok   = UNET_MODEL_PATH.exists()

with st.sidebar:
    st.header("🧠 Knowledge Base Health")
    st.write(f"Database: {'✅ Found' if db_ok else '❌ Not Found'}")
    st.write(f"Semantic Index: {'✅ Found' if index_ok else '❌ Not Found'}")
    st.write(f"ResNet Model: {'✅ Found' if resnet_ok else '❌ Not Found'}")
    st.write(f"U-Net Model: {'✅ Found' if unet_ok else '❌ Not Found'}")
    st.caption(f"EE Project: `{ee_project}`")

# ---------------- Load components (independently cached) ----------------
@st.cache_resource(show_spinner=False)
def get_kb():
    try:
        return KnowledgeBase(DB_URL)
    except Exception as e:
        logging.error(f"Failed to create KnowledgeBase: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_recommender():
    if not index_ok:
        return None
    try:
        return IntelligentRecommender(EMBEDDINGS_DIR)
    except Exception as e:
        logging.info(f"Semantic recommender disabled: {e}")
        return None

kb = get_kb()
recommender = get_recommender()

if kb is None:
    st.error("Database connection failed. Please ensure data/manthan.db exists and is accessible.")
    st.stop()

# ---------------- Sidebar Inputs ----------------
with st.sidebar:
    st.header("1. Your Goal")
    goal_query = st.text_area(
        "Describe your restoration goal:",
        "A drought-tolerant, fast-growing forest with high biodiversity for the plains of Uttar Pradesh."
    )

    st.header("2. Your Land")
    aoi_input_str = st.text_input("Area of Interest (lon1,lat1,lon2,lat2):", "78.10,27.50,78.11,27.51")

    st.header("3. Species Filtering")
    strictness = st.selectbox(
        "Filter strictness",
        ["Balanced", "Strict", "Soft"],
        index=0,
        help="Strict removes species outside pH/rainfall ranges. Soft keeps near misses with penalty."
    )
    min_compat = st.slider("Minimum environmental compatibility", 0.0, 1.0, 0.4, 0.05)

    st.header("4. AOI Raster for Inference")
    st.caption("The app will try to download a Sentinel-2 RGB to `data/inference/current_aoi.tif`.")

    run_clicked = st.button("Generate Plan", type="primary")

# ---------------- DB utilities for fallbacks & stats ----------------
def _get_all_species_df(db_url: str) -> pd.DataFrame:
    p = _sqlite_path_from_url(db_url)
    if not p.exists():
        return pd.DataFrame(columns=["species_key", "scientific_name", "canonical_name"])
    conn = sqlite3.connect(str(p))
    try:
        return pd.read_sql("SELECT species_key, scientific_name, canonical_name FROM species", conn)
    finally:
        conn.close()

def _get_all_traits_df(db_url: str) -> pd.DataFrame:
    p = _sqlite_path_from_url(db_url)
    if not p.exists():
        return pd.DataFrame(columns=["species_key", "trait_name", "trait_value"])
    conn = sqlite3.connect(str(p))
    try:
        return pd.read_sql("SELECT species_key, trait_name, trait_value FROM species_traits", conn)
    finally:
        conn.close()

# ---------------- AOI download (real attempt via signed URL) ----------------
def try_real_aoi_download(aoi: ee.Geometry, out_path: Path, scale: int = 10) -> bool:
    """
    Try to fetch a real Sentinel-2 RGB GeoTIFF using a signed EE download URL.
    Falls back to False if requests is unavailable or any error occurs.
    """
    if requests is None:
        return False
    try:
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate("2023-01-01", "2024-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .median()
            .select(["B4","B3","B2"])
            .multiply(0.0001)
            .clip(aoi)
        )
        region = aoi.coordinates().getInfo()
        params = {
            "scale": scale,
            "crs": "EPSG:4326",
            "region": region,
            "format": "GEO_TIFF",
        }
        url = s2.getDownloadURL(params)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info(f"Downloaded AOI GeoTIFF to {out_path}")
        return True
    except Exception as e:
        logging.warning(f"Real AOI download failed: {e}")
        return False

# ---------------- Quicklook helpers ----------------
def _save_labelmap_png(arr: np.ndarray, out_png: Path) -> None:
    """Save a label map (HxW uint8) as a PNG grayscale for quicklook."""
    try:
        im = Image.fromarray(arr.astype(np.uint8), mode="L")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_png)
    except Exception as e:
        logging.warning(f"Failed to save quicklook {out_png.name}: {e}")

# ---------------- Synonym-aware micro filter ----------------
PH_MIN_KEYS   = ["min_ph", "ph_min", "soil_ph_min", "ph_low"]
PH_MAX_KEYS   = ["max_ph", "ph_max", "soil_ph_max", "ph_high"]
RAIN_MIN_KEYS = ["min_rain_mm", "min_precip_mm", "rain_min_mm", "rainfall_min_mm", "precip_min_mm"]
RAIN_MAX_KEYS = ["max_rain_mm", "max_precip_mm", "rain_max_mm", "rainfall_max_mm", "precip_max_mm"]
TEMP_MIN_KEYS = ["min_temp_c", "temp_min_c", "tmin_c"]
TEMP_MAX_KEYS = ["max_temp_c", "temp_max_c", "tmax_c"]
CANOPY_KEYS   = ["canopy_layer", "strata", "growth_form"]

def _coerce_float(x: Any) -> Optional[float]:
    try:
        s = str(x).strip()
        if s == "" or s.lower() in {"none","nan","null"}:
            return None
        return float(s)
    except Exception:
        return None

def _first_num(traits: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in traits:
            v = _coerce_float(traits[k])
            if v is not None:
                return v
    return None

def _first_text(traits: Dict[str, Any], keys: List[str], default="Unknown") -> str:
    for k in keys:
        v = traits.get(k)
        if v:
            return str(v)
    return default

def apply_environmental_filter(species_list: List[str], site_conditions: Dict[str, Any], kb: KnowledgeBase,
                               min_compat: float = 0.4) -> List[Dict[str, Any]]:
    """Return a list of {'species','compatibility_score','traits','canopy_layer','rationale'} sorted by score."""
    trait_map = kb.get_species_traits(species_list)  # {species: {trait_name: trait_value}}
    site_ph   = _coerce_float(site_conditions.get("soil_ph"))
    site_rain = _coerce_float(site_conditions.get("annual_precip_mm"))
    site_tmp  = _coerce_float(site_conditions.get("mean_temp_c"))

    out = []
    for sp in species_list:
        traits = trait_map.get(sp, {}) or {}
        min_ph = _first_num(traits, PH_MIN_KEYS)
        max_ph = _first_num(traits, PH_MAX_KEYS)
        min_r  = _first_num(traits, RAIN_MIN_KEYS)
        max_r  = _first_num(traits, RAIN_MAX_KEYS)
        min_t  = _first_num(traits, TEMP_MIN_KEYS)
        max_t  = _first_num(traits, TEMP_MAX_KEYS)
        canopy = _first_text(traits, CANOPY_KEYS, "Mixed")

        score, facts = 1.0, []

        # pH (strong)
        if site_ph is not None and min_ph is not None and max_ph is not None:
            if not (min_ph <= site_ph <= max_ph):
                score *= 0.5
            facts.append(f"pH {min_ph}-{max_ph}")
        else:
            facts.append("pH unk")

        # rainfall (strong)
        if site_rain is not None and min_r is not None and max_r is not None:
            if not (min_r <= site_rain <= max_r):
                score *= 0.5
            facts.append(f"rain {int(min_r)}-{int(max_r)}mm")
        else:
            facts.append("rain unk")

        # temperature (milder)
        if site_tmp is not None and min_t is not None and max_t is not None:
            if not (min_t <= site_tmp <= max_t):
                score *= 0.7
            facts.append(f"temp {min_t}-{max_t}°C")
        else:
            facts.append("temp unk")

        if score >= min_compat:
            out.append({
                "species": sp,
                "compatibility_score": float(score),
                "traits": traits,
                "canopy_layer": canopy,
                "rationale": ", ".join(facts),
            })

    out.sort(key=lambda r: r["compatibility_score"], reverse=True)
    return out

# ---------------- Catalog fallback (score rows by pH/rain) ----------------
def build_catalog_from_db(db_url: str) -> pd.DataFrame:
    """Pivot species_traits and map synonyms into normalized columns."""
    species_df = _get_all_species_df(db_url)
    traits_df  = _get_all_traits_df(db_url)
    if species_df.empty:
        return pd.DataFrame(columns=["species", "min_ph", "max_ph", "min_rain_mm", "max_rain_mm", "canopy_layer", "notes"])

    pivot = pd.DataFrame({"species_key": []})
    if not traits_df.empty:
        pivot = (
            traits_df.dropna(subset=["trait_name"])
                     .pivot_table(index="species_key", columns="trait_name", values="trait_value", aggfunc="first")
                     .reset_index()
        )

    df = species_df.merge(pivot, on="species_key", how="left")
    df["species"] = df["scientific_name"].fillna(df["canonical_name"])

    def _pick_num(row, keys):  # helper to pull numeric with synonyms
        for k in keys:
            if k in row and pd.notna(row[k]):
                return _coerce_float(row[k])
        return None

    def _pick_text(row, keys):
        for k in keys:
            if k in row and pd.notna(row[k]) and str(row[k]).strip():
                return str(row[k])
        return "Unknown"

    df["min_ph"]       = df.apply(lambda r: _pick_num(r, PH_MIN_KEYS), axis=1)
    df["max_ph"]       = df.apply(lambda r: _pick_num(r, PH_MAX_KEYS), axis=1)
    df["min_rain_mm"]  = df.apply(lambda r: _pick_num(r, RAIN_MIN_KEYS), axis=1)
    df["max_rain_mm"]  = df.apply(lambda r: _pick_num(r, RAIN_MAX_KEYS), axis=1)
    df["canopy_layer"] = df.apply(lambda r: _pick_text(r, CANOPY_KEYS), axis=1)
    df["notes"]        = df.get("notes", pd.Series([None]*len(df)))

    keep = ["species", "min_ph", "max_ph", "min_rain_mm", "max_rain_mm", "canopy_layer", "notes"]
    return df[keep].dropna(subset=["species"]).reset_index(drop=True)

def compatibility_fallback(site: Dict[str, Any], catalog: pd.DataFrame, strictness: str, k: int = 20) -> pd.DataFrame:
    """Score species by pH and rainfall compatibility; return top-k rows."""
    if catalog.empty:
        return pd.DataFrame(columns=["species", "score", "rationale", "canopy_layer", "notes"])

    ph   = _coerce_float(site.get("soil_ph"))
    rain = _coerce_float(site.get("annual_precip_mm"))

    def score_row(r: pd.Series) -> float:
        s = 0.0
        # pH (60%)
        if ph is not None and pd.notna(r.get("min_ph")) and pd.notna(r.get("max_ph")):
            if r["min_ph"] <= ph <= r["max_ph"]:
                s += 0.6
            else:
                d = min(abs(ph - r["min_ph"]), abs(ph - r["max_ph"]))
                if strictness == "Strict":
                    s += 0.0
                elif strictness == "Soft":
                    s += max(0.0, 0.6 - 0.15 * d)
                else:
                    s += max(0.0, 0.6 - 0.25 * d)
        # rainfall (40%)
        if rain is not None and pd.notna(r.get("min_rain_mm")) and pd.notna(r.get("max_rain_mm")):
            if r["min_rain_mm"] <= rain <= r["max_rain_mm"]:
                s += 0.4
            else:
                d = min(abs(rain - r["min_rain_mm"]), abs(rain - r["max_rain_mm"]))
                if strictness == "Strict":
                    s += 0.0
                elif strictness == "Soft":
                    s += max(0.0, 0.4 - 0.00006 * d)
                else:
                    s += max(0.0, 0.4 - 0.0001 * d)
        return s

    df = catalog.copy()
    if "species" not in df.columns:
        df["species"] = [f"species_{i}" for i in range(len(df))]

    df["score"] = df.apply(score_row, axis=1)
    rows = []
    for _, r in df.sort_values("score", ascending=False).head(k).iterrows():
        rationale = []
        if ph is not None and pd.notna(r.get("min_ph")) and pd.notna(r.get("max_ph")):
            rationale.append(f"pH {'within' if r['min_ph'] <= ph <= r['max_ph'] else 'near'} {r['min_ph']}-{r['max_ph']}")
        if rain is not None and pd.notna(r.get("min_rain_mm")) and pd.notna(r.get("max_rain_mm")):
            rationale.append(
                f"rain {'within' if r['min_rain_mm'] <= rain <= r['max_rain_mm'] else 'near'} "
                f"{int(r['min_rain_mm'])}-{int(r['max_rain_mm'])} mm"
            )
        if not rationale:
            rationale.append("Insufficient trait ranges in DB.")
        rows.append({
            "species": str(r.get("species", "Unknown")),
            "score": float(r.get("score", 0.0)),
            "rationale": "; ".join(rationale),
            "canopy_layer": r.get("canopy_layer") or "Unknown",
            "notes": r.get("notes"),
        })
    return pd.DataFrame(rows)

# ---------------- Main ----------------
if run_clicked:
    # Parse AOI
    try:
        coords = [float(c.strip()) for c in aoi_input_str.split(",")]
        if len(coords) != 4:
            raise ValueError("Provide exactly 4 numbers: lon1,lat1,lon2,lat2")
        aoi_geom = ee.Geometry.Rectangle(coords)
    except Exception as e:
        st.error(f"Invalid AOI coordinates: {e}")
        st.stop()

    st.header("📊 Comprehensive Restoration Plan")

    with st.spinner("Analyzing site with GEE and running AI models..."):
        # --- Site fingerprint via Earth Engine ---
        site_fingerprint = get_site_fingerprint(aoi_geom)
        if site_fingerprint.get("status") != "Success":
            st.error(f"Site analysis failed: {site_fingerprint.get('message')}")
            st.stop()

        # --- AOI download (real attempt first; then placeholder-friendly helper) ---
        downloaded = try_real_aoi_download(aoi_geom, AOI_TIF_PATH, scale=10)
        if not downloaded:
            # Prepare directories and continue; no actual export in demo helper
            _ = download_aoi_as_geotiff(aoi_geom, AOI_TIF_PATH)

        # --- Optional local inference on AOI GeoTIFF ---
        can_run_resnet = resnet_ok and AOI_TIF_PATH.exists()
        can_run_unet   = unet_ok and AOI_TIF_PATH.exists()

        resnet_map = None
        unet_map = None
        if can_run_resnet:
            try:
                resnet_map = run_resnet_inference(RESNET_MODEL_PATH, AOI_TIF_PATH, num_classes=12)
                if isinstance(resnet_map, np.ndarray):
                    _save_labelmap_png(resnet_map, RESNET_PNG)
            except Exception as e:
                logging.warning(f"ResNet inference skipped: {e}")
        if can_run_unet:
            try:
                unet_map = run_unet_inference(UNET_MODEL_PATH, AOI_TIF_PATH, num_classes=3)
                if isinstance(unet_map, np.ndarray):
                    _save_labelmap_png(unet_map, UNET_PNG)
            except Exception as e:
                logging.warning(f"U-Net inference skipped: {e}")

        # --- Semantic recommendations (candidate list) ---
        semantic_species: List[str] = []
        if recommender is not None:
            try:
                # ask for more, filter down later
                semantic_species = recommender.recommend_species(goal_query, k=50) or []
            except Exception as e:
                logging.warning(f"Semantic recommendation skipped: {e}")

        # --- Filter semantic candidates with micro-level env. filter ---
        filtered_semantic = []
        if semantic_species:
            filtered_semantic = apply_environmental_filter(
                semantic_species, site_fingerprint, kb, min_compat=min_compat
            )

        # --- DB-backed fallback (all species) if semantic path is empty or sparse ---
        catalog_df  = build_catalog_from_db(DB_URL)
        fallback_df = pd.DataFrame()
        if not filtered_semantic:
            fallback_df = compatibility_fallback(site_fingerprint, catalog_df, strictness=strictness, k=20)

    col1, col2 = st.columns(2, gap="large")

    # 1A) Spatial analysis rendering
    with col1:
        st.subheader("1. AI-Powered Spatial Analysis")
        # ResNet quicklook or placeholder
        if AOI_TIF_PATH.exists() and (RESNET_PNG.exists() or PLACEHOLDER_RESNET_IMG.exists()):
            if RESNET_PNG.exists():
                st.image(str(RESNET_PNG), caption="Land Cover Classification (ResNet)")
            elif PLACEHOLDER_RESNET_IMG.exists():
                st.image(str(PLACEHOLDER_RESNET_IMG), caption="Land Cover Classification (ResNet placeholder)")
        else:
            if PLACEHOLDER_RESNET_IMG.exists():
                st.image(str(PLACEHOLDER_RESNET_IMG), caption="Land Cover Classification (ResNet placeholder)")
            else:
                st.info("Add a placeholder at assets/placeholder_resnet_map.png")

        # U-Net quicklook or placeholder
        if AOI_TIF_PATH.exists() and (UNET_PNG.exists() or PLACEHOLDER_UNET_IMG.exists()):
            if UNET_PNG.exists():
                st.image(str(UNET_PNG), caption="Restoration Suitability Map (U-Net)")
            elif PLACEHOLDER_UNET_IMG.exists():
                st.image(str(PLACEHOLDER_UNET_IMG), caption="Restoration Suitability Map (U-Net placeholder)")
        else:
            if PLACEHOLDER_UNET_IMG.exists():
                st.image(str(PLACEHOLDER_UNET_IMG), caption="Restoration Suitability Map (U-Net placeholder)")
            else:
                st.info("Add a placeholder at assets/placeholder_unet_map.png")

        if not AOI_TIF_PATH.exists():
            st.caption("ℹ️ AOI GeoTIFF not found: either the download failed or GEE export isn't fully enabled.")
        if not resnet_ok or not unet_ok:
            st.caption("ℹ️ Model inference skipped: missing weights in `models/artifacts/`.")

    # 1B) Site fingerprint
    with col2:
        st.subheader("2. Site Fingerprint")
        ndvi = site_fingerprint.get("ndvi_mean", 0) or 0
        rain = site_fingerprint.get("annual_precip_mm", 0) or 0
        ph   = site_fingerprint.get("soil_ph", 0) or 0
        elev = site_fingerprint.get("elevation_mean_m", 0) or 0
        slope = site_fingerprint.get("slope_mean_deg", 0) or 0

        st.metric("Vegetation Health (NDVI)", f"{ndvi:.2f}")
        st.metric("Annual Rainfall", f"{rain:.0f} mm")
        st.metric("Avg. Soil pH", f"{ph:.2f}")
        st.metric("Elevation", f"{elev:.0f} m")
        st.metric("Slope", f"{slope:.1f}°")

        with st.expander("View Complete Site Fingerprint"):
            st.json(site_fingerprint)

    # 2) Recommended species blueprint
    st.subheader("3. Recommended Species Blueprint")

    if filtered_semantic:
        st.write("#### Intelligent Recommender (semantic → micro-filter)")
        df = pd.DataFrame(filtered_semantic)
        # Present by canopy layer
        lower = df["canopy_layer"].astype(str).str.lower().fillna("unknown")

        def _show_group(lbl: str, mask: pd.Series):
            tbl = df[mask]
            if not tbl.empty:
                st.write(f"##### {lbl}")
                st.dataframe(
                    tbl[["species", "compatibility_score", "rationale", "canopy_layer"]]
                    .rename(columns={"compatibility_score": "score"})
                    .reset_index(drop=True),
                    use_container_width=True
                )

        _show_group("Upper Canopy", lower.str.contains("upper", na=False))
        _show_group("Mid-story",   lower.str.contains("mid", na=False))
        _show_group("Shrub Layer / Understory", lower.str.contains("shrub|under", na=False, regex=True))

        with st.expander("See full semantic table"):
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

    else:
        st.caption("ℹ️ Semantic recommender returned no items (or traits were insufficient); using DB-driven compatibility instead.")

    if not filtered_semantic:
        if fallback_df is None or fallback_df.empty:
            st.warning("No species could be recommended with the available DB traits.")
            with st.expander("Why am I seeing few/no species?"):
                catalog_df = build_catalog_from_db(DB_URL)
                stats = {
                    "species_in_db": int(len(_get_all_species_df(DB_URL))),
                    "traits_rows": int(len(_get_all_traits_df(DB_URL))),
                    "catalog_with_min_ph": int((catalog_df["min_ph"].notna().sum() if "min_ph" in catalog_df else 0)),
                    "catalog_with_max_ph": int((catalog_df["max_ph"].notna().sum() if "max_ph" in catalog_df else 0)),
                    "catalog_with_rain_min": int((catalog_df["min_rain_mm"].notna().sum() if "min_rain_mm" in catalog_df else 0)),
                    "catalog_with_rain_max": int((catalog_df["max_rain_mm"].notna().sum() if "max_rain_mm" in catalog_df else 0)),
                }
                st.json(stats)
                st.caption("Tip: add quantitative traits in `species_traits`: "
                           "`min_ph`, `max_ph`, `min_rain_mm`, `max_rain_mm`, and optionally `canopy_layer`.")
        else:
            st.write("#### DB-driven compatibility (micro-filter)")
            df = fallback_df.copy()
            lower = df["canopy_layer"].astype(str).str.lower().fillna("unknown")

            def _show_group2(lbl: str, mask: pd.Series):
                tbl = df[mask]
                if not tbl.empty:
                    st.write(f"##### {lbl}")
                    cols = ["species", "score", "rationale"]
                    if "notes" in tbl.columns:
                        cols.append("notes")
                    st.dataframe(tbl[cols].reset_index(drop=True), use_container_width=True)

            _show_group2("Upper Canopy", lower.str.contains("upper", na=False))
            _show_group2("Mid-story",   lower.str.contains("mid", na=False))
            _show_group2("Shrub Layer / Understory", lower.str.contains("shrub|under", na=False, regex=True))

            with st.expander("See full fallback table"):
                st.dataframe(df.reset_index(drop=True), use_container_width=True)

else:
    st.info("Enter an AOI rectangle in the sidebar and click **Generate Plan** to get started.")
