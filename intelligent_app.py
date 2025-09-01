# intelligent_app.py — Map-driven AOI + Region/District detection + Water-aware species planning
# + Integrated Forest Blueprint + (optional) Predictive Forecasting
from __future__ import annotations

# --- macOS OpenMP workaround (put FIRST; dev only) ---
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------- Standard libs ----------------
import os
import sys
import json
import math
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------- Third-party ----------------
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Leaflet map integration
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw

# Optional: Plotly (we will gracefully fallback to Matplotlib if not installed)
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    px = None
    _HAS_PLOTLY = False

# Optional: requests (used for direct GEE download URL)
try:
    import requests
except Exception:
    requests = None

# Shapely for district lookup
try:
    from shapely.geometry import shape as shp_shape, Point
    from shapely.strtree import STRtree
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

# Google Earth Engine
import ee

# ---------------- Paths & Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Manthan: Living Forest Intelligence", layout="wide")
st.title("🌱 Manthan: Living Forest Intelligence")

# ---------------- App Config ----------------
DB_URL = os.getenv("MANTHAN_DB_URL", "sqlite:///data/manthan.db")
DISTRICTS_GEOJSON_PATH = os.getenv(
    "MANTHAN_DISTRICTS_GEOJSON",
    "/Users/oye.arore/Documents/GitHub/Project/Manthan/data/boundaries/India_districts.json"
)

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
from manthan_core.utils.model_runner import run_resnet_inference, run_unet_inference
from manthan_core.site_assessment.gee_pipeline import (
    get_site_fingerprint,
    download_aoi_as_geotiff,  # placeholder-friendly fallback
)
from manthan_core.utils.db_connector import KnowledgeBase

# ---------------- Predictive Forecasting (optional) ----------------
try:
    from economic.predictive_viability_assessment import run_predictive_assessment
    HAS_PREDICTIVE = True
except Exception:
    HAS_PREDICTIVE = False
    logging.warning("Predictive assessment module not available")

def format_prediction_summary(prediction_results: Dict[str, Any]) -> str:
    """Create a text summary of predictions for download/export."""
    if prediction_results.get('status') != 'success':
        return "Prediction assessment failed."

    project_type = prediction_results.get('project_type', 'unknown')
    confidence = prediction_results.get('confidence_score', 0)

    lines = []
    lines.append("MANTHAN PREDICTIVE ASSESSMENT SUMMARY")
    lines.append("="*50)
    lines.append("")
    lines.append(f"Project Type: {project_type.title()}")
    lines.append(f"Prediction Confidence: {confidence:.2f}")
    lines.append(f"Assessment Date: {prediction_results.get('timestamp', 'N/A')}")
    lines.append("")

    if project_type == 'agroforestry' and prediction_results.get('agroforestry_metrics'):
        agro = prediction_results['agroforestry_metrics']
        lines += [
            "AGROFORESTRY ECONOMIC PROJECTIONS",
            f"Annual Food Income: ₹{agro['annual_food_income']:,}",
            f"5-Year Cumulative Income: ₹{agro['cumulative_5yr_income']:,}",
            f"Payback Period: {agro['payback_period_months']/12:.1f} years",
            f"Net Profit Margin: {agro['net_profit_margin']:.1%}",
            f"Cash Flow Stability: {agro['cash_flow_stability']:.2f}",
            f"Market Risk Factor: {agro['market_risk_factor']:.2f}",
            f"Crop Diversification Index: {agro['diversification_index']:.2f}",
            f"Annual Labor Hours: {agro['farmer_labor_hours_annual']:,}",
            "",
        ]
    elif project_type == 'miyawaki' and prediction_results.get('miyawaki_metrics'):
        miy = prediction_results['miyawaki_metrics']
        lines += [
            "MIYAWAKI CARBON SEQUESTRATION PROJECTIONS",
            f"Total Carbon Sequestration (20 years): {miy['carbon_sequestration_total_tons']:.1f} tons CO₂",
            f"Carbon Credits Revenue (20 years): ₹{miy['carbon_credits_revenue_20yr']:,}",
            f"Carbon Payback Period: {miy['carbon_payback_period_years']} years",
            f"Long-term ROI: {miy['long_term_roi_percentage']:.1f}%",
            f"Biodiversity Enhancement Score: {miy['biodiversity_enhancement_score']:.2f}",
            f"Restoration Success Probability: {miy['restoration_success_probability']:.2f}",
            f"Ecological Resilience Index: {miy['ecological_resilience_index']:.2f}",
            f"Ecosystem Services Value (20 years): ₹{miy['ecosystem_services_value']:,}",
            "",
        ]

    risks = prediction_results.get('risk_assessment', {})
    if risks:
        lines.append("RISK ASSESSMENT")
        for risk_name, risk_value in risks.items():
            level = 'LOW' if risk_value < 0.3 else 'MODERATE' if risk_value < 0.6 else 'HIGH'
            lines.append(f"{risk_name.replace('_',' ').title()}: {risk_value:.2f} ({level})")

    recs = prediction_results.get('recommendations', [])
    if recs:
        lines += ["", "RECOMMENDATIONS"]
        lines += [f"{i+1}. {r}" for i, r in enumerate(recs)]

    lines += ["", "="*50, "Generated by Manthan Forest Intelligence System", "Predictive models based on LightGBM machine learning"]
    return "\n".join(lines)

# ---------------- Water Integration (robust import) ----------------
import importlib.util
from importlib.machinery import SourceFileLoader

_HAS_WATER = False
_WATER_IMPORT_ERR = None
try:
    from manthan_core.water_management.enhanced_water_management_system import (
        EnhancedWaterManagementSystem,
        integrate_water_management_with_recommendations,
    )
    _HAS_WATER = True
except Exception as e1:
    try:
        from enhanced_water_management_system import (
            EnhancedWaterManagementSystem,
            integrate_water_management_with_recommendations,
        )
        _HAS_WATER = True
    except Exception as e2:
        default_candidate = "/Users/oye.arore/Documents/GitHub/Project/Manthan/manthan_core/water_management/enhanced_water_management_system.py"
        raw_path = os.getenv("MANTHAN_WATER_PATH") or default_candidate
        candidate = raw_path.strip().replace("\n", "").replace("\r", "")
        if candidate.endswith(".py.py"):
            candidate = candidate[:-3]

        if Path(candidate).exists():
            try:
                spec = importlib.util.spec_from_loader(
                    "manthan_water_fallback",
                    SourceFileLoader("manthan_water_fallback", candidate),
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                EnhancedWaterManagementSystem = getattr(mod, "EnhancedWaterManagementSystem")
                integrate_water_management_with_recommendations = getattr(mod, "integrate_water_management_with_recommendations")
                _HAS_WATER = True
            except Exception as e3:
                _WATER_IMPORT_ERR = f"{e1} / {e2} / {e3}"
        else:
            _WATER_IMPORT_ERR = f"{e1} / {e2} / file not found: {candidate}"

if not _HAS_WATER:
    st.warning(f"Dashboard - Water system not found. Continuing without water optimization. Details: {_WATER_IMPORT_ERR}")

# =============================================================================
# AOI PICKER HELPERS
# =============================================================================
def _bbox_from_geojson(d: dict) -> Optional[List[float]]:
    """Return [lon1, lat1, lon2, lat2] from a polygon/rectangle GeoJSON feature."""
    try:
        geom = d.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords:
            return None
        ring = coords[0]  # outer ring for Polygon/Rectangle
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        return [min(lons), min(lats), max(lons), max(lats)]
    except Exception:
        return None

def aoi_leaflet_picker(initial_coords: Optional[List[float]]) -> Optional[List[float]]:
    """
    Render a Leaflet map with draw tools. Returns bbox [lon1, lat1, lon2, lat2]
    if the user draws/edits a shape; otherwise None.
    """
    # Fallback center if initial not provided
    if (not initial_coords) or len(initial_coords) != 4:
        center = [27.505, 78.105]
    else:
        lon1, lat1, lon2, lat2 = initial_coords
        center = [(lat1 + lat2) / 2, (lon1 + lon2) / 2]

    m = folium.Map(location=center, zoom_start=12, control_scale=True)

    # Optional: show initial bbox (if available)
    if (initial_coords is not None) and (len(initial_coords) == 4):
        lon1, lat1, lon2, lat2 = initial_coords
        folium.Rectangle(
            bounds=[[lat1, lon1], [lat2, lon2]],
            fill=True, fill_opacity=0.1, color="#3388ff", weight=2
        ).add_to(m)

    Draw(
        draw_options={
            "polyline": False, "circle": False, "circlemarker": False,
            "marker": False, "polygon": True, "rectangle": True
        },
        edit_options={"edit": True, "remove": True}
    ).add_to(m)

    out = st_folium(
        m,
        height=480,
        width=None,
        key="aoi_map",  # explicit unique key
        returned_objects=["all_drawings", "last_active_drawing", "last_drawn"]
    )

    # Prefer most recent drawing first
    gj = out.get("last_active_drawing") or out.get("last_drawn")
    if isinstance(gj, dict):
        bbox = _bbox_from_geojson(gj)
        if bbox:
            return bbox

    # If nothing “last”, try any existing drawings (take the last)
    drawings = out.get("all_drawings") or []
    if isinstance(drawings, list) and len(drawings) > 0 and isinstance(drawings[-1], dict):
        bbox = _bbox_from_geojson(drawings[-1])
        if bbox:
            return bbox

    return None

# =============================================================================
# EE INIT (quiet)
# =============================================================================
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

# =============================================================================
# Health checks + resources
# =============================================================================
def _sqlite_path_from_url(db_url: str) -> Path:
    return Path(db_url.replace("sqlite:///", "", 1)) if db_url.startswith("sqlite:///") else Path(db_url)

db_ok     = _sqlite_path_from_url(DB_URL).exists()
resnet_ok = RESNET_MODEL_PATH.exists()
unet_ok   = UNET_MODEL_PATH.exists()

@st.cache_resource(show_spinner=False)
def get_kb():
    try:
        return KnowledgeBase(DB_URL)
    except Exception as e:
        logging.error(f"Failed to create KnowledgeBase: {e}")
        return None

kb = get_kb()
if kb is None:
    st.error("Database connection failed. Please ensure data/manthan.db exists and is accessible.")
    st.stop()

with st.sidebar:
    st.header("🧠 System Health")
    st.write(f"Database: {'✅ Found' if db_ok else '❌ Not Found'}")
    st.write(f"ResNet Model: {'✅ Found' if resnet_ok else '❌ Not Found'}")
    st.write(f"U-Net Model: {'✅ Found' if unet_ok else '❌ Not Found'}")
    st.caption(f"EE Project: `{ee_project}`")
    if not _HAS_WATER:
        st.caption("Water optimization: disabled")

# =============================================================================
# Regional boundaries + helpers
# =============================================================================
REGIONAL_BOUNDARIES: Dict[str, Dict[str, Tuple[float, float]]] = {
    'western_ghats_north': {'lat_range': (17.0, 20.8), 'lon_range': (72.8, 75.3)},
    'western_ghats_central': {'lat_range': (13.0, 16.5), 'lon_range': (74.0, 76.0)},
    'western_ghats_south': {'lat_range': (8.0, 12.2), 'lon_range': (76.0, 77.5)},
    'eastern_ghats_north': {'lat_range': (18.0, 21.0), 'lon_range': (83.0, 85.5)},
    'eastern_ghats_south': {'lat_range': (13.0, 16.0), 'lon_range': (79.0, 81.0)},
    'himalaya_uttarakhand': {'lat_range': (29.5, 31.5), 'lon_range': (79.0, 81.1)},
    'himalaya_himachal': {'lat_range': (31.0, 32.8), 'lon_range': (76.0, 78.0)},
    'kashmir_valley': {'lat_range': (33.5, 34.7), 'lon_range': (74.0, 75.7)},
    'meghalaya_hills': {'lat_range': (25.0, 26.5), 'lon_range': (90.0, 92.7)},
    'assam_floodplains': {'lat_range': (26.5, 27.7), 'lon_range': (91.0, 95.0)},
    'thar_desert_core': {'lat_range': (25.0, 28.5), 'lon_range': (70.0, 73.5)},
    'rann_of_kutch': {'lat_range': (23.0, 24.8), 'lon_range': (68.5, 71.0)},
    'sundarbans': {'lat_range': (21.5, 22.6), 'lon_range': (88.0, 89.6)},
    'deccan_plateau_core': {'lat_range': (15.0, 20.0), 'lon_range': (75.0, 80.0)},
    'gangetic_plains': {'lat_range': (24.0, 30.0), 'lon_range': (77.0, 88.0)},
    'central_india': {'lat_range': (20.0, 26.0), 'lon_range': (75.0, 85.0)},
    'south_india': {'lat_range': (8.0, 20.0), 'lon_range': (75.0, 85.0)},
    'northeast_india': {'lat_range': (23.0, 29.0), 'lon_range': (88.0, 97.0)},
    'northwest_india': {'lat_range': (26.0, 32.0), 'lon_range': (70.0, 78.0)}
}

def determine_region_from_coordinates(lon: float, lat: float) -> str:
    matching: List[Tuple[str, int]] = []
    for region_name, bounds in REGIONAL_BOUNDARIES.items():
        lat_range = bounds['lat_range']; lon_range = bounds['lon_range']
        if (lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]):
            if region_name in ['gangetic_plains', 'central_india', 'south_india', 'northeast_india', 'northwest_india']:
                matching.append((region_name, 1))  # broader: lower priority
            else:
                matching.append((region_name, 2))  # specific: higher priority
    if matching:
        matching.sort(key=lambda x: x[1], reverse=True)
        return matching[0][0]
    if lat > 28:
        return 'northwest_india' if lon < 78 else 'northeast_india'
    elif lat > 23:
        return 'central_india' if lon < 78 else 'gangetic_plains'
    else:
        return 'south_india' if lon < 80 else 'central_india'

# =============================================================================
# District lookup (cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_district_index(geojson_path: str):
    """Load districts geojson and build a spatial index. Works with Shapely 1.x and 2.x."""
    if not _HAS_SHAPELY:
        return None

    p = Path(geojson_path)
    if not p.exists():
        alt = Path(str(p).replace("India_districts.json", "india_districts.json"))
        if alt.exists():
            p = alt
        else:
            logging.warning(f"Districts file not found: {geojson_path}")
            return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            gj = json.load(f)
        feats = gj.get("features", [])
        geoms = []
        props = []
        for ft in feats:
            geom = ft.get("geometry")
            if not geom:
                continue
            try:
                poly = shp_shape(geom)
            except Exception:
                continue
            geoms.append(poly)
            pr = ft.get("properties", {}) or {}
            props.append({
                "district": pr.get("district", pr.get("DISTRICT", pr.get("District", "Unknown"))),
                "state": pr.get("state", pr.get("STATE", pr.get("State", "Unknown")))
            })
        if not geoms:
            return None
        tree = STRtree(geoms)
        return {"tree": tree, "geoms": geoms, "props": props}
    except Exception as e:
        logging.warning(f"Failed loading districts: {e}")
        return None


def find_district(lon: float, lat: float) -> Optional[Tuple[str, str]]:
    """Find (district, state) for a point. Handles Shapely 1.x and 2.x return types safely."""
    idx = load_district_index(DISTRICTS_GEOJSON_PATH)
    if not idx:
        return None

    pt = Point(lon, lat)
    tree = idx["tree"]
    geoms = idx["geoms"]
    props = idx["props"]

    # Shapely 2.x fast path: ask STRtree for indices directly
    try:
        indices = tree.query(pt, predicate="contains", return_geometries=False)
        if hasattr(indices, "__len__") and len(indices) == 0:
            return None
        i = int(indices[0])
        prop = props[i]
        return (prop.get("district", "Unknown"), prop.get("state", "Unknown"))
    except TypeError:
        # Fallback compatible with Shapely 1.x
        try:
            candidates = tree.query(pt)
        except Exception:
            candidates = []
        if not hasattr(candidates, "__len__") or len(candidates) == 0:
            return None
        for i, g in enumerate(geoms):
            try:
                if g.contains(pt):
                    prop = props[i]
                    return (prop.get("district", "Unknown"), prop.get("state", "Unknown"))
            except Exception:
                continue
        return None

# =============================================================================
# SQL helpers + scoring
# =============================================================================
def _sqlite_table_exists(conn: sqlite3.Connection, name: str) -> bool:
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,)
        ).fetchone()
        return row is not None
    except Exception:
        return False

def calculate_tolerance_score(site_value, min_val, max_val, opt_val):
    """Calculate tolerance score for a single environmental variable."""
    if min_val <= site_value <= max_val:
        if abs(site_value - opt_val) < (max_val - min_val) * 0.1:
            return 1.0
        distance_from_opt = abs(site_value - opt_val)
        range_width = max(max_val - min_val, 1e-6)
        return max(0.6, 1.0 - (distance_from_opt / range_width))
    else:
        return 0.3

def get_compatibility_level(score: float) -> str:
    if score >= 0.8:
        return 'Excellent'
    elif score >= 0.6:
        return 'Good'
    elif score >= 0.4:
        return 'Moderate'
    else:
        return 'Poor'

def calculate_enhanced_compatibility_scores(df: pd.DataFrame, site_conditions: Dict[str, Any]) -> pd.DataFrame:
    """Enhanced compatibility scoring with rainfall + pH (+ elevation temp proxy)."""
    rainfall = site_conditions.get('annual_precip_mm', 800)
    soil_ph = site_conditions.get('soil_ph', 7.0)
    elevation = site_conditions.get('elevation_m', 200)

    scores = []
    for _, row in df.iterrows():
        try:
            # Rainfall (50%)
            min_rain = float(row['min_rain']) if pd.notna(row['min_rain']) else 0
            max_rain = float(row['max_rain']) if pd.notna(row['max_rain']) else 3000
            opt_rain = float(row['opt_rain']) if pd.notna(row['opt_rain']) else (min_rain + max_rain) / 2
            rain_score = calculate_tolerance_score(rainfall, min_rain, max_rain, opt_rain)

            # pH (40%)
            min_ph = float(row['min_ph']) if pd.notna(row['min_ph']) else 5.0
            max_ph = float(row['max_ph']) if pd.notna(row['max_ph']) else 9.0
            opt_ph = float(row['opt_ph']) if pd.notna(row['opt_ph']) else (min_ph + max_ph) / 2
            ph_score = calculate_tolerance_score(soil_ph, min_ph, max_ph, opt_ph)

            # Temp proxy (10%) via lapse rate
            site_temp = 26 - (elevation / 1000) * 6.5
            temp_score = 1.0
            if pd.notna(row.get('min_temp')) and pd.notna(row.get('max_temp')):
                min_temp = float(row['min_temp']); max_temp = float(row['max_temp'])
                opt_temp = (min_temp + max_temp) / 2
                temp_score = calculate_tolerance_score(site_temp, min_temp, max_temp, opt_temp)

            native_bonus = 0.1 if row.get('is_native', 0) == 1 else 0.0

            overall_score = (rain_score*0.5 + ph_score*0.4 + temp_score*0.1) + native_bonus
            overall_score = float(min(1.0, overall_score))

            scores.append({
                'rainfall_score': round(rain_score, 3),
                'ph_score': round(ph_score, 3),
                'temp_score': round(temp_score, 3),
                'overall_score': round(overall_score, 3),
                'compatibility_level': get_compatibility_level(overall_score)
            })
        except Exception:
            scores.append({'rainfall_score': 0.5, 'ph_score': 0.5, 'temp_score': 0.5,
                           'overall_score': 0.5, 'compatibility_level': 'Moderate'})

    return pd.concat([df, pd.DataFrame(scores)], axis=1)

def categorize_forest_layers(canopy_layer_text):
    """Categorize species into proper 3-layer forest structure + climbers."""
    if pd.isna(canopy_layer_text) or canopy_layer_text is None:
        return 'Ground Layer'
    canopy_lower = str(canopy_layer_text).lower()
    if any(term in canopy_lower for term in ['upper', 'emergent', 'tall']):
        return 'Upper Canopy'
    elif any(term in canopy_lower for term in ['mid', 'canopy', 'tree']):
        return 'Mid Canopy'
    elif any(term in canopy_lower for term in ['climb', 'vine', 'liana']):
        return 'Climbers'
    else:
        return 'Ground Layer'

# --- Goal scoring helpers (no economics, just food/byproduct signals) ---
FOOD_KEYWORDS = [
    # (abbreviated for brevity; keep full list from your DB domain)
    'mangifera','mango','artocarpus','jackfruit','cocos','coconut','anacardium','cashew',
    'psidium','guava','punica','pomegranate','syzygium','jamun','tamarindus','tamarind',
    'moringa','musa','banana','amla','citrus','litchi','annona','phoenix','date',
]
BYPRODUCT_KEYWORDS = [
    'bamboo','teak','rosewood','dalbergia','santalum','eucalyptus','casuarina','acacia','neem',
    'vetiver','cymbopogon','jatropha','sesamum','gossypium','crotalaria'
]

def _has_any(name: str, keys: List[str]) -> bool:
    n = (name or "").lower()
    return any(k in n for k in keys)

def apply_agroforestry_food_priority(df: pd.DataFrame) -> pd.DataFrame:
    primary, secondary, agro_score = [], [], []
    for _, r in df.iterrows():
        nm = str(r.get('canonical_name') or '')
        fam = str(r.get('family') or '')
        edible = _has_any(nm, FOOD_KEYWORDS)
        byp    = _has_any(nm, BYPRODUCT_KEYWORDS) or ('fabaceae' in fam.lower())
        primary.append('Edible' if edible else '')
        secondary.append('Byproduct' if byp else '')
        base = float(r.get('overall_score_with_water', r.get('overall_score', 0.5)))
        bonus = 0.25 if edible else (0.12 if byp else 0.0)
        if r.get('is_native', 0) == 1:
            bonus += 0.05
        agro_score.append(min(1.0, base*0.7 + bonus))
    df['primary_category'] = primary
    df['secondary_category'] = secondary
    df['agroforestry_score'] = agro_score
    return df

def apply_miyawaki_restoration_priority(df: pd.DataFrame) -> pd.DataFrame:
    rest_score = []
    for _, r in df.iterrows():
        base = float(r.get('overall_score_with_water', r.get('overall_score', 0.5)))
        bonus = 0.0
        if r.get('is_native', 0) == 1:
            bonus += 0.15
        layer = str(r.get('forest_layer') or '')
        if layer in ('Ground Layer', 'Mid Canopy'):
            bonus += 0.05
        rest_score.append(min(1.0, base*0.8 + bonus))
    df['restoration_score'] = rest_score
    return df

# =============================================================================
# Utilities
# =============================================================================
def try_real_aoi_download(aoi: ee.Geometry, out_path: Path, scale: int = 10) -> bool:
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
        params = {"scale": scale, "crs": "EPSG:4326", "region": region, "format": "GEO_TIFF"}
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

# --- Color utilities for labelmaps (fixes "black image" issue) ---
def _ensure_label_ids(arr: np.ndarray) -> np.ndarray:
    """Turn logits/probabilities into class ids via argmax, else pass through."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] <= 64:  # (H,W,C) with small C => per-class scores
        return np.argmax(a, axis=-1).astype(np.int32)
    if a.ndim == 2:
        return a.astype(np.int32)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
        if a.ndim == 3:
            return np.argmax(a, axis=-1).astype(np.int32)
    a = np.squeeze(a)
    if a.ndim == 3:
        return np.argmax(a, axis=-1).astype(np.int32)
    return a.astype(np.int32)

def _get_palette(num_classes: int) -> np.ndarray:
    base = np.array([
        [  0,   0,   0], [230,  25,  75], [ 60, 180,  75], [255, 225,  25], [  0, 130, 200],
        [245, 130,  48], [145,  30, 180], [ 70, 240, 240], [240,  50, 230], [210, 245,  60],
        [250, 190, 190], [  0, 128, 128], [230, 190, 255], [170, 110,  40], [255, 250, 200],
        [128,   0,   0], [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128],
    ], dtype=np.uint8)
    if num_classes <= base.shape[0]:
        return base[:num_classes]
    rng = np.random.default_rng(42)
    extra = rng.integers(0, 256, size=(num_classes - base.shape[0], 3), dtype=np.uint8)
    return np.vstack([base, extra])

def _colorize_labels(label_ids: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = label_ids.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (label_ids >= 0) & (label_ids < len(palette))
    out[valid] = palette[label_ids[valid]]
    return out

def _save_labelmap_png(arr: np.ndarray, out_png: Path, num_classes_hint: Optional[int] = None) -> None:
    """Save a model labelmap (logits or ids) as a color PNG."""
    try:
        labels = _ensure_label_ids(arr)
        uniq = np.unique(labels)
        max_class = int(uniq.max()) if uniq.size else 0
        num_classes = max(num_classes_hint or (max_class + 1), 1)
        palette = _get_palette(num_classes)
        rgb = _colorize_labels(labels, palette)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(out_png)
    except Exception as e:
        logging.warning(f"Failed to save color quicklook {out_png.name}: {e}")

def auto_infer_water_priority(fingerprint: Dict[str, Any]) -> str:
    """Simple heuristic: rainfall + NDVI -> Low/Balanced/High water priority."""
    rain = float(fingerprint.get("annual_precip_mm", 800) or 800)
    ndvi = float(fingerprint.get("ndvi_mean", 0.4) or 0.4)
    if rain < 650 or (rain < 750 and ndvi < 0.35):
        return "High"
    if rain > 950 and ndvi > 0.45:
        return "Low"
    return "Balanced"

def planting_density_for_goal(goal: str) -> int:
    return 30000 if goal == "miyawaki" else 800

# ---------- Water harvesting safe invoker (handles different module signatures)
def _design_harvesting_safe(water_system, base_site_ctx: Dict[str, float], area_ha: float, balance: Dict[str, float]):
    """
    Best-effort compatibility across versions of design_water_harvesting_system:
      (site_ctx, water_deficit_m3=...), (site_ctx, total_deficit_m3=...),
      (site_ctx, deficit_m3=...), (site_ctx, water_deficit_m3 as positional),
      (site_ctx, area_ha, water_deficit_m3).
    """
    total_deficit = float(balance.get('deficit_per_hectare_m3', 0)) * float(area_ha)
    site_ctx = dict(base_site_ctx)
    site_ctx['total_area_ha'] = float(area_ha)

    attempts = [
        lambda: water_system.design_water_harvesting_system(site_ctx, water_deficit_m3=total_deficit),
        lambda: water_system.design_water_harvesting_system(site_ctx, total_deficit_m3=total_deficit),
        lambda: water_system.design_water_harvesting_system(site_ctx, deficit_m3=total_deficit),
        lambda: water_system.design_water_harvesting_system(site_ctx, total_deficit),
        lambda: water_system.design_water_harvesting_system(site_ctx, float(area_ha), total_deficit),
    ]

    last_err = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("design_water_harvesting_system: no compatible signature found")

# ---------- Integrated Forest Blueprint helpers
def _layer_proportions(goal: str) -> Dict[str, float]:
    if goal == "miyawaki":
        return {"Upper Canopy": 0.15, "Mid Canopy": 0.40, "Ground Layer": 0.35, "Climbers": 0.10}
    return {"Upper Canopy": 0.25, "Mid Canopy": 0.45, "Ground Layer": 0.25, "Climbers": 0.05}

def _spacing_from_density(density_per_ha: int) -> float:
    return math.sqrt(10000.0 / max(density_per_ha, 1))

def _pit_size_for_layer(layer: str) -> str:
    if layer == "Upper Canopy": return "60×60×60 cm"
    if layer == "Mid Canopy":   return "45×45×45 cm"
    return "30×30×30 cm"  # Ground & Climbers

def build_forest_blueprint(df: pd.DataFrame, area_ha: float, goal: str, region: str, water_priority: str, density: int) -> Dict[str, Any]:
    score_col = 'agroforestry_score' if goal == 'agroforestry' else 'restoration_score'
    layers = ["Upper Canopy", "Mid Canopy", "Ground Layer", "Climbers"]

    if 'forest_layer' not in df.columns:
        df = df.copy()
        df['forest_layer'] = df.get('canopy_layer', '').apply(categorize_forest_layers)

    total_plants = int(round(max(area_ha, 0) * max(density, 0)))
    spacing_m = _spacing_from_density(max(density, 1))

    base_props = _layer_proportions(goal)
    available = {L: max(len(df[df['forest_layer'] == L]), 0) for L in layers}
    props = base_props.copy()

    zero_layers = [L for L in layers if available.get(L, 0) == 0]
    if len(zero_layers) == 4:
        return {"total_plants": 0, "layer_plan_df": pd.DataFrame(), "species_mix_df": pd.DataFrame(), "spacing_m": spacing_m, "timeline": []}
    for z in zero_layers:
        props[z] = 0.0
    s = sum(props.values())
    if s <= 0:
        present_layers = [L for L in layers if available.get(L, 0) > 0]
        props = {L: 1.0/len(present_layers) if L in present_layers else 0.0 for L in layers}
    else:
        props = {k: v/s for k, v in props.items()}

    layer_counts = {L: int(round(total_plants * props[L])) for L in layers}
    diff = total_plants - sum(layer_counts.values())
    if diff != 0:
        order = sorted(layers, key=lambda L: props[L], reverse=True)
        i = 0
        while diff != 0 and i < len(order):
            L = order[i]
            if props[L] > 0:
                if diff > 0:
                    layer_counts[L] += 1; diff -= 1
                elif layer_counts[L] > 0:
                    layer_counts[L] -= 1; diff += 1
            i = (i + 1) % len(order)

    rows = []
    for L in layers:
        ldf = df[df['forest_layer'] == L].sort_values(score_col, ascending=False)
        if ldf.empty or layer_counts[L] <= 0:
            continue
        top = ldf.head(8)
        per = int(max(layer_counts[L] // len(top), 0))
        remainder = int(layer_counts[L] - per * len(top))
        for j, (_, r) in enumerate(top.iterrows()):
            n = per + (1 if j < remainder else 0)
            rows.append({
                "forest_layer": L,
                "canonical_name": r.get("canonical_name", ""),
                "family": r.get("family", ""),
                "is_native": r.get("is_native", 0),
                "score": float(r.get(score_col, 0)),
                "planned_count": int(n),
                "pit_size": _pit_size_for_layer(L),
            })

    species_mix = pd.DataFrame(rows)
    layer_plan = []
    for L in layers:
        count = int(species_mix[species_mix['forest_layer'] == L]['planned_count'].sum()) if not species_mix.empty else 0
        layer_plan.append({"forest_layer": L, "planned_count": count, "pit_size": _pit_size_for_layer(L)})
    layer_plan_df = pd.DataFrame(layer_plan)

    timeline = [
        {"phase": "Baseline & Layout", "month": "M0–M1", "tasks": "Survey, staking, nursery tie-ups, pit marking"},
        {"phase": "Soil & Water Prep", "month": "M1–M2", "tasks": "Pitting as per sizes, compost, mulching, swales/trenches"},
        {"phase": "Planting Window", "month": "M2–M3", "tasks": "Staggered planting; climber supports; initial irrigation"},
        {"phase": "Establishment", "month": "M3–M6", "tasks": "Weeding, mulching top-up, gap filling (≤5%)"},
        {"phase": "Stabilization", "month": "M6–M12", "tasks": "Formative pruning (minimal), pest watch, irrigation taper"},
    ]

    return {
        "region": region,
        "goal": goal,
        "water_priority": water_priority,
        "area_ha": float(area_ha),
        "density_per_ha": int(density),
        "spacing_m": float(round(spacing_m, 2)),
        "total_plants": int(total_plants),
        "layer_plan_df": layer_plan_df,
        "species_mix_df": species_mix.sort_values(["forest_layer", "score"], ascending=[True, False]),
        "timeline": timeline
    }

# =============================================================================
# UI — Sidebar (single instance, explicit keys)
# =============================================================================
with st.sidebar:
    st.header("🎯 Primary Goal")
    goal_choice = st.selectbox(
        "Select project goal",
        ["Agroforestry (Food only)", "Miyawaki (Carbon & Ecological Restoration)"],
        index=0,
        key="sb_goal_choice"
    )

    st.header("📍 Location")
    if "aoi_input_str" not in st.session_state:
        st.session_state["aoi_input_str"] = "78.100000,27.500000,78.110000,27.510000"
    st.caption("Select the AOI on the map (main panel). Current AOI:")
    st.code(st.session_state["aoi_input_str"], language="text")

    with st.expander("Filters"):
        max_species = st.slider("Maximum species to show", 10, 100, 40, key="sb_max_species")
        native_only = st.checkbox("Native species only", value=False, key="sb_native_only")
        min_compatibility_sql = st.slider("Minimum compatibility score", 0.0, 1.0, 0.3, 0.1, key="sb_min_score")

    # Developer utility: clear caches/state
    if st.button("♻️ Clear cache & reset app", key="sb_clear_all"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

    run_clicked = st.button("Generate Plan", type="primary", key="sb_run")

# =============================================================================
# Map (main panel)
# =============================================================================
st.write("### 🗺️ Select Area of Interest")

# Seed map from stored AOI string
try:
    _seed = [float(x) for x in st.session_state["aoi_input_str"].split(",")]
    if len(_seed) != 4:
        _seed = None
except Exception:
    _seed = None

picked_bbox = aoi_leaflet_picker(_seed)

if picked_bbox:
    st.session_state["aoi_input_str"] = (
        f"{picked_bbox[0]:.6f},{picked_bbox[1]:.6f},{picked_bbox[2]:.6f},{picked_bbox[3]:.6f}"
    )
    st.success(f"AOI set from map: {st.session_state['aoi_input_str']}")

# =============================================================================
# MAIN
# =============================================================================
if run_clicked:
    # Parse AOI from session
    try:
        coords = [float(c.strip()) for c in st.session_state["aoi_input_str"].split(",")]
        if len(coords) != 4:
            raise ValueError("Provide exactly 4 numbers: lon1,lat1,lon2,lat2")
        aoi_geom = ee.Geometry.Rectangle(coords)
    except Exception as e:
        st.error(f"Invalid AOI (set it on the map above): {e}")
        st.stop()

    # Area from AOI (hectares)
    lon1, lat1, lon2, lat2 = coords
    lat_mid_rad = math.radians((lat1 + lat2) / 2.0)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat_mid_rad)
    width_m = abs(lon2 - lon1) * m_per_deg_lon
    height_m = abs(lat2 - lat1) * m_per_deg_lat
    area_ha = (width_m * height_m) / 10000.0

    st.header("📊 Comprehensive Restoration Plan")

    # 1) Site analysis & quicklooks
    with st.spinner("Analyzing site with GEE and preparing base layers..."):
        site_fingerprint = get_site_fingerprint(aoi_geom)
        if site_fingerprint.get("status") != "Success":
            st.error(f"Site analysis failed: {site_fingerprint.get('message')}")
            st.stop()

        downloaded = try_real_aoi_download(aoi_geom, AOI_TIF_PATH, scale=10)
        if not downloaded:
            _ = download_aoi_as_geotiff(aoi_geom, AOI_TIF_PATH)

        can_run_resnet = RESNET_MODEL_PATH.exists() and AOI_TIF_PATH.exists()
        can_run_unet   = UNET_MODEL_PATH.exists() and AOI_TIF_PATH.exists()

        if can_run_resnet:
            try:
                resnet_map = run_resnet_inference(RESNET_MODEL_PATH, AOI_TIF_PATH, num_classes=12)
                if isinstance(resnet_map, np.ndarray):
                    _save_labelmap_png(resnet_map, RESNET_PNG, num_classes_hint=12)
            except Exception as e:
                logging.warning(f"ResNet inference skipped: {e}")
        if can_run_unet:
            try:
                unet_map = run_unet_inference(UNET_MODEL_PATH, AOI_TIF_PATH, num_classes=3)
                if isinstance(unet_map, np.ndarray):
                    _save_labelmap_png(unet_map, UNET_PNG, num_classes_hint=3)
            except Exception as e:
                logging.warning(f"U-Net inference skipped: {e}")

    col1, col2 = st.columns(2, gap="large")

    # 1A) Spatial analysis rendering
    with col1:
        st.subheader("1. AI-Powered Spatial Analysis")
        if AOI_TIF_PATH.exists() and (RESNET_PNG.exists() or PLACEHOLDER_RESNET_IMG.exists()):
            st.image(str(RESNET_PNG if RESNET_PNG.exists() else PLACEHOLDER_RESNET_IMG),
                     caption="Land Cover Classification (ResNet)", use_container_width=True)
        else:
            st.info("Add a placeholder at assets/placeholder_resnet_map.png")

        if AOI_TIF_PATH.exists() and (UNET_PNG.exists() or PLACEHOLDER_UNET_IMG.exists()):
            st.image(str(UNET_PNG if UNET_PNG.exists() else PLACEHOLDER_UNET_IMG),
                     caption="Restoration Suitability Map (U-Net)", use_container_width=True)
        else:
            st.info("Add a placeholder at assets/placeholder_unet_map.png")

        if not AOI_TIF_PATH.exists():
            st.caption("ℹ️ AOI GeoTIFF not found: either the download failed or GEE export isn't fully enabled.")
        if not RESNET_MODEL_PATH.exists() or not UNET_MODEL_PATH.exists():
            st.caption("ℹ️ Model inference skipped: missing weights in `models/artifacts/`.")

        with st.expander("Debug: label stats", expanded=False):
            try:
                if 'resnet_map' in locals() and isinstance(resnet_map, np.ndarray):
                    ids = _ensure_label_ids(resnet_map)
                    st.write("ResNet classes:", np.unique(ids).tolist())
                if 'unet_map' in locals() and isinstance(unet_map, np.ndarray):
                    ids = _ensure_label_ids(unet_map)
                    st.write("U-Net classes:", np.unique(ids).tolist())
            except Exception as e:
                st.caption(f"Label stats unavailable: {e}")

    # 1B) Site fingerprint + admin tags
    with col2:
        st.subheader("2. Site Fingerprint")
        ndvi = float(site_fingerprint.get("ndvi_mean", 0) or 0)
        rain = float(site_fingerprint.get("annual_precip_mm", 0) or 0)
        ph   = float(site_fingerprint.get("soil_ph", 0) or 0)
        elev = float(site_fingerprint.get("elevation_mean_m", 0) or 0)
        slope = float(site_fingerprint.get("slope_mean_deg", 0) or 0)

        st.metric("Vegetation Health (NDVI)", f"{ndvi:.2f}")
        st.metric("Annual Rainfall", f"{rain:.0f} mm")
        st.metric("Avg. Soil pH", f"{ph:.2f}")
        st.metric("Elevation", f"{elev:.0f} m")
        st.metric("Slope", f"{slope:.1f}°")
        st.metric("AOI Area", f"{area_ha:.2f} ha")

        center_lon = (lon1 + lon2) / 2
        center_lat = (lat1 + lat2) / 2
        region_name = determine_region_from_coordinates(center_lon, center_lat)
        dist_info = find_district(center_lon, center_lat)
        if dist_info:
            st.info(f"📍 Region: **{region_name.replace('_',' ').title()}**  •  District: **{dist_info[0]}**, **{dist_info[1]}**")
        else:
            st.info(f"📍 Region: **{region_name.replace('_',' ').title()}**  •  District: *(not found)*")

        with st.expander("View Complete Site Fingerprint"):
            st.json(site_fingerprint)

    # 2) Auto water priority (no user input)
    primary_goal = "agroforestry" if "Agroforestry" in goal_choice else "miyawaki"
    water_priority = auto_infer_water_priority(site_fingerprint)
    st.subheader("3. Automatic Water Optimization")
    st.write(f"Water priority inferred as **{water_priority}** based on rainfall & vegetation.")

    # 3) Recommended species blueprint — Regional with goal/water aware scoring
    st.subheader("4. Recommended Species")

    def get_region_filtered_recommendations(site_conditions: Dict[str, Any],
                                            aoi_coords: List[float],
                                            primary_goal: str,
                                            water_priority: str,
                                            max_results: int = 40) -> Tuple[pd.DataFrame, str]:
        """Get species recommendations filtered by detected region and ranked by goal-aware score."""
        center_lon = (aoi_coords[0] + aoi_coords[2]) / 2
        center_lat = (aoi_coords[1] + aoi_coords[3]) / 2
        primary_region = determine_region_from_coordinates(center_lon, center_lat)

        rainfall = site_conditions.get('annual_precip_mm', 800)
        soil_ph = site_conditions.get('soil_ph', 7.0)

        sqlite_path = _sqlite_path_from_url(DB_URL)
        conn = sqlite3.connect(str(sqlite_path))

        regional_query = """
        SELECT DISTINCT 
            s.canonical_name,
            s.scientific_name,
            s.family,
            s.species_key,
            sd.region_name,
            sd.is_native,
            st_rain_min.trait_value as min_rain,
            st_rain_max.trait_value as max_rain,
            st_rain_opt.trait_value as opt_rain,
            st_ph_min.trait_value as min_ph,
            st_ph_max.trait_value as max_ph,
            st_ph_opt.trait_value as opt_ph,
            st_temp_min.trait_value as min_temp,
            st_temp_max.trait_value as max_temp,
            st_drought.trait_value as drought_tolerance,
            st_canopy.trait_value as canopy_layer,
            st_succession.trait_value as successional_role
        FROM species s
        JOIN species_distribution sd ON s.species_key = sd.species_key
        JOIN species_traits st_rain_min ON s.species_key = st_rain_min.species_key 
            AND st_rain_min.trait_name = 'min_rainfall_mm'
        JOIN species_traits st_rain_max ON s.species_key = st_rain_max.species_key 
            AND st_rain_max.trait_name = 'max_rainfall_mm'
        JOIN species_traits st_ph_min ON s.species_key = st_ph_min.species_key 
            AND st_ph_min.trait_name = 'min_ph'
        JOIN species_traits st_ph_max ON s.species_key = st_ph_max.species_key 
            AND st_ph_max.trait_name = 'max_ph'
        LEFT JOIN species_traits st_rain_opt ON s.species_key = st_rain_opt.species_key 
            AND st_rain_opt.trait_name = 'optimal_rainfall_mm'
        LEFT JOIN species_traits st_ph_opt ON s.species_key = st_ph_opt.species_key 
            AND st_ph_opt.trait_name = 'optimal_ph'
        LEFT JOIN species_traits st_temp_min ON s.species_key = st_temp_min.species_key 
            AND st_temp_min.trait_name = 'min_temp_c'
        LEFT JOIN species_traits st_temp_max ON s.species_key = st_temp_max.species_key 
            AND st_temp_max.trait_name = 'max_temp_c'
        LEFT JOIN species_traits st_drought ON s.species_key = st_drought.species_key 
            AND st_drought.trait_name = 'drought_tolerance'
        LEFT JOIN species_traits st_canopy ON s.species_key = st_canopy.species_key 
            AND st_canopy.trait_name = 'canopy_layer'
        LEFT JOIN species_traits st_succession ON s.species_key = st_succession.species_key 
            AND st_succession.trait_name = 'successional_role'
        WHERE sd.region_name = ?
          AND CAST(st_rain_min.trait_value AS REAL) <= ?
          AND CAST(st_rain_max.trait_value AS REAL) >= ?
          AND CAST(st_ph_min.trait_value AS REAL) <= ?
          AND CAST(st_ph_max.trait_value AS REAL) >= ?
        """

        params = [primary_region, rainfall, rainfall, soil_ph, soil_ph]
        final_query = regional_query + f" LIMIT {max_results*2}"

        try:
            df = pd.read_sql_query(final_query, conn, params=params)
            if df.empty:
                broader_query = regional_query.replace("WHERE sd.region_name = ?", "WHERE 1=1")
                df = pd.read_sql_query(broader_query, conn, params=params[1:])
        except Exception as e:
            st.error(f"Database query failed: {e}")
            df = pd.DataFrame()
        finally:
            conn.close()

        if df.empty:
            return df, primary_region

        df = calculate_enhanced_compatibility_scores(df, site_conditions)

        if 'canopy_layer' in df.columns:
            df['forest_layer'] = df['canopy_layer'].apply(categorize_forest_layers)
        else:
            df['forest_layer'] = 'Ground Layer'

        if _HAS_WATER:
            water_weight_map = {"Low": 0.10, "Balanced": 0.25, "High": 0.40}
            water_weight = water_weight_map.get(water_priority, 0.25)
            try:
                df = integrate_water_management_with_recommendations(df, site_conditions, water_weight)
            except Exception as e:
                st.warning(f"Water optimization failed: {e}")

        if primary_goal == "agroforestry":
            df = apply_agroforestry_food_priority(df)
            score_col_local = 'agroforestry_score'
        else:
            df = apply_miyawaki_restoration_priority(df)
            score_col_local = 'restoration_score'

        df = df.sort_values(score_col_local, ascending=False, ignore_index=True)
        return df.head(max_results), primary_region

    regional_df, region = get_region_filtered_recommendations(
        {
            "annual_precip_mm": site_fingerprint.get("annual_precip_mm"),
            "soil_ph": site_fingerprint.get("soil_ph"),
            "elevation_m": site_fingerprint.get("elevation_mean_m", 200),
        },
        coords,
        primary_goal=primary_goal,
        water_priority=water_priority,
        max_results=max_species
    )

    if regional_df.empty:
        st.error("❌ No compatible species found for these conditions.")
        st.stop()

    score_col = 'agroforestry_score' if primary_goal == 'agroforestry' else 'restoration_score'
    df = regional_df.copy()
    if native_only and 'is_native' in df.columns:
        df = df[df['is_native'] == 1]
    if 'overall_score' in df.columns:
        df = df[df['overall_score'] >= float(min_compatibility_sql)]
    if df.empty:
        st.warning("No species meet the selected filters. Try relaxing them.")
        st.stop()

    st.success(f"✅ Found **{len(df)}** goal- and water-optimized species for **{region.replace('_', ' ').title()}**")

    excellent = (df['compatibility_level'] == 'Excellent').sum() if 'compatibility_level' in df.columns else 0
    good = (df['compatibility_level'] == 'Good').sum() if 'compatibility_level' in df.columns else 0
    moderate = (df['compatibility_level'] == 'Moderate').sum() if 'compatibility_level' in df.columns else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🟢 Excellent", int(excellent))
    with c2: st.metric("🟡 Good", int(good))
    with c3: st.metric("🟠 Moderate", int(moderate))
    with c4: st.metric("Avg Score", f"{df[score_col].mean():.2f}")

    if 'canopy_layer' in df.columns:
        df['forest_layer'] = df['canopy_layer'].apply(categorize_forest_layers)
    else:
        df['forest_layer'] = 'Ground Layer'

    layer_order = ['Upper Canopy', 'Mid Canopy', 'Ground Layer', 'Climbers']
    layer_icons = {'Upper Canopy': '🌲', 'Mid Canopy': '🌳', 'Ground Layer': '🌿', 'Climbers': '🍃'}
    show_agro = (primary_goal == "agroforestry")

    for layer in layer_order:
        layer_species = df[df['forest_layer'] == layer]
        if not layer_species.empty:
            layer_species = layer_species.sort_values(score_col, ascending=False)
            with st.expander(f"{layer_icons[layer]} **{layer}** ({len(layer_species)} species)", expanded=True):
                for i, (_, row) in enumerate(layer_species.head(10).iterrows()):
                    level = row.get('compatibility_level', '')
                    color = '🟢' if level == 'Excellent' else ('🟡' if level == 'Good' else '🟠')
                    native_indicator = '🏠' if row.get('is_native', 0) == 1 else '🌍'
                    water_score = row.get('water_efficiency_score', None)
                    water_tag = f" • 💧 {water_score:.2f}" if isinstance(water_score, (float, int)) else ""
                    agro_tags = ""
                    if show_agro:
                        prim = row.get('primary_category', '')
                        sec = row.get('secondary_category', '')
                        parts = []
                        if prim: parts.append(f"Primary: {prim}")
                        if sec: parts.append(f"Secondary: {sec}")
                        if parts:
                            agro_tags = " • " + " | ".join(parts)
                    st.write(f"**{i+1}. {color} {row['canonical_name']}** ({row.get('family','')}) — Score: **{row[score_col]:.2f}**{water_tag}{agro_tags} • {native_indicator}")

    # Detailed table
    st.subheader("📊 Detailed List")
    show_cols = [c for c in [
        'canonical_name', 'family', 'forest_layer', score_col,
        'overall_score', 'water_efficiency_score', 'drought_tolerance', 'is_native',
        'primary_category', 'secondary_category'
    ] if c in df.columns]
    tbl = df[show_cols].copy()
    if 'is_native' in tbl.columns:
        tbl['is_native'] = tbl['is_native'].map({1:'✅',0:'❌'})
    st.dataframe(tbl.head(30), use_container_width=True)

    # 5) Water balance AFTER species recommendation (auto density by goal)
    if _HAS_WATER:
        st.subheader("5. Water Balance & Harvesting (Based on Selected Species)")
        try:
            water_system = EnhancedWaterManagementSystem()
            top_species = df['canonical_name'].head(20).tolist()
            density = planting_density_for_goal(primary_goal)
            balance = water_system.calculate_comprehensive_water_balance(
                top_species,
                {
                    'annual_precip_mm': float(site_fingerprint.get("annual_precip_mm", 800) or 800),
                    'slope_deg': float(site_fingerprint.get("slope_mean_deg", 5) or 5),
                    'ndvi_mean': float(site_fingerprint.get("ndvi_mean", 0.4) or 0.4),
                    'elevation_m': float(site_fingerprint.get("elevation_mean_m", 200) or 200),
                    'soil_ph': float(site_fingerprint.get("soil_ph", 7.0) or 7.0),
                },
                planting_density=density
            )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                eff = float(balance.get('water_efficiency_ratio', 0))
                level = str(balance.get('water_stress_level', 'Unknown'))
                badge = "🟢" if eff >= 0.8 else ("🟡" if eff >= 0.5 else "🔴")
                st.metric("Water Efficiency", f"{eff:.2f}", delta=f"{badge} {level}")
            with c2:
                st.metric("Effective Rainfall", f"{float(balance.get('effective_rainfall_mm', 0)):.0f} mm")
            with c3:
                deficit = float(balance.get('deficit_per_hectare_m3', 0))
                surplus = float(balance.get('surplus_per_hectare_m3', 0))
                if deficit > 0:
                    st.metric("Water Deficit", f"{deficit:.0f} m³/ha", delta="🚰 Irrigation needed")
                else:
                    st.metric("Water Surplus", f"{surplus:.0f} m³/ha", delta="💧 Self-sufficient")
            with c4:
                st.metric("Avg Species Demand", f"{float(balance.get('average_species_demand_mm', 0)):.0f} mm/yr")

            if float(balance.get('deficit_per_hectare_m3', 0)) > 0:
                st.markdown("**🏗️ Suggested Water Harvesting Structures**")
                site_ctx_for_harvesting = {
                    'annual_precip_mm': float(site_fingerprint.get("annual_precip_mm", 800) or 800),
                    'slope_deg': float(site_fingerprint.get("slope_mean_deg", 5) or 5),
                    'ndvi_mean': float(site_fingerprint.get("ndvi_mean", 0.4) or 0.4),
                    'elevation_m': float(site_fingerprint.get("elevation_mean_m", 200) or 200),
                    'soil_ph': float(site_fingerprint.get("soil_ph", 7.0) or 7.0),
                }
                try:
                    recs = _design_harvesting_safe(water_system, site_ctx_for_harvesting, area_ha, balance)
                    for rec in recs[:3]:
                        st.write(
                            f"- **{getattr(rec, 'structure_type', 'Structure')}** — Capacity: {getattr(rec, 'capacity_m3', 0):.0f} m³ "
                            f"• Suitability: {getattr(rec, 'suitability_score', 0):.2f} "
                            f"• Maintenance: {getattr(rec, 'maintenance_level', 'N/A')}"
                        )
                except Exception as e:
                    st.warning(f"Water harvesting design step failed: {e}")
        except Exception as e:
            st.warning(f"Water balance step failed: {e}")

    # 6) Integrated Forest Blueprint
    st.subheader("6. Integrated Forest Blueprint")
    density = planting_density_for_goal(primary_goal)
    blueprint = build_forest_blueprint(
        df=df,
        area_ha=area_ha,
        goal=primary_goal,
        region=region,
        water_priority=water_priority,
        density=density
    )

    if blueprint.get("total_plants", 0) <= 0 or blueprint.get("species_mix_df") is None:
        st.warning("Could not build a blueprint (no species available per layer).")
    else:
        cols = st.columns(4)
        with cols[0]: st.metric("Total Plants", f"{blueprint['total_plants']:,}")
        with cols[1]: st.metric("Density", f"{blueprint['density_per_ha']:,} /ha")
        with cols[2]: st.metric("Avg Spacing", f"{blueprint['spacing_m']:.2f} m")
        with cols[3]: st.metric("Area", f"{blueprint['area_ha']:.2f} ha")

        st.markdown("**Layer Plan (counts & pit sizes)**")
        st.dataframe(blueprint["layer_plan_df"], use_container_width=True)

        st.markdown("**Species Mix (allocation by layer)**")
        species_view = blueprint["species_mix_df"][[
            "forest_layer", "canonical_name", "family", "is_native", "score", "planned_count", "pit_size"
        ]].copy()
        species_view["is_native"] = species_view["is_native"].map({1:"✅",0:"❌"})
        st.dataframe(species_view, use_container_width=True, hide_index=True)
        st.markdown("**12-Month Execution Timeline**")
        for item in blueprint["timeline"]:
            st.write(f"- **{item['phase']}** ({item['month']}): {item['tasks']}")

        # Exports for blueprint
        bp_species_csv = blueprint["species_mix_df"].to_csv(index=False)
        bp_layers_csv = blueprint["layer_plan_df"].to_csv(index=False)
        st.download_button("Download Species Mix (CSV)", bp_species_csv, file_name="forest_blueprint_species_mix.csv", mime="text/csv")
        st.download_button("Download Layer Plan (CSV)", bp_layers_csv, file_name="forest_blueprint_layers.csv", mime="text/csv")

    # 7) Advanced Predictive Forecasting (optional module)
    prediction_results = None
    if HAS_PREDICTIVE and blueprint.get("total_plants", 0) > 0:
        st.subheader("7. 🔮 Advanced Predictive Forecasting & ROI Analysis")

        with st.spinner("Running LightGBM predictive models for long-term outcomes..."):
            prediction_results = run_predictive_assessment(blueprint, {
                'annual_precip_mm': float(site_fingerprint.get("annual_precip_mm", 800) or 800),
                'soil_ph': float(site_fingerprint.get("soil_ph", 7.0) or 7.0),
                'elevation_m': float(site_fingerprint.get("elevation_mean_m", 200) or 200),
                'slope_deg': float(site_fingerprint.get("slope_mean_deg", 5) or 5),
                'ndvi_mean': float(site_fingerprint.get("ndvi_mean", 0.4) or 0.4),
            })

        if prediction_results.get('status') == 'success':
            project_type = prediction_results['project_type']
            confidence = prediction_results['confidence_score']

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction Confidence", f"{confidence:.2f}",
                          delta="🎯 High" if confidence > 0.8 else "🎲 Moderate" if confidence > 0.6 else "⚠️ Low")
            with col2:
                st.metric("Optimized For", project_type.title(),
                          delta="🚜 Farmer Income" if project_type == 'agroforestry' else "🌿 Carbon Credits")

            if project_type == 'agroforestry' and prediction_results.get('agroforestry_metrics'):
                st.markdown("### 🚜 Agroforestry Economic Predictions (Short-term Focus)")
                agro = prediction_results['agroforestry_metrics']
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Annual Food Income", f"₹{agro['annual_food_income']:,.0f}")
                with c2: st.metric("5-Year Total", f"₹{agro['cumulative_5yr_income']:,.0f}")
                with c3: st.metric("Payback Period", f"{agro['payback_period_months']/12:.1f} years")
                with c4: st.metric("Profit Margin", f"{agro['net_profit_margin']:.1%}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    s_icon = "🟢" if agro['cash_flow_stability'] > 0.7 else "🟡" if agro['cash_flow_stability'] > 0.5 else "🔴"
                    st.metric("Cash Flow Stability", f"{agro['cash_flow_stability']:.2f}", delta=f"{s_icon}")
                with c2:
                    r_icon = "🟢" if agro['market_risk_factor'] < 0.3 else "🟡" if agro['market_risk_factor'] < 0.5 else "🔴"
                    st.metric("Market Risk", f"{agro['market_risk_factor']:.2f}", delta=f"{r_icon}")
                with c3:
                    st.metric("Crop Diversity", f"{agro['diversification_index']:.2f}")
                st.markdown("**👨‍🌾 Farmer Impact Analysis:**")
                st.write(f"• **Labor Requirement**: {agro['farmer_labor_hours_annual']:,.0f} hours/year ({agro['farmer_labor_hours_annual']/250:.1f} days)")
                st.write(f"• **Monthly Income**: ₹{agro['annual_food_income']/12:,.0f} (after year 3)")
                st.write(f"• **Income per Hour**: ₹{agro['annual_food_income']/agro['farmer_labor_hours_annual']:,.0f}")

            elif project_type == 'miyawaki' and prediction_results.get('miyawaki_metrics'):
                st.markdown("### 🌿 Miyawaki Carbon Sequestration Predictions (Long-term Focus)")
                miy = prediction_results['miyawaki_metrics']
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Carbon Sequestered", f"{miy['carbon_sequestration_total_tons']:.1f} tons", delta="20 years")
                with c2: st.metric("Carbon Revenue", f"₹{miy['carbon_credits_revenue_20yr']:,.0f}", delta="20 years")
                with c3: st.metric("Carbon Payback", f"{miy['carbon_payback_period_years']} years")
                with c4: st.metric("Long-term ROI", f"{miy['long_term_roi_percentage']:.1f}%")
                c1, c2, c3 = st.columns(3)
                with c1:
                    bio_icon = "🟢" if miy['biodiversity_enhancement_score'] > 0.7 else "🟡" if miy['biodiversity_enhancement_score'] > 0.5 else "🟠"
                    st.metric("Biodiversity Score", f"{miy['biodiversity_enhancement_score']:.2f}", delta=f"{bio_icon}")
                with c2:
                    succ_icon = "🟢" if miy['restoration_success_probability'] > 0.8 else "🟡" if miy['restoration_success_probability'] > 0.6 else "🔴"
                    st.metric("Success Probability", f"{miy['restoration_success_probability']:.2f}", delta=f"{succ_icon}")
                with c3:
                    res_icon = "🟢" if miy['ecological_resilience_index'] > 0.7 else "🟡" if miy['ecological_resilience_index'] > 0.5 else "🟠"
                    st.metric("Resilience Index", f"{miy['ecological_resilience_index']:.2f}", delta=f"{res_icon}")
                st.markdown("**🌍 Carbon Market Analysis:**")
                annual_carbon = miy['carbon_sequestration_total_tons'] / 20
                st.write(f"• **Annual Sequestration**: {annual_carbon:.1f} tons CO₂/year")
                st.write(f"• **Per Hectare Impact**: {annual_carbon/area_ha:.1f} tons CO₂/ha/year")
                st.write(f"• **Ecosystem Services**: ₹{miy['ecosystem_services_value']:,.0f} (20-year value)")
                st.write(f"• **Carbon Price Assumed**: ₹800/ton CO₂ (voluntary market)")

            # Risk Assessment (common)
            st.markdown("### ⚠️ Risk Assessment & Mitigation")
            risks = prediction_results['risk_assessment']
            risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
            risk_df = []
            for risk_name, risk_value in risks.items():
                risk_level = 'low' if risk_value < 0.3 else 'medium' if risk_value < 0.6 else 'high'
                risk_df.append({
                    'Risk Category': risk_name.replace('_', ' ').title(),
                    'Level': f"{risk_colors[risk_level]} {risk_level.title()}",
                    'Score': f"{risk_value:.2f}",
                    'Impact': 'Low' if risk_value < 0.3 else 'Moderate' if risk_value < 0.6 else 'High'
                })
            st.dataframe(pd.DataFrame(risk_df), use_container_width=True, hide_index=True)
            overall_risk = np.mean(list(risks.values()))
            overall_color = "🟢" if overall_risk < 0.3 else "🟡" if overall_risk < 0.6 else "🔴"
            st.metric("Overall Risk Score", f"{overall_risk:.2f}", delta=f"{overall_color}")

            st.markdown("### 💡 AI-Generated Recommendations")
            recommendations = prediction_results.get('recommendations', [])
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"**{i}.** {rec}")
            else:
                st.info("No specific recommendations generated for this configuration.")

            # Downloadable text summary
            summary_txt = format_prediction_summary(prediction_results)
            st.download_button("Download Predictive Summary (TXT)", summary_txt, file_name="predictive_summary.txt", mime="text/plain")

        else:
            st.error(f"Predictive assessment failed: {prediction_results.get('error', 'Unknown error')}")

    elif not HAS_PREDICTIVE:
        st.info("💡 **Predictive forecasting module not available.** Install optional dependencies for ML-based predictions.")

    # 8) Download — master species table
    st.subheader("📥 Export")
    csv_data = df.to_csv(index=False)
    file_stub = f"manthan_{'agro' if primary_goal=='agroforestry' else 'miyawaki'}_{region}_{lat1:.2f}N_{lon1:.2f}E"
    st.download_button(
        label="Download Species List (CSV)",
        data=csv_data,
        file_name=f"{file_stub}.csv",
        mime="text/csv"
    )

else:
    st.info("Select an AOI on the map, choose your goal, and click **Generate Plan** to get started.")
