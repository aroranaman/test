# intelligent_app.py
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

# Optional geo/map deps (all gracefully optional)
try:
    import leafmap.foliumap as leafmap
    _LEAFMAP_OK = True
except Exception:
    leafmap = None
    _LEAFMAP_OK = False

try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except Exception:
    folium = None
    st_folium = None
    _FOLIUM_OK = False

try:
    import rasterio
except Exception:
    rasterio = None

# Optional Earth Engine (for water balance + JRC water overlay)
try:
    import ee
    _EE_IMPORTED = True
except Exception:
    ee = None
    _EE_IMPORTED = False

# ------------------------------------------------------------------------------
# Bootstrap import path for local packages
# ------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ------------------------------------------------------------------------------
# Core Manthan imports (defensive)
# ------------------------------------------------------------------------------
RECOMMENDER_OK = True
MODEL_RUNNER_OK = True
DB_OK = True

try:
    from manthan_core.recommender.intelligent_recommender import IntelligentRecommender
except Exception as e:
    RECOMMENDER_OK = False
    _RECOMMENDER_IMPORT_ERR = e

try:
    from manthan_core.utils.model_runner import run_resnet_inference, run_unet_inference
except Exception as e:
    MODEL_RUNNER_OK = False
    _MODEL_RUNNER_IMPORT_ERR = e

try:
    from manthan_core.utils.db_connector import KnowledgeBase
except Exception as e:
    DB_OK = False
    _DB_IMPORT_ERR = e

# ------------------------------------------------------------------------------
# Streamlit base config
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Manthan: Living Forest Intelligence", layout="wide")
st.title("🌱 Manthan: Living Forest Intelligence")

# ------------------------------------------------------------------------------
# Config / Paths (env-overridable)
# ------------------------------------------------------------------------------
DB_URL = os.getenv("MANTHAN_DB_URL", f"sqlite:///{REPO_ROOT}/data/manthan.db")
EMBEDDINGS_DIR = Path(os.getenv("MANTHAN_EMBEDDINGS_DIR", str(REPO_ROOT / "data/embeddings")))
RESNET_MODEL_PATH = Path(os.getenv("MANTHAN_RESNET_MODEL", str(REPO_ROOT / "models/artifacts/resnet34_pan_india_weights.pth")))
UNET_MODEL_PATH = Path(os.getenv("MANTHAN_UNET_MODEL", str(REPO_ROOT / "models/artifacts/unet_pan_india_final_v1.pth")))
RESNET_NUM_CLASSES = int(os.getenv("MANTHAN_RESNET_CLASSES", "12"))
UNET_NUM_CLASSES = int(os.getenv("MANTHAN_UNET_CLASSES", "3"))

# ------------------------------------------------------------------------------
# Initialize Earth Engine (best-effort)
# ------------------------------------------------------------------------------
_EE_OK = False
if _EE_IMPORTED:
    try:
        # If user has set GOOGLE_APPLICATION_CREDENTIALS or has authenticated locally
        ee.Initialize()
        _EE_OK = True
    except Exception:
        # Leave as False; we’ll surface a gentle tip in Diagnostics
        _EE_OK = False

# ------------------------------------------------------------------------------
# Sidebar — Stage 1: INPUT (Profile → Priorities, AOI, Site file)
# ------------------------------------------------------------------------------
st.sidebar.header("1) Who are you?")
profile = st.sidebar.selectbox(
    "Select your profile to set project priorities:",
    ["Select a profile...", "Farmer (Agroforestry)", "NGO / Government (Ecological Restoration)", "Corporate (Carbon Credits)"],
)

priorities: Dict[str, str] = {}
if profile == "Farmer (Agroforestry)":
    priorities = {"primary": "short_term_economic_returns", "secondary": "food_production"}
    st.sidebar.info("Goal: Maximize early-stage income from NTFPs and edible plants.")
elif profile == "NGO / Government (Ecological Restoration)":
    priorities = {"primary": "biodiversity_uplift", "secondary": "threatened_species"}
    st.sidebar.info("Goal: Maximize native biodiversity and prioritize endangered species.")
elif profile == "Corporate (Carbon Credits)":
    priorities = {"primary": "carbon_sequestration", "secondary": "fast_growth_rate"}
    st.sidebar.info("Goal: Maximize carbon capture potential.")

st.sidebar.header("2) Where is your land?")
# AOI selection: draw on map OR upload/paste GeoJSON
aoi_geojson: Optional[Dict[str, Any]] = None

if _LEAFMAP_OK:
    m = leafmap.Map(center=[22.5, 79.0], zoom=5, draw_export=True)
    m.add_basemap("HYBRID")
    m.add_draw_control(export=False)
    st.sidebar.markdown("Draw a polygon on the map, then click **Use drawn AOI** below.")
    aoi_container = st.sidebar.empty()
else:
    st.sidebar.warning("leafmap not available; falling back to basic GeoJSON inputs.")

uploaded_geojson = st.sidebar.file_uploader("…or upload AOI GeoJSON", type=["geojson", "json"])
aoi_text = st.sidebar.text_area("…or paste AOI GeoJSON", height=120)

use_drawn = False
if _LEAFMAP_OK:
    st.sidebar.button("Use drawn AOI", key="use_drawn_btn")
    # Leafmap keeps drawn features in m.user_roi (Polygon) or m.user_rois
    if "use_drawn_btn" in st.session_state:
        use_drawn = True

if _LEAFMAP_OK:
    with st.sidebar.expander("Map (zoom & draw AOI)", expanded=True):
        m.to_streamlit(height=320)

# Resolve AOI from sources (priority: drawn > upload > pasted)
if _LEAFMAP_OK and use_drawn:
    try:
        geom = m.user_roi or (m.user_rois[0] if m.user_rois else None)
        if geom is not None:
            aoi_geojson = json.loads(geom.to_geojson())
    except Exception:
        aoi_geojson = None

if aoi_geojson is None and uploaded_geojson is not None:
    try:
        aoi_geojson = json.loads(uploaded_geojson.getvalue().decode("utf-8"))
    except Exception:
        st.sidebar.error("Uploaded AOI is not valid GeoJSON.")

if aoi_geojson is None and aoi_text.strip():
    try:
        aoi_geojson = json.loads(aoi_text)
    except Exception:
        st.sidebar.error("Pasted AOI is not valid GeoJSON.")

st.sidebar.header("3) Site analysis inputs")
uploaded_tif = st.sidebar.file_uploader("Upload site GeoTIFF (optional)", type=["tif", "tiff"])
site_path_text = st.sidebar.text_input("…or path to GeoTIFF on disk", str(REPO_ROOT / "data/input/sample_aoi.tif"))

st.sidebar.header("4) Models")
use_unet = st.sidebar.checkbox("Run U-Net (suitability map)", True)
use_resnet = st.sidebar.checkbox("Run ResNet (site quality)", False)

st.sidebar.header("5) Goal prompt")
goal_query = st.sidebar.text_area(
    "Describe your restoration goal",
    "A drought-tolerant, fast-growing forest with high biodiversity for the Uttar Pradesh plains.",
    height=80,
)

run_clicked = st.sidebar.button("🚀 Generate Plan", type="primary")

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _file_from_uploader(upl) -> Optional[Path]:
    if upl is None:
        return None
    out_dir = REPO_ROOT / ".streamlit_cache" / "uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / upl.name
    with open(out_path, "wb") as f:
        f.write(upl.getbuffer())
    return out_path

def _resolve_site_path(text_path: str, upload) -> Optional[Path]:
    if upload is not None:
        return _file_from_uploader(upload)
    p = Path(text_path).expanduser()
    return p if p.exists() else None

def _rgb_preview(path: Path, max_px: int = 900) -> Optional[np.ndarray]:
    if rasterio is None:
        return None
    try:
        with rasterio.open(path) as ds:
            bands = [1, 2, 3] if ds.count >= 3 else [1]
            img = ds.read(bands)  # (B,H,W)
            img = np.moveaxis(img, 0, -1).astype(np.float32)  # (H,W,B)
            if img.size == 0:
                return None
            eps = 1e-6
            mn = img.min(axis=(0, 1), keepdims=True)
            mx = img.max(axis=(0, 1), keepdims=True)
            img = (img - mn) / (mx - mn + eps)
            img = (img * 255).clip(0, 255).astype(np.uint8)
            h, w = img.shape[:2]
            scale = max(h, w) / float(max_px)
            if scale > 1:
                new_h, new_w = int(h / scale), int(w / scale)
                try:
                    import cv2
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                except Exception:
                    from PIL import Image
                    img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
            return img
    except Exception:
        return None

# ------------------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_intelligence(db_url: str, embeddings_dir: Path):
    kb = None
    if DB_OK:
        try:
            kb = KnowledgeBase(db_url)
        except Exception as e:
            st.warning(f"DB unavailable: {e}")
    else:
        st.warning(f"DB module missing: {repr(_DB_IMPORT_ERR)}")

    recommender = None
    if RECOMMENDER_OK:
        try:
            try:
                recommender = IntelligentRecommender(embeddings_dir, kb=kb)
            except TypeError:
                recommender = IntelligentRecommender(embeddings_dir)
        except Exception as e:
            st.error(f"Recommender init failed: {e}")
    else:
        st.error(f"Recommender module missing: {repr(_RECOMMENDER_IMPORT_ERR)}")

    return recommender, kb

recommender, kb = load_intelligence(DB_URL, EMBEDDINGS_DIR)

# ------------------------------------------------------------------------------
# Analysis Engine — Stage 2: ANALYSIS
# ------------------------------------------------------------------------------
@dataclass
class SiteFingerprint:
    rainfall_mm: Optional[float] = None
    pet_mm: Optional[float] = None
    water_balance_mm: Optional[float] = None
    biodiversity_baseline_count: Optional[int] = None
    notes: List[str] = None

class AnalysisEngine:
    def __init__(self, kb: Optional[KnowledgeBase], recommender: Optional[IntelligentRecommender]):
        self.kb = kb
        self.rec = recommender

    # ---------- A) Enhanced Site Fingerprint ----------
    def site_fingerprint(self, aoi_geojson: Dict[str, Any]) -> SiteFingerprint:
        notes: List[str] = []
        rainfall = pet = water_balance = None
        bio_count = None

        if _EE_OK and ee is not None:
            try:
                geom = ee.Geometry(aoi_geojson)
                # CHIRPS last 12 months precipitation (mm)
                chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(ee.Date.now().advance(-12, "month"), ee.Date.now())
                prcp = chirps.sum().rename("prcp_12mo")

                # MOD16A2 ET (mm/8day); approx PET proxy over 12 months
                mod16 = ee.ImageCollection("MODIS/061/MOD16A2").filterDate(ee.Date.now().advance(-12, "month"), ee.Date.now())
                et = mod16.select("ET").sum().rename("pet_12mo")

                # Reduce to mean over AOI
                pr_val = prcp.reduceRegion(ee.Reducer.mean(), geom, 1000).get("prcp_12mo")
                et_val = et.reduceRegion(ee.Reducer.mean(), geom, 1000).get("pet_12mo")
                rainfall = float(pr_val.getInfo()) if pr_val else None
                pet = float(et_val.getInfo()) if et_val else None
                if rainfall is not None and pet is not None:
                    water_balance = rainfall - pet
            except Exception as e:
                notes.append(f"EE extraction failed: {e}")
        else:
            notes.append("Earth Engine not initialized; water balance unavailable.")

        # Biodiversity baseline (very rough): count species records for the enclosing state
        if self.kb is not None:
            try:
                # This assumes you have a `species` table with a `state` column and
                # a helper to map AOI → state. Replace with your own spatial join
                # if available. Here we fall back to None if not present.
                state_name = self._infer_state_name_from_aoi(aoi_geojson)
                if state_name:
                    sql = "SELECT COUNT(*) AS n FROM species WHERE state = :state"
                    df = self.kb.query(sql, {"state": state_name})
                    if isinstance(df, pd.DataFrame) and "n" in df.columns:
                        bio_count = int(df["n"].iloc[0])
                else:
                    notes.append("Could not infer state from AOI; baseline biodiversity skipped.")
            except Exception as e:
                notes.append(f"DB biodiversity baseline failed: {e}")
        else:
            notes.append("DB not available; biodiversity baseline skipped.")

        return SiteFingerprint(
            rainfall_mm=rainfall, pet_mm=pet, water_balance_mm=water_balance,
            biodiversity_baseline_count=bio_count, notes=notes
        )

    def _infer_state_name_from_aoi(self, aoi_geojson: Dict[str, Any]) -> Optional[str]:
        # Placeholder: if your DB exposes a spatial layer, do a real spatial join.
        # Here we return None to keep the engine generic.
        return None

    # ---------- B) Guild Recommender ----------
    def recommend_guild(self, goal: str, priorities: Dict[str, str], k: int = 8) -> List[str]:
        if self.rec is None:
            return []
        try:
            # Strategy: pick a seed species then expand by semantic similarity.
            # If your IntelligentRecommender exposes dedicated methods, swap these calls.
            seed_list = self.rec.recommend_species(f"{goal}. Priorities: {priorities}", k=1)
            seed = seed_list[0] if seed_list else None
            if seed is None:
                return self.rec.recommend_species(goal, k=k)
            # Expand around the seed
            guild = [seed] + [s for s in self.rec.recommend_species(f"similar to {seed}", k=k*2) if s != seed]
            # Deduplicate and trim to k
            seen = set()
            out = []
            for s in guild:
                if s not in seen:
                    out.append(s); seen.add(s)
                if len(out) >= k:
                    break
            return out
        except Exception:
            # Fallback: single-pass recommendation
            try:
                return self.rec.recommend_species(goal, k=k)
            except Exception:
                return []

    # ---------- C) Scoring (DB-backed when available) ----------
    def score_guild(self, species_list: List[str], priorities: Dict[str, str]) -> pd.DataFrame:
        # Columns: species, iucn, carbon_rate, canopy_layer, successional_role, scores...
        rows: List[Dict[str, Any]] = []
        for sp in species_list:
            rows.append({
                "species": sp,
                "iucn": None,
                "carbon_rate": None,
                "canopy_layer": None,
                "successional_role": None,
            })
        df = pd.DataFrame(rows)

        if self.kb is not None and not df.empty:
            try:
                # Example trait fetch; adjust to your schema
                placeholders = ", ".join([":s" + str(i) for i in range(len(df))])
                params = {("s" + str(i)): sp for i, sp in enumerate(df["species"].tolist())}
                sql = f"""
                    SELECT species, iucn_status AS iucn, carbon_sequestration_rate AS carbon_rate,
                           canopy_layer, successional_role
                    FROM species_traits
                    WHERE species IN ({placeholders})
                """
                trait_df = self.kb.query(sql, params)
                if isinstance(trait_df, pd.DataFrame) and not trait_df.empty:
                    df = df.drop(columns=[c for c in ["iucn","carbon_rate","canopy_layer","successional_role"] if c in df.columns]) \
                           .merge(trait_df, on="species", how="left")
            except Exception:
                pass

        # Compute simple scores
        def iucn_points(x: str) -> int:
            if x is None: return 0
            x = x.upper()
            return {"CR": 5, "EN": 4, "VU": 3, "NT": 2}.get(x, 1)  # LC=1, unknown=0→1

        df["biodiversity_score"] = df["iucn"].map(iucn_points).fillna(0)

        # Carbon score: normalize by max among guild
        if "carbon_rate" in df.columns and df["carbon_rate"].notna().any():
            mx = float(df["carbon_rate"].max(skipna=True))
            df["carbon_score"] = (df["carbon_rate"].astype(float) / (mx if mx > 0 else 1.0)).clip(0, 1)
        else:
            df["carbon_score"] = 0.0

        # Canopy structure bonus if guild spans all layers
        layers = set([str(x) for x in df["canopy_layer"].dropna().unique().tolist()])
        layer_bonus = 1.0 if {"Upper Canopy", "Mid-story", "Shrub Layer"}.issubset(layers) else 0.5 if layers else 0.0
        df["canopy_diversity_bonus"] = layer_bonus

        # Weighted overall score by priorities
        p1 = priorities.get("primary", "")
        p2 = priorities.get("secondary", "")
        w_bio = 0.5 if "biodiversity" in (p1 + p2) else 0.25
        w_car = 0.5 if "carbon" in (p1 + p2) else 0.25
        w_other = 1.0 - (w_bio + w_car)
        df["overall_score"] = (w_bio * df["biodiversity_score"].astype(float) / 5.0) + \
                              (w_car * df["carbon_score"].astype(float)) + \
                              (w_other * df["canopy_diversity_bonus"])

        return df.sort_values("overall_score", ascending=False).reset_index(drop=True)

    # ---------- Models ----------
    def run_models(self, site_tif: Optional[Path], use_unet: bool, use_resnet: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if site_tif is None:
            return out

        if use_unet and MODEL_RUNNER_OK and UNET_MODEL_PATH.exists():
            try:
                out["unet_pred"] = run_unet_inference(
                    model_path=UNET_MODEL_PATH, geotiff_path=site_tif, num_classes=UNET_NUM_CLASSES
                )
            except Exception as e:
                out["unet_error"] = str(e)
        elif use_unet:
            out["unet_error"] = "U-Net model or runner unavailable."

        if use_resnet and MODEL_RUNNER_OK and RESNET_MODEL_PATH.exists():
            try:
                out["resnet_pred"] = run_resnet_inference(
                    model_path=RESNET_MODEL_PATH, geotiff_path=site_tif, num_classes=RESNET_NUM_CLASSES
                )
            except Exception as e:
                out["resnet_error"] = str(e)
        elif use_resnet:
            out["resnet_error"] = "ResNet model or runner unavailable."

        return out

    # ---------- Phase Timeline ----------
    def make_timeline(self, scored_df: pd.DataFrame) -> Dict[str, List[str]]:
        # Group species by successional role into phases
        phases = {
            "Year 1–3 (Establishment)": [],
            "Year 4–7 (Growth)": [],
            "Year 8+ (Maturity)": [],
        }
        if scored_df is None or scored_df.empty:
            return phases

        for _, row in scored_df.iterrows():
            sp = str(row["species"])
            role = (str(row.get("successional_role", "")) or "").lower()
            if "pioneer" in role or "early" in role:
                phases["Year 1–3 (Establishment)"].append(sp)
            elif "mid" in role or "intermediate" in role:
                phases["Year 4–7 (Growth)"].append(sp)
            elif "climax" in role or "late" in role:
                phases["Year 8+ (Maturity)"].append(sp)
            else:
                # distribute by canopy for variety if no role is present
                canopy = (str(row.get("canopy_layer", "")) or "").lower()
                if "shrub" in canopy:
                    phases["Year 1–3 (Establishment)"].append(sp)
                elif "mid" in canopy:
                    phases["Year 4–7 (Growth)"].append(sp)
                else:
                    phases["Year 8+ (Maturity)"].append(sp)
        return phases

engine = AnalysisEngine(kb=kb, recommender=recommender)

# ------------------------------------------------------------------------------
# Output — Stage 3: PLAN (only when user clicks)
# ------------------------------------------------------------------------------
def _render_water_overlay_on_map(aoi_geojson: Dict[str, Any]):
    # Show AOI + optional JRC water layer using folium (fallback if no leafmap)
    if not _FOLIUM_OK:
        st.info("Map overlay requires folium + streamlit-folium.")
        return

    fmap = folium.Map(location=[22.5, 79.0], zoom_start=5, control_scale=True, tiles="CartoDB positron")

    # AOI polygon
    try:
        folium.GeoJson(aoi_geojson, name="AOI", style_function=lambda x: {"color": "#0d6efd", "weight": 2, "fillOpacity": 0.05}).add_to(fmap)
    except Exception:
        pass

    if _EE_OK and ee is not None:
        try:
            geom = ee.Geometry(aoi_geojson)
            jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
            vis = {"min": 0, "max": 100, "palette": ["#ffffff", "#2ca2f0"]}
            map_id = ee.data.getMapId({"image": jrc.visualize(**vis).getMapId()["image"]})  # compat trick
        except Exception:
            map_id = None

        # Simpler approach via leaflet tile if map_id available
        try:
            tile = jrc.visualize(**vis).getMapId()
            folium.raster_layers.TileLayer(
                tiles=tile["tile_fetcher"].url_format, attr="JRC GSW", name="Water Occurrence", overlay=True, control=True, opacity=0.6
            ).add_to(fmap)
        except Exception:
            pass

    folium.LayerControl().add_to(fmap)
    st_folium(fmap, height=420, use_container_width=True)

def _to_frame(items: Union[List[str], List[Tuple[Any, ...]], List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(items, pd.DataFrame):
        return items
    if isinstance(items, list):
        if not items:
            return pd.DataFrame(columns=["Result"])
        if isinstance(items[0], dict):
            return pd.DataFrame(items)
        if isinstance(items[0], (list, tuple)):
            m = max(len(x) for x in items)
            return pd.DataFrame([list(x) for x in items], columns=[f"col_{i}" for i in range(m)])
        return pd.DataFrame({"Recommended Species": items})
    return pd.DataFrame({"Result": [items]})

if run_clicked:
    # Resolve AOI + site path
    if aoi_geojson is None:
        st.error("Please provide an AOI (draw on map, upload, or paste GeoJSON).")
        st.stop()

    site_path = _resolve_site_path(site_path_text, uploaded_tif)
    if use_unet or use_resnet:
        if site_path is None:
            st.warning("No site GeoTIFF provided; model outputs will be skipped.")

    # ------------ Analysis panel ------------
    st.header("📊 Comprehensive Restoration Plan")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1) Site Analysis")
        with st.spinner("Computing site fingerprint…"):
            fp = engine.site_fingerprint(aoi_geojson)
        kpis = []
        if fp.water_balance_mm is not None:
            kpis.append(("Water balance (12 mo)", f"{fp.water_balance_mm:,.0f} mm"))
        if fp.rainfall_mm is not None:
            kpis.append(("Rainfall (12 mo)", f"{fp.rainfall_mm:,.0f} mm"))
        if fp.pet_mm is not None:
            kpis.append(("PET proxy (12 mo)", f"{fp.pet_mm:,.0f} mm"))
        if fp.biodiversity_baseline_count is not None:
            kpis.append(("Baseline species records", f"{fp.biodiversity_baseline_count:,}"))
        if kpis:
            a, b = st.columns(2)
            for i, (k, v) in enumerate(kpis):
                (a if i % 2 == 0 else b).metric(k, v)
        if fp.notes:
            st.caption("Notes:")
            for n in fp.notes:
                st.caption(f"• {n}")

        st.markdown("**Interactive Water Context**")
        _render_water_overlay_on_map(aoi_geojson)

    with col2:
        st.subheader("2) AI Model Insights")
        # Preview
        if site_path is not None:
            prev = _rgb_preview(site_path)
            if prev is not None:
                st.image(prev, caption="Site RGB preview", use_container_width=True)
        with st.spinner("Running selected models…"):
            model_out = engine.run_models(site_path, use_unet=use_unet, use_resnet=use_resnet)
        if "unet_error" in model_out:
            st.warning(f"U-Net: {model_out['unet_error']}")
        if "resnet_error" in model_out:
            st.warning(f"ResNet: {model_out['resnet_error']}")
        if "unet_pred" in model_out:
            up = model_out["unet_pred"]
            # If your runner returns a mask ndarray, try showing; otherwise show path/summary
            if isinstance(up, np.ndarray):
                # quick palette
                cmap = np.array([[220, 53, 69], [255, 193, 7], [40, 167, 69]], dtype=np.uint8)
                idx = np.clip(up.astype(int), 0, cmap.shape[0]-1)
                rgba = np.concatenate([cmap[idx], np.full((*idx.shape,1), 200, dtype=np.uint8)], axis=-1)
                st.image(rgba, caption="Suitability (U-Net)", use_column_width=True)
            else:
                st.write("U-Net output:", up)
        if "resnet_pred" in model_out:
            st.write("ResNet output:", model_out["resnet_pred"])

    st.divider()
    st.subheader("3) Intelligent Species Recommendation")
    with st.spinner("Forming a synergistic species guild…"):
        guild = engine.recommend_guild(goal_query, priorities, k=10)
        scored = engine.score_guild(guild, priorities) if guild else pd.DataFrame()
    if scored is not None and not scored.empty:
        st.dataframe(scored, use_container_width=True)
        st.download_button(
            "⬇️ Download scored guild (CSV)",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="manthan_scored_guild.csv",
            mime="text/csv",
        )
    else:
        st.info("No recommendations available (recommender unavailable or returned no results).")

    st.divider()
    st.subheader("4) Phased Planting Timeline")
    phases = engine.make_timeline(scored if scored is not None else pd.DataFrame())
    with st.expander("Year 1–3 (Establishment Phase)", expanded=True):
        est = phases.get("Year 1–3 (Establishment)", [])
        if est:
            st.markdown("- Soil builders & early yield (NTFPs):")
            st.write(", ".join(est))
        else:
            st.write("No assignments yet.")
    with st.expander("Year 4–7 (Growth Phase)", expanded=True):
        gr = phases.get("Year 4–7 (Growth)", [])
        if gr:
            st.markdown("- Introduce mid-story for layered canopy:")
            st.write(", ".join(gr))
        else:
            st.write("No assignments yet.")
    with st.expander("Year 8+ (Maturity Phase)", expanded=True):
        mat = phases.get("Year 8+ (Maturity)", [])
        if mat:
            st.markdown("- Long-term stability & carbon capture:")
            st.write(", ".join(mat))
        else:
            st.write("No assignments yet.")

    st.divider()
    st.subheader("5) Executive Summary")
    bullets = [
        f"Profile: **{profile}** — priorities: **{priorities.get('primary','-')}**, **{priorities.get('secondary','-')}**.",
        "Target high-suitability micro-zones first; phase pioneers → mid-story → climax.",
        "Blend fast growers with drought-tolerant natives; diversify canopy layers.",
        "Water plan: harvest & store in upper catchments; supplement in dry months.",
        "Review survival quarterly; iterate guild with field feedback.",
    ]
    st.markdown("\n".join([f"- {b}" for b in bullets]))

# ------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------
with st.expander("⚙️ Diagnostics"):
    st.write("Repo root:", REPO_ROOT)
    st.write("DB URL:", DB_URL)
    st.write("Embeddings dir exists:", EMBEDDINGS_DIR.exists())
    st.write("U-Net model:", str(UNET_MODEL_PATH), "| exists:", UNET_MODEL_PATH.exists())
    st.write("ResNet model:", str(RESNET_MODEL_PATH), "| exists:", RESNET_MODEL_PATH.exists())
    st.write("Recommender import OK:", RECOMMENDER_OK)
    if not RECOMMENDER_OK: st.code(repr(_RECOMMENDER_IMPORT_ERR))
    st.write("Model runner import OK:", MODEL_RUNNER_OK)
    if not MODEL_RUNNER_OK: st.code(repr(_MODEL_RUNNER_IMPORT_ERR))
    st.write("DB import OK:", DB_OK)
    if not DB_OK: st.code(repr(_DB_IMPORT_ERR))
    st.write("Earth Engine initialized:", _EE_OK)
    if not _EE_OK and _EE_IMPORTED:
        st.info("Tip: `earthengine authenticate` (locally) or set GOOGLE_APPLICATION_CREDENTIALS for service account.")
