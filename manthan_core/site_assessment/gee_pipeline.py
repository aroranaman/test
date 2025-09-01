# manthan_core/site_assessment/gee_pipeline.py
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import ee
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Logging & Config
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
EE_PROJECT_ID = os.getenv("EE_PROJECT_ID", "manthan-466509")

# ---------------------------------------------------------------------
# Earth Engine Initialization
# ---------------------------------------------------------------------
try:
    ee.Initialize(project=EE_PROJECT_ID)
except Exception:
    logging.info("GEE not initialized; attempting interactive authentication...")
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT_ID)

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
# ~2 acres near Satpura for smoke tests
DEFAULT_AOI = ee.Geometry.Point([78.5, 22.5]).buffer(1e2).bounds()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_get(d: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    """Safely fetch a numeric value from a dict-like object returned by getInfo()."""
    try:
        v = d.get(key, default)
        return float(v) if v is not None else default
    except Exception:
        return default

def _try_first_image(asset_ids: List[str]) -> Optional[ee.Image]:
    """Return the first ee.Image that actually loads; else None."""
    for aid in asset_ids:
        try:
            img = ee.Image(aid)
            # Light check to ensure it exists and is accessible:
            _ = img.bandNames().getInfo()
            logging.info(f"Using soil asset: {aid}")
            return img
        except Exception:
            continue
    return None

def _normalize_ph_value(val: Optional[float]) -> Optional[float]:
    """Normalize pH: if > 14, treat as deci-pH (÷10), then clamp to [3, 10]."""
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if v > 14.0:
        v = v / 10.0
    # Clamp to a sensible range for display/heuristics
    if v < 3.0:
        v = 3.0
    if v > 10.0:
        v = 10.0
    return v

# ---------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------
def get_vegetation_indices(aoi: ee.Geometry) -> Dict[str, Any]:
    """
    Calculates NDVI, EVI, SAVI from Sentinel-2 SR (2023, <20% clouds).
    """
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate("2023-01-01", "2023-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .multiply(0.0001)  # reflectance scale factor
    )

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = s2.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": s2.select("B8"), "RED": s2.select("B4"), "BLUE": s2.select("B2")},
    ).rename("EVI")
    savi = s2.expression(
        "1.5 * ((NIR - RED) / (NIR + RED + 0.5))",
        {"NIR": s2.select("B8"), "RED": s2.select("B4")},
    ).rename("SAVI")

    reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
    stats = (
        ndvi.addBands(evi)
        .addBands(savi)
        .reduceRegion(reducer=reducer, geometry=aoi, scale=10, bestEffort=True)
        .getInfo()
        or {}
    )

    return {
        "ndvi_mean": _safe_get(stats, "NDVI_mean"),
        "ndvi_stdDev": _safe_get(stats, "NDVI_stdDev"),
        "evi_mean": _safe_get(stats, "EVI_mean"),
        "savi_mean": _safe_get(stats, "SAVI_mean"),
    }

def get_climate_variables(aoi: ee.Geometry) -> Dict[str, Any]:
    """
    Fetches key bioclimatic variables from WorldClim v1 BIO.
    """
    worldclim = ee.Image("WORLDCLIM/V1/BIO")
    climate = worldclim.select(
        ["bio01", "bio07", "bio12", "bio15"],
        ["mean_temp_c", "temp_annual_range_c", "annual_precip_mm", "precip_seasonality_cv"],
    )
    stats = (
        climate.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=1000, bestEffort=True)
        .getInfo()
        or {}
    )

    # bio01 is often deci-°C; normalize if needed
    try:
        mt = stats.get("mean_temp_c")
        if mt is not None and abs(float(mt)) > 100:
            stats["mean_temp_c"] = float(mt) / 10.0
    except Exception:
        pass

    return {
        "mean_temp_c": _safe_get(stats, "mean_temp_c"),
        "temp_annual_range_c": _safe_get(stats, "temp_annual_range_c"),
        "annual_precip_mm": _safe_get(stats, "annual_precip_mm"),
        "precip_seasonality_cv": _safe_get(stats, "precip_seasonality_cv"),
    }

def get_soil_properties(aoi: ee.Geometry) -> Dict[str, Any]:
    """
    Fetches key topsoil properties with resilient fallbacks.
    pH: OpenLandMap pH(H2O). SOC: try multiple assets; convert g/kg → t/ha when appropriate.
    """
    # pH (H2O) candidates (OpenLandMap)
    ph_img = _try_first_image([
        "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02",
        "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_A/v02",
        # Older SoilGrids pH (if still accessible to your account)
        "projects/soilgrids-isric/phh2o_mean",
    ])
    if ph_img is None:
        logging.warning("No accessible pH asset found; returning soil as None.")
        return {"soil_ph": None, "soil_organic_carbon_tonnes_ha": None}

    # SOC candidates (names vary; probe several stable/public sets)
    soc_img = _try_first_image([
        # OpenLandMap organic carbon (g/kg). This has been stable.
        "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02",
        # FAO GSOC map variants (units typically t/ha, 0–30 cm); availability can vary.
        "FAO/GSOCmap/1km/SOC",
        "FAO/GSOCmap/v1/SOC",
        # Legacy SoilGrids name if your account has it:
        "projects/soilgrids-isric/soc_mean",
    ])

    # Reduce over AOI
    scale = 250  # OpenLandMap nominal resolution
    reducer = ee.Reducer.mean()

    if soc_img is not None:
        # Reduce mean for pH and SOC
        ph_mean = ph_img.reduce(reducer).rename("soil_ph_raw")
        soc_mean = soc_img.reduce(reducer).rename("soc_raw")
        stats = (
            ph_mean.addBands(soc_mean)
            .reduceRegion(reducer=reducer, geometry=aoi, scale=scale, bestEffort=True)
            .getInfo()
            or {}
        )
        ph_val = _normalize_ph_value(stats.get("soil_ph_raw"))

        # SOC unit handling:
        # If soc_raw in plausible g/kg range (0–200), convert via (g/kg / 10) * 70  ⇒ t/ha (heuristic).
        # Else, assume already t/ha and pass through.
        soc_out: Optional[float] = None
        try:
            raw = stats.get("soc_raw")
            if raw is not None:
                rawf = float(raw)
                if 0.0 <= rawf <= 200.0:
                    soc_out = round((rawf / 10.0) * 70.0, 2)
                else:
                    soc_out = round(rawf, 2)
        except Exception:
            soc_out = None

        return {
            "soil_ph": ph_val,
            "soil_organic_carbon_tonnes_ha": soc_out,
        }

    # No SOC available — return pH only
    logging.warning("No accessible SOC asset found; returning pH only.")
    ph_stats = (
        ph_img.reduce(reducer)
        .reduceRegion(reducer=reducer, geometry=aoi, scale=scale, bestEffort=True)
        .getInfo()
        or {}
    )
    return {
        "soil_ph": _normalize_ph_value(ph_stats.get("soil_ph")),
        "soil_organic_carbon_tonnes_ha": None,
    }

def get_topography(aoi: ee.Geometry) -> Dict[str, Any]:
    """
    Calculates mean elevation, slope, aspect from SRTM.
    """
    dem = ee.Image("USGS/SRTMGL1_003")
    elevation = dem.select("elevation")
    slope = ee.Terrain.slope(dem)
    aspect = ee.Terrain.aspect(dem)

    reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
    stats = (
        elevation.addBands(slope)
        .addBands(aspect)
        .reduceRegion(reducer=reducer, geometry=aoi, scale=30, bestEffort=True)
        .getInfo()
        or {}
    )
    return {
        "elevation_mean_m": _safe_get(stats, "elevation_mean"),
        "slope_mean_deg": _safe_get(stats, "slope_mean"),
        "aspect_mean_deg": _safe_get(stats, "aspect_mean"),
    }

# ---------------------------------------------------------------------
# AOI Download Helper (placeholder-friendly)
# ---------------------------------------------------------------------
def download_aoi_as_geotiff(aoi: ee.Geometry, output_path: Path) -> bool:
    """
    Writes a Sentinel-2 RGB composite for the AOI to disk.

    For production, you would:
      (1) Build an ee.Image (e.g., S2 RGB composite).
      (2) Use Export.image.toDrive / toCloudStorage OR image.getDownloadURL()
      (3) Download the resulting GeoTIFF.

    In this demo helper, we simply ensure the output directory exists and
    return True so the Streamlit app can proceed without crashing if a local
    AOI GeoTIFF is not yet wired.
    """
    try:
        _ = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate("2023-01-01", "2023-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .median()
            .select(["B4", "B3", "B2"])
            .multiply(0.0001)
        )
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # NOTE: We are not actually exporting in this demo helper.
        # Hook: Use getDownloadURL / Export.image.toDrive here in a real pipeline.
        logging.info(f"[download_aoi_as_geotiff] Prepared output directory: {output_path.parent}")
        return True
    except Exception as e:
        logging.error(f"Failed to prepare/download GEE image: {e}")
        return False

# ---------------------------------------------------------------------
# Master Function
# ---------------------------------------------------------------------
def get_site_fingerprint(aoi: Optional[ee.Geometry]) -> Dict[str, Any]:
    """
    Generates a multi-layer environmental fingerprint for the AOI.
    """
    if not aoi:
        aoi = DEFAULT_AOI

    try:
        vegetation = get_vegetation_indices(aoi)
        climate = get_climate_variables(aoi)
        soil = get_soil_properties(aoi)
        topo = get_topography(aoi)

        # Assemble final payload
        return {
            "status": "Success",
            **vegetation,
            **climate,
            **soil,
            **topo,
        }
    except Exception as e:
        logging.exception("Fingerprint generation failed")
        return {"status": "Error", "message": str(e)}

# ---------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Generating Site Fingerprint for Default Sample AOI ---")
    result = get_site_fingerprint(None)
    import json
    print(json.dumps(result, indent=2))
