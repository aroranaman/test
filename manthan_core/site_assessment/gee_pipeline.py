# manthan_core/site_assessment/gee_pipeline.py
from __future__ import annotations
import os
from typing import Dict, Any, Optional

import ee
from dotenv import load_dotenv

# --- Configuration & Setup ---
load_dotenv()
EE_PROJECT_ID = os.getenv("EE_PROJECT_ID", "manthan-466509")

# --- GEE Initialization ---
try:
    ee.Initialize(project=EE_PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT_ID)

# --- Default AOI Fallback ---
# A sample 1-acre plot in a degraded area of Uttar Pradesh for testing
DEFAULT_AOI = ee.Geometry.Rectangle([78.10, 27.50, 78.11, 27.51])

# ---
# Analysis Functions
# ---

def get_biodiversity_baseline(aoi: ee.Geometry) -> Dict[str, Any]:
    """Calculates the current biodiversity state using vegetation density (NDVI) as a proxy."""
    try:
        ndvi_collection = ee.ImageCollection('MODIS/006/MOD13A1').filter(ee.Filter.date('2023-01-01', '2023-12-31'))
        ndvi_image = ndvi_collection.select('NDVI').median().multiply(0.0001)
        mean_ndvi = ndvi_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=500, bestEffort=True).get('NDVI')
        
        # Ensure the GEE object is evaluated to a number
        ndvi_value = ee.Number(mean_ndvi).getInfo()
        
        return {
            "mean_ndvi": ndvi_value,
            "biodiversity_score_proxy": round(ndvi_value * 100, 2) if ndvi_value is not None else 0
        }
    except Exception as e:
        return {"mean_ndvi": "Error", "biodiversity_score_proxy": f"Error: {e}"}


def get_water_balance(aoi: ee.Geometry) -> Dict[str, Any]:
    """Calculates a simplified water balance (Precipitation - Evapotranspiration)."""
    try:
        precipitation = ee.Image("WORLDCLIM/V1/BIO").select('bio12')
        evapotranspiration = ee.ImageCollection('MODIS/061/MOD16A2').filter(ee.Filter.date('2022-01-01', '2022-12-31')).select('PET').sum().multiply(0.1)

        mean_precip = ee.Number(precipitation.reduceRegion(ee.Reducer.mean(), aoi, 1000, bestEffort=True).get('bio12')).getInfo() or 0
        mean_et = ee.Number(evapotranspiration.reduceRegion(ee.Reducer.mean(), aoi, 500, bestEffort=True).get('PET')).getInfo() or 0
        
        water_balance = mean_precip - mean_et
        
        return {
            "annual_precipitation_mm": mean_precip,
            "annual_pet_mm": mean_et,
            "water_balance_mm": round(water_balance, 2)
        }
    except Exception as e:
        return {"annual_precipitation_mm": "Error", "annual_pet_mm": "Error", "water_balance_mm": f"Error: {e}"}

def get_carbon_baseline(aoi: ee.Geometry) -> Dict[str, Any]:
    """Estimates the current soil organic carbon stock as a baseline."""
    try:
        soil_carbon = ee.Image("projects/soilgrids-isric/soc_mean").select('soc_0-5cm_mean')
        mean_soc = ee.Number(soil_carbon.reduceRegion(ee.Reducer.mean(), aoi, 250, bestEffort=True).get('soc_0-5cm_mean')).getInfo() or 0
        
        # Convert from decigrams/kg to tonnes/hectare (approximate)
        tonnes_per_ha = (mean_soc / 10) * 70 

        return {"soil_organic_carbon_tonnes_ha": round(tonnes_per_ha, 2)}
    except Exception as e:
        return {"soil_organic_carbon_tonnes_ha": f"Error: {e}"}

# ---
# The Master Function
# ---

def get_site_fingerprint(aoi: Optional[ee.Geometry]) -> Dict[str, Any]:
    """
    The main analysis engine function. Takes a user's AOI and generates a
    complete, multi-layered environmental fingerprint.
    """
    if not aoi:
        aoi = DEFAULT_AOI

    # Run all analysis modules in parallel on GEE's servers
    biodiversity_data = get_biodiversity_baseline(aoi)
    water_data = get_water_balance(aoi)
    carbon_data = get_carbon_baseline(aoi)
    
    # Combine results into a single fingerprint dictionary
    fingerprint = {
        "status": "Success",
        **biodiversity_data,
        **water_data,
        **carbon_data
    }
    
    return fingerprint