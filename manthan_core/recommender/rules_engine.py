# manthan_core/recommender/rules_engine.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# -----------------------------------------------------------------------------
# Import bootstrap so this file works both as a module and as a script
# -----------------------------------------------------------------------------
# If run directly (python manthan_core/recommender/rules_engine.py),
# ensure the repo root is on sys.path so "manthan_core" is importable.
if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Project/Manthan
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

# Prefer relative import when executed as a proper module; fall back to absolute.
try:
    from ..utils.db_connector import KnowledgeBase
except Exception:
    from manthan_core.utils.db_connector import KnowledgeBase  # type: ignore

# -----------------------------------------------------------------------------
# Configuration / KB connector
# -----------------------------------------------------------------------------
DB_URL = os.getenv("MANTHAN_DB_URL", "sqlite:///data/manthan.db")
kb = KnowledgeBase(db_url=DB_URL)

# -----------------------------------------------------------------------------
# Catalog-based recommendation (qualitative trait scoring)
# -----------------------------------------------------------------------------
def recommend_species_for_site(
    site_fingerprint: Dict[str, Any],
    species_catalog: pd.DataFrame,
    k: int = 20,
) -> pd.DataFrame:
    """
    Scores and ranks species from a catalog based on a site's environmental fingerprint.
    Works with qualitative traits (e.g., soil_type_preference, water_need).

    Expected columns in species_catalog (string values are fine):
      - 'species' (or 'scientific_name')
      - 'soil_type_preference'  e.g., 'acidic', 'neutral', 'alkaline'
      - 'water_need'            e.g., 'Low', 'Medium', 'High'

    Returns a DataFrame with at least:
      ['species', 'soil_type_preference', 'water_need', 'score']
    """

    if species_catalog is None or species_catalog.empty:
        return pd.DataFrame(columns=["species", "soil_type_preference", "water_need", "score"])

    catalog = species_catalog.copy()

    # Normalize name column for display
    if "species" not in catalog.columns:
        catalog["species"] = catalog.get("scientific_name", pd.Series(["Unknown"] * len(catalog)))

    site_ph = site_fingerprint.get("soil_ph", None)
    site_rain = site_fingerprint.get("annual_precip_mm", None)

    def _lower_str(x: Any) -> str:
        try:
            return str(x).strip().lower()
        except Exception:
            return ""

    def calculate_score(row: pd.Series) -> float:
        score = 0.0

        # ---- Soil pH match via qualitative preference (max 0.4) ----
        pref = _lower_str(row.get("soil_type_preference", ""))
        if site_ph is not None:
            try:
                ph = float(site_ph)
            except Exception:
                ph = None
            if ph is not None:
                if "acid" in pref and ph < 7.0:
                    score += 0.4
                elif "neutral" in pref and 6.5 <= ph <= 7.5:
                    score += 0.4
                elif "alkal" in pref and ph > 7.0:
                    score += 0.4

        # ---- Water/rain match via water_need proxy (max 0.3) ----
        need = _lower_str(row.get("water_need", "medium"))
        if site_rain is not None:
            try:
                rain = float(site_rain)
            except Exception:
                rain = None
            if rain is not None:
                if need == "low" and rain < 800:
                    score += 0.3
                elif need == "medium" and 800 <= rain <= 1500:
                    score += 0.3
                elif need == "high" and rain > 1500:
                    score += 0.3

        # (Future) Add drought_tolerance, temp range, canopy matching, etc.
        return round(score, 3)

    catalog["score"] = catalog.apply(calculate_score, axis=1)
    topk = catalog.sort_values("score", ascending=False).head(k).reset_index(drop=True)
    keep_cols = [c for c in ["species", "soil_type_preference", "water_need", "score"] if c in topk.columns]
    return topk[keep_cols]

# -----------------------------------------------------------------------------
# KB-driven helpers (compatibility, water, agroforestry)
# -----------------------------------------------------------------------------
def _normalize_truthy(val: Any) -> bool:
    """Accept a variety of truthy encodings from the traits table."""
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y", "t", "high", "symbiotic"}

def calculate_water_requirement_score(species_name: str, site_water_availability: str) -> float:
    """
    Scores a species based on its water needs vs site water availability, using the KB.

    Looks up 'water_need'. If absent, infers from 'drought_tolerance':
       High drought_tolerance -> Low water_need
       Medium -> Medium
       Low -> High

    Returns:
      1.0 if exact match,
      0.5 if off by one bucket,
      0.1 if far off.
    """
    traits = kb.get_species_traits([species_name]).get(species_name, {}) or {}

    water_need = traits.get("water_need")
    if not water_need:
        dtol = (traits.get("drought_tolerance") or "").strip().lower()
        if dtol in {"very high", "high"}:
            water_need = "Low"
        elif dtol in {"medium", "moderate"}:
            water_need = "Medium"
        elif dtol in {"low", "very low"}:
            water_need = "High"
        else:
            water_need = "Medium"

    water_map = {"low": 1, "medium": 2, "high": 3}
    species_val = water_map.get(str(water_need).lower(), 2)
    site_val = water_map.get(str(site_water_availability).lower(), 2)

    diff = abs(species_val - site_val)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    else:
        return 0.1

def calculate_species_compatibility(species_list: List[str]) -> Dict[str, Any]:
    """
    Assesses compatibility based on traits fetched from the knowledge base.
    Currently checks for nitrogen-fixing species (multiple trait name variants).
    """
    if not species_list:
        return {"compatibility_score": 0.5, "comments": ["No species provided."]}

    all_traits = kb.get_species_traits(species_list)
    comments: List[str] = []

    fixer_names = {"nitrogen_fixing", "n_fixation", "fixer"}
    fixers: List[str] = []
    for name, traits in all_traits.items():
        if any(_normalize_truthy(traits.get(k)) for k in fixer_names if k in traits):
            fixers.append(name)

    if fixers:
        comments.append(f"Positive: includes nitrogen-fixing species: {', '.join(sorted(set(fixers)))}.")
        score = 0.8
    else:
        comments.append("Neutral: no nitrogen-fixing species identified.")
        score = 0.5

    # Future: canopy compatibility, allelopathy, pest/disease complementarity, etc.
    return {"compatibility_score": score, "comments": comments}

def generate_agroforestry_plan(species_list: List[str], farmer_income_needs: float) -> Dict[str, Any]:
    """
    Suggests intercropping with NTFPs/Fruit to ensure short-term income,
    using live economic data from the knowledge base.

    Optional table:
      economic_data(species_key, product_type, yield_per_ha, price_per_unit)

    Returns a single "best" intercrop suggestion based on yield * price.
    """
    econ_data = kb.get_economic_data(species_list)

    intercrops = {
        name: data
        for name, data in econ_data.items()
        if str(data.get("type") or "").lower() in {"ntfp", "fruit"}
           and data.get("yield_per_ha") is not None
           and data.get("price") is not None
    }

    if not intercrops:
        return {
            "suggested_intercrops": [],
            "projected_short_term_income_per_ha": 0.0,
            "income_continuity_status": "Low",
        }

    def _rev(entry: Dict[str, Any]) -> float:
        try:
            return float(entry["yield_per_ha"]) * float(entry["price"])
        except Exception:
            return 0.0

    best_name = max(intercrops, key=lambda k: _rev(intercrops[k]))
    best = intercrops[best_name]
    projected_income = _rev(best)

    status = (
        "High" if projected_income >= 0.6 * float(farmer_income_needs)
        else "Medium" if projected_income >= 0.3 * float(farmer_income_needs)
        else "Low"
    )

    return {
        "suggested_intercrops": [best_name],
        "projected_short_term_income_per_ha": projected_income,
        "income_continuity_status": status,
    }

# -----------------------------------------------------------------------------
# Optional helper: build a minimal qualitative catalog from KB
# -----------------------------------------------------------------------------
def build_qualitative_catalog_from_kb(species_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Construct a small catalog DataFrame (species, soil_type_preference, water_need)
    from the live KnowledgeBase. If species_names is provided, narrows to those;
    otherwise returns traits for the provided names only (empty if none provided).
    """
    names = species_names or []
    if not names:
        # Without a list, we avoid scanning the whole DB here; return empty fast.
        return pd.DataFrame(columns=["species", "soil_type_preference", "water_need"])

    traits_map = kb.get_species_traits(names)
    rows = []
    for nm, traits in traits_map.items():
        rows.append({
            "species": nm,
            "soil_type_preference": traits.get("soil_type_preference"),
            "water_need": traits.get("water_need"),
        })
    return pd.DataFrame(rows, columns=["species", "soil_type_preference", "water_need"])

# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Run with: python -m manthan_core.recommender.rules_engine   (preferred)
    # or:       python manthan_core/recommender/rules_engine.py
    demo_species = ["Azadirachta indica", "Dalbergia sissoo"]
    print("compatibility:", calculate_species_compatibility(demo_species))
    print("water score (Neem, site=Medium):", calculate_water_requirement_score("Azadirachta indica", "Medium"))

    demo_site = {"soil_ph": 7.2, "annual_precip_mm": 900}
    demo_catalog = pd.DataFrame([
        {"species": "Azadirachta indica", "soil_type_preference": "alkaline", "water_need": "Low"},
        {"species": "Dalbergia sissoo",   "soil_type_preference": "neutral",  "water_need": "Medium"},
        {"species": "Ficus religiosa",    "soil_type_preference": "acidic",   "water_need": "High"},
    ])
    print("catalog ranking:\n", recommend_species_for_site(demo_site, demo_catalog, k=3))
