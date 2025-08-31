# manthan_core/recommender/rules_engine.py
from __future__ import annotations

import os
from typing import List, Dict, Any

# Prefer relative import (works when run as module: python -m manthan_core.recommender.rules_engine)
try:
    from ..utils.db_connector import KnowledgeBase
except ImportError:  # fallback if someone runs the file directly
    from manthan_core.utils.db_connector import KnowledgeBase

DB_URL = os.getenv("MANTHAN_DB_URL", "sqlite:///data/manthan.db")
kb = KnowledgeBase(db_url=DB_URL)


def _normalize_truthy(val: Any) -> bool:
    """Accept a variety of truthy encodings from the traits table."""
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y", "t", "high", "symbiotic"}


def calculate_water_requirement_score(species_name: str, site_water_availability: str) -> float:
    """
    Scores a species based on its water needs vs site water availability,
    using live data from the knowledge base.

    Looks up 'water_need'. If absent, infers from 'drought_tolerance':
       High drought_tolerance -> Low water_need
       Medium -> Medium
       Low -> High
    """
    traits = kb.get_species_traits([species_name]).get(species_name, {}) or {}

    water_need = traits.get("water_need")
    if not water_need:
        # Infer from drought_tolerance if available
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
        # Find any fixer-like trait and treat truthy values as positive
        if any(_normalize_truthy(traits.get(k)) for k in fixer_names if k in traits):
            fixers.append(name)

    if fixers:
        comments.append(f"Positive: includes nitrogen-fixing species: {', '.join(sorted(set(fixers)))}.")
        score = 0.8
    else:
        comments.append("Neutral: no nitrogen-fixing species identified.")
        score = 0.5

    # Future: layer in canopy compatibility, allelopathy, pest/disease complementarity, etc.
    return {"compatibility_score": score, "comments": comments}


def generate_agroforestry_plan(species_list: List[str], farmer_income_needs: float) -> Dict[str, Any]:
    """
    Suggests intercropping with NTFPs/Fruit to ensure short-term income,
    using live economic data from the knowledge base.

    Expects an optional 'economic_data' table:
      economic_data(species_key, product_type, yield_per_ha, price_per_unit)
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


if __name__ == "__main__":
    # Tiny smoke test (won't fail if tables are missing)
    demo_species = ["Azadirachta indica", "Dalbergia sissoo"]
    print("compatibility:", calculate_species_compatibility(demo_species))
    print("water score (Neem, site=Medium):", calculate_water_requirement_score("Azadirachta indica", "Medium"))
