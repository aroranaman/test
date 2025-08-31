# manthan_core/water_management/water_optimizer.py
from __future__ import annotations
from typing import List, Dict, Any

class WaterManagementOptimizer:
    """
    A class to handle water-related calculations for a restoration project.
    It receives data and performs calculations based on it.
    """

    def __init__(self, species_traits_db: Dict[str, Dict[str, Any]]):
        """
        Initializes the optimizer with a dictionary of species traits.
        
        Args:
            species_traits_db: A dictionary where keys are scientific names and
                               values are dictionaries of traits for that species.
                               e.g., {'Azadirachta indica': {'water_need': 'Low', ...}}
        """
        self.species_traits_db = species_traits_db

    def calculate_irrigation_needs(self, species_list: List[str], site_rainfall_mm: float) -> Dict[str, Any]:
        """
        Calculates the estimated irrigation needed per species per year.

        Args:
            species_list: List of species names for the project.
            site_rainfall_mm: Average annual rainfall at the site in mm.

        Returns:
            A dictionary with the total deficit and per-species needs in mm/year.
        """
        # A simplified model mapping qualitative needs to quantitative values
        water_need_map = {"Low": 600, "Medium": 1000, "High": 1500}
        total_deficit = 0
        species_deficits = {}

        for species_name in species_list:
            # Look up traits for the current species
            traits = self.species_traits_db.get(species_name, {})
            # Get the water need, defaulting to 'Medium' if not specified
            need_category = traits.get("water_need", "Medium")
            
            species_water_need_mm = water_need_map.get(need_category, 1000)
            
            deficit = max(0, species_water_need_mm - site_rainfall_mm)
            species_deficits[species_name] = round(deficit, 2)
            total_deficit += deficit
        
        # Calculate the average deficit across all selected species
        average_deficit = (total_deficit / len(species_list)) if species_list else 0

        return {
            "total_irrigation_deficit_mm_per_year": round(average_deficit, 2),
            "deficits_by_species": species_deficits
        }

    def design_water_harvesting(self, aoi_avg_slope: float, aoi_area_ha: float) -> Dict[str, Any]:
        """
        Suggests water harvesting structures based on topography.

        Args:
            aoi_avg_slope: Average slope of the AOI in degrees.
            aoi_area_ha: Area of the AOI in hectares.

        Returns:
            A dictionary with suggested structures and estimated capacity.
        """
        suggestions = []
        if aoi_avg_slope > 8:
            suggestions.append("Contour Trenches: Effective for steep slopes to slow runoff and increase infiltration.")
        elif aoi_avg_slope > 3:
            suggestions.append("Check Dams: Suitable for moderately sloped areas with gullies or streams.")
        else:
            suggestions.append("Farm Ponds: Ideal for flat areas to store rainwater for later use.")

        # Simplified capacity estimation: assuming 1m rainfall and 20% runoff capture
        estimated_harvest_m3 = (10000 * aoi_area_ha * 0.20)
            
        return {
            "suggestions": suggestions,
            "estimated_annual_harvest_capacity_m3": round(estimated_harvest_m3, 2)
        }