# enhanced_water_management_system.py
"""
Enhanced Water Management System for Manthan
Integrates water optimization as a core factor in restoration planning.

Key Features:
1. Water balance analysis using site rainfall and species water needs
2. Water-efficient species selection and scoring
3. Water harvesting structure optimization
4. Seasonal water demand forecasting
5. Integration with environmental compatibility scoring
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class WaterBalance:
    """Data class for water balance calculations"""
    annual_supply_mm: float
    annual_demand_mm: float
    deficit_mm: float
    surplus_mm: float
    efficiency_ratio: float

@dataclass
class WaterHarvestingRecommendation:
    """Data class for water harvesting recommendations"""
    structure_type: str
    capacity_m3: float
    cost_estimate_inr: float
    maintenance_level: str
    suitability_score: float

class EnhancedWaterManagementSystem:
    """
    Enhanced water management system that integrates with Manthan's species database
    and environmental analysis for comprehensive restoration planning.
    """
    
    def __init__(self, db_path: str = "data/manthan.db"):
        self.db_path = db_path
        
        # Water requirement categories (mm/year) - more refined
        self.water_need_categories = {
            'Very Low': 300,    # Desert species, succulents
            'Low': 600,         # Drought-tolerant species
            'Moderate': 1000,   # Most temperate species
            'High': 1500,       # Water-loving species
            'Very High': 2500   # Riparian, wetland species
        }
        
        # Seasonal distribution (% of annual need by month)
        self.seasonal_distribution = {
            'Jan': 0.05, 'Feb': 0.05, 'Mar': 0.08, 'Apr': 0.12,
            'May': 0.15, 'Jun': 0.12, 'Jul': 0.08, 'Aug': 0.08,
            'Sep': 0.08, 'Oct': 0.08, 'Nov': 0.06, 'Dec': 0.05
        }
    
    def get_species_water_requirements(self, species_list: List[str]) -> Dict[str, float]:
        """
        Get water requirements for species from database, with intelligent fallback
        
        Args:
            species_list: List of species names
            
        Returns:
            Dictionary mapping species names to annual water needs (mm)
        """
        
        conn = sqlite3.connect(self.db_path)
        water_requirements = {}
        
        for species_name in species_list:
            try:
                # Try to get water need from traits table
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT trait_value FROM species_traits 
                    WHERE species_key IN (
                        SELECT species_key FROM species WHERE canonical_name = ?
                    ) AND trait_name = 'water_need'
                """, (species_name,))
                
                result = cursor.fetchone()
                
                if result:
                    water_category = result[0]
                    water_requirements[species_name] = self.water_need_categories.get(water_category, 1000)
                else:
                    # Fallback: Infer water needs from drought tolerance and rainfall range
                    water_need = self._infer_water_need_from_traits(species_name, conn)
                    water_requirements[species_name] = water_need
                    
            except Exception as e:
                # Default to moderate water need
                water_requirements[species_name] = 1000
        
        conn.close()
        return water_requirements
    
    def _infer_water_need_from_traits(self, species_name: str, conn) -> float:
        """Infer water needs from existing environmental traits"""
        
        cursor = conn.cursor()
        
        # Get drought tolerance and rainfall range
        cursor.execute("""
            SELECT 
                MAX(CASE WHEN trait_name = 'drought_tolerance' THEN trait_value END) as drought_tol,
                MAX(CASE WHEN trait_name = 'min_rainfall_mm' THEN CAST(trait_value AS REAL) END) as min_rain,
                MAX(CASE WHEN trait_name = 'optimal_rainfall_mm' THEN CAST(trait_value AS REAL) END) as opt_rain
            FROM species_traits st
            JOIN species s ON st.species_key = s.species_key
            WHERE s.canonical_name = ?
            AND trait_name IN ('drought_tolerance', 'min_rainfall_mm', 'optimal_rainfall_mm')
        """, (species_name,))
        
        result = cursor.fetchone()
        
        if result and any(result):
            drought_tol, min_rain, opt_rain = result
            
            # Infer water need from drought tolerance
            if drought_tol == 'High':
                base_need = 600  # Low water need
            elif drought_tol == 'Moderate':
                base_need = 1000  # Moderate water need
            else:
                base_need = 1200  # Higher water need
            
            # Adjust based on optimal rainfall if available
            if opt_rain and opt_rain > 0:
                # Species adapted to higher rainfall likely need more water
                if opt_rain > 1500:
                    base_need = min(2000, base_need * 1.5)
                elif opt_rain < 500:
                    base_need = max(400, base_need * 0.7)
            
            return base_need
        
        return 1000  # Default moderate need
    
    def calculate_comprehensive_water_balance(self, 
                                            species_list: List[str], 
                                            site_conditions: Dict[str, float],
                                            planting_density: int = 400) -> Dict[str, Any]:
        """
        Calculate comprehensive water balance for restoration site
        
        Args:
            species_list: List of species to be planted
            site_conditions: Site environmental conditions
            planting_density: Plants per hectare
            
        Returns:
            Comprehensive water balance analysis
        """
        
        annual_rainfall = site_conditions.get('annual_precip_mm', 800)
        slope_deg = site_conditions.get('slope_deg', 5)
        ndvi = site_conditions.get('ndvi_mean', 0.4)
        elevation = site_conditions.get('elevation_m', 200)
        
        # Get species water requirements
        species_water_needs = self.get_species_water_requirements(species_list)
        
        # Calculate effective rainfall (accounting for runoff and evaporation)
        effective_rainfall = self._calculate_effective_rainfall(annual_rainfall, slope_deg, ndvi)
        
        # Calculate weighted average water demand
        total_demand = sum(species_water_needs.values())
        avg_demand_per_species = total_demand / len(species_list) if species_list else 1000
        
        # Adjust for planting density (more plants = higher total demand per hectare)
        demand_per_hectare = (avg_demand_per_species * planting_density) / 1000  # m3/ha/year
        
        # Water balance calculations
        supply_per_hectare = effective_rainfall / 100  # m3/ha/year (convert mm to m3/ha)
        
        balance = {
            'annual_rainfall_mm': annual_rainfall,
            'effective_rainfall_mm': effective_rainfall,
            'average_species_demand_mm': round(avg_demand_per_species, 2),
            'demand_per_hectare_m3': round(demand_per_hectare, 2),
            'supply_per_hectare_m3': round(supply_per_hectare, 2),
            'deficit_per_hectare_m3': round(max(0, demand_per_hectare - supply_per_hectare), 2),
            'surplus_per_hectare_m3': round(max(0, supply_per_hectare - demand_per_hectare), 2),
            'water_efficiency_ratio': round(supply_per_hectare / demand_per_hectare if demand_per_hectare > 0 else 1, 3),
            'species_water_needs': species_water_needs,
            'irrigation_needed': demand_per_hectare > supply_per_hectare
        }
        
        # Water stress analysis
        balance['water_stress_level'] = self._assess_water_stress(balance['water_efficiency_ratio'])
        
        # Seasonal analysis
        balance['seasonal_analysis'] = self._calculate_seasonal_demands(
            species_water_needs, annual_rainfall
        )
        
        return balance
    
    def _calculate_effective_rainfall(self, annual_rainfall: float, slope_deg: float, ndvi: float) -> float:
        """Calculate effective rainfall accounting for runoff and infiltration"""
        
        # Runoff coefficient based on slope and vegetation
        # Higher slope = more runoff, higher NDVI = less runoff
        runoff_coefficient = min(0.8, max(0.1, (slope_deg / 30) - (ndvi * 0.3)))
        
        # Evaporation losses (rough estimate)
        evaporation_loss = 0.15  # 15% loss to evaporation
        
        # Effective rainfall
        effective_rainfall = annual_rainfall * (1 - runoff_coefficient) * (1 - evaporation_loss)
        
        return max(100, effective_rainfall)  # Minimum 100mm effective
    
    def _assess_water_stress(self, efficiency_ratio: float) -> str:
        """Assess water stress level based on supply/demand ratio"""
        
        if efficiency_ratio >= 1.2:
            return 'Low Stress'
        elif efficiency_ratio >= 0.8:
            return 'Moderate Stress'
        elif efficiency_ratio >= 0.5:
            return 'High Stress'
        else:
            return 'Critical Stress'
    
    def _calculate_seasonal_demands(self, species_water_needs: Dict[str, float], annual_rainfall: float) -> Dict:
        """Calculate seasonal water demands and identify critical periods"""
        
        # Typical Indian monsoon rainfall distribution (approximate)
        monthly_rainfall_distribution = {
            'Jan': 0.02, 'Feb': 0.02, 'Mar': 0.03, 'Apr': 0.05,
            'May': 0.08, 'Jun': 0.20, 'Jul': 0.25, 'Aug': 0.20,
            'Sep': 0.10, 'Oct': 0.03, 'Nov': 0.01, 'Dec': 0.01
        }
        
        seasonal_analysis = {}
        critical_months = []
        
        avg_species_demand = sum(species_water_needs.values()) / len(species_water_needs) if species_water_needs else 1000
        
        for month, demand_fraction in self.seasonal_distribution.items():
            monthly_demand = avg_species_demand * demand_fraction
            monthly_supply = annual_rainfall * monthly_rainfall_distribution.get(month, 0.05)
            
            deficit = max(0, monthly_demand - monthly_supply)
            
            seasonal_analysis[month] = {
                'demand_mm': round(monthly_demand, 2),
                'supply_mm': round(monthly_supply, 2),
                'deficit_mm': round(deficit, 2)
            }
            
            if deficit > 50:  # More than 50mm deficit
                critical_months.append(month)
        
        return {
            'monthly_analysis': seasonal_analysis,
            'critical_months': critical_months,
            'peak_demand_months': ['May', 'Jun', 'Jul'],  # Growth season
            'peak_supply_months': ['Jun', 'Jul', 'Aug']   # Monsoon
        }
    
    def design_water_harvesting_system(self, 
                                     site_conditions: Dict[str, float],
                                     area_ha: float,
                                     water_deficit_m3: float) -> List[WaterHarvestingRecommendation]:
        """
        Design comprehensive water harvesting system
        
        Args:
            site_conditions: Site environmental conditions
            area_ha: Area in hectares
            water_deficit_m3: Annual water deficit in cubic meters
            
        Returns:
            List of water harvesting recommendations
        """
        
        slope_deg = site_conditions.get('slope_deg', 5)
        annual_rainfall = site_conditions.get('annual_precip_mm', 800)
        soil_type = 'mixed'  # Could be inferred from pH and other factors
        
        recommendations = []
        
        # Calculate potential harvest capacity
        potential_harvest = (area_ha * 10000 * annual_rainfall / 1000) * 0.25  # 25% capture efficiency
        
        # Structure recommendations based on slope and area
        if slope_deg > 15:  # Steep slopes
            recommendations.extend([
                WaterHarvestingRecommendation(
                    structure_type="Contour Trenches",
                    capacity_m3=area_ha * 100,  # 100 m3/ha
                    cost_estimate_inr=area_ha * 15000,  # ₹15,000 per ha
                    maintenance_level="Medium",
                    suitability_score=0.9
                ),
                WaterHarvestingRecommendation(
                    structure_type="Stone Check Dams",
                    capacity_m3=min(500, area_ha * 50),
                    cost_estimate_inr=min(50000, area_ha * 8000),
                    maintenance_level="Low",
                    suitability_score=0.8
                )
            ])
            
        elif slope_deg > 5:  # Moderate slopes
            recommendations.extend([
                WaterHarvestingRecommendation(
                    structure_type="Farm Ponds",
                    capacity_m3=min(1000, water_deficit_m3 * 0.6),
                    cost_estimate_inr=min(100000, water_deficit_m3 * 80),
                    maintenance_level="Medium",
                    suitability_score=0.9
                ),
                WaterHarvestingRecommendation(
                    structure_type="Percolation Tanks",
                    capacity_m3=area_ha * 200,
                    cost_estimate_inr=area_ha * 25000,
                    maintenance_level="High",
                    suitability_score=0.7
                )
            ])
            
        else:  # Gentle slopes
            recommendations.extend([
                WaterHarvestingRecommendation(
                    structure_type="Large Farm Ponds",
                    capacity_m3=min(2000, water_deficit_m3 * 0.8),
                    cost_estimate_inr=water_deficit_m3 * 100,
                    maintenance_level="Medium",
                    suitability_score=0.95
                ),
                WaterHarvestingRecommendation(
                    structure_type="Recharge Wells",
                    capacity_m3=area_ha * 150,
                    cost_estimate_inr=area_ha * 20000,
                    maintenance_level="Low",
                    suitability_score=0.8
                )
            ])
        
        # Sort by suitability score
        recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def calculate_water_efficiency_scores(self, species_list: List[str], site_rainfall: float) -> Dict[str, float]:
        """
        Calculate water efficiency scores for species selection
        
        Args:
            species_list: List of species names
            site_rainfall: Annual rainfall at site (mm)
            
        Returns:
            Dictionary mapping species to water efficiency scores (0-1)
        """
        
        species_water_needs = self.get_species_water_requirements(species_list)
        efficiency_scores = {}
        
        for species, water_need in species_water_needs.items():
            # Score based on how well species water need matches site rainfall
            if water_need <= site_rainfall:
                # Species can survive on rainfall alone
                efficiency_scores[species] = min(1.0, site_rainfall / water_need)
            else:
                # Species needs irrigation
                deficit = water_need - site_rainfall
                # Penalize high water need species on dry sites
                efficiency_scores[species] = max(0.1, 1.0 - (deficit / water_need))
        
        return efficiency_scores

def integrate_water_management_with_recommendations(recommendations_df: pd.DataFrame, 
                                                  site_conditions: Dict[str, float],
                                                  water_weight: float = 0.2) -> pd.DataFrame:
    """
    Integrate water management scoring with existing species recommendations
    
    Args:
        recommendations_df: DataFrame with species recommendations
        site_conditions: Site environmental conditions
        water_weight: Weight for water efficiency in overall scoring (0-1)
        
    Returns:
        Enhanced recommendations with water efficiency scoring
    """
    
    water_system = EnhancedWaterManagementSystem()
    species_list = recommendations_df['canonical_name'].tolist()
    site_rainfall = site_conditions.get('annual_precip_mm', 800)
    
    # Calculate water efficiency scores
    water_scores = water_system.calculate_water_efficiency_scores(species_list, site_rainfall)
    
    # Add water efficiency scores to dataframe
    recommendations_df['water_efficiency_score'] = recommendations_df['canonical_name'].map(
        lambda x: water_scores.get(x, 0.5)
    )
    
    # Recalculate overall score including water efficiency
    recommendations_df['overall_score_with_water'] = (
        recommendations_df['overall_score'] * (1 - water_weight) +
        recommendations_df['water_efficiency_score'] * water_weight
    )
    
    # Add water requirement information
    species_water_needs = water_system.get_species_water_requirements(species_list)
    recommendations_df['annual_water_need_mm'] = recommendations_df['canonical_name'].map(
        lambda x: species_water_needs.get(x, 1000)
    )
    
    # Sort by new overall score
    recommendations_df = recommendations_df.sort_values('overall_score_with_water', ascending=False)
    
    return recommendations_df

# Example usage function
def demo_water_integration():
    """Demonstrate water management integration"""
    
    print("🌊 MANTHAN WATER MANAGEMENT INTEGRATION DEMO")
    print("="*60)
    
    # Example site conditions
    site_conditions = {
        'annual_precip_mm': 650,
        'slope_deg': 8.5,
        'ndvi_mean': 0.35,
        'elevation_m': 250,
        'soil_ph': 7.2
    }
    
    # Example species list
    species_list = [
        'Azadirachta indica',
        'Prosopis cineraria', 
        'Acacia nilotica',
        'Dalbergia sissoo',
        'Terminalia arjuna'
    ]
    
    water_system = EnhancedWaterManagementSystem()
    
    # Water balance analysis
    print("\n💧 Water Balance Analysis:")
    balance = water_system.calculate_comprehensive_water_balance(
        species_list, site_conditions, planting_density=400
    )
    
    print(f"Annual Rainfall: {balance['annual_rainfall_mm']}mm")
    print(f"Effective Rainfall: {balance['effective_rainfall_mm']}mm")
    print(f"Water Efficiency Ratio: {balance['water_efficiency_ratio']}")
    print(f"Water Stress Level: {balance['water_stress_level']}")
    print(f"Irrigation Needed: {'Yes' if balance['irrigation_needed'] else 'No'}")
    
    if balance['deficit_per_hectare_m3'] > 0:
        print(f"Water Deficit: {balance['deficit_per_hectare_m3']} m³/ha/year")
        
        # Water harvesting recommendations
        print("\n🏗️ Water Harvesting Recommendations:")
        recommendations = water_system.design_water_harvesting_system(
            site_conditions, 5.0, balance['deficit_per_hectare_m3'] * 5
        )
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec.structure_type}")
            print(f"   Capacity: {rec.capacity_m3} m³")
            print(f"   Cost: ₹{rec.cost_estimate_inr:,.0f}")
            print(f"   Suitability: {rec.suitability_score:.1f}/1.0")
    
    # Species water efficiency
    print("\n🌱 Species Water Efficiency:")
    efficiency_scores = water_system.calculate_water_efficiency_scores(
        species_list, site_conditions['annual_precip_mm']
    )
    
    for species, score in sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True):
        water_need = water_system.get_species_water_requirements([species])[species]
        print(f"• {species}: {score:.2f} (needs {water_need}mm/year)")

if __name__ == "__main__":
    demo_water_integration()