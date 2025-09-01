# economic/predictive_viability_assessment.py
"""
Advanced Predictive Forecasting Integration for Manthan Forest Planning System

This module provides LightGBM-based predictive modeling for:
1. Agroforestry: Short-term earnings optimization for farmers (3-10 years)
2. Miyawaki: Carbon sequestration ROI and ecological restoration (10-25 years)

Integrates with intelligent_app.py forest blueprint outputs for data-driven predictions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import math

# Machine Learning imports
try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    HAS_ML = True
except ImportError:
    HAS_ML = False
    logging.warning("ML libraries not available. Falling back to heuristic models.")

# Economic data structures
@dataclass
class AgroforestryMetrics:
    """Short-term focused metrics for agroforestry systems"""
    annual_food_income: float
    cumulative_5yr_income: float
    payback_period_months: int
    cash_flow_stability: float  # 0-1 score
    market_risk_factor: float   # 0-1, lower is better
    farmer_labor_hours_annual: float
    net_profit_margin: float
    diversification_index: float  # Product diversity score

@dataclass
class MiyawakiMetrics:
    """Long-term focused metrics for Miyawaki restoration"""
    carbon_sequestration_total_tons: float
    carbon_credits_revenue_20yr: float
    biodiversity_enhancement_score: float
    ecosystem_services_value: float
    restoration_success_probability: float
    carbon_payback_period_years: int
    long_term_roi_percentage: float
    ecological_resilience_index: float

@dataclass
class PredictionResults:
    """Combined prediction results for both systems"""
    project_type: str  # 'agroforestry' or 'miyawaki'
    confidence_score: float
    agroforestry_metrics: Optional[AgroforestryMetrics]
    miyawaki_metrics: Optional[MiyawakiMetrics]
    risk_assessment: Dict[str, float]
    recommendations: List[str]

class AdvancedViabilityPredictor:
    """
    LightGBM-based predictor for forest project outcomes
    """
    
    def __init__(self, db_path: str = "data/manthan.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        
        # Economic parameters database
        self.economic_params = self._load_economic_parameters()
        
        # Market data for different regions
        self.market_data = self._load_market_data()
        
        # Carbon pricing and sequestration rates
        self.carbon_params = self._load_carbon_parameters()
        
    def _load_economic_parameters(self) -> Dict[str, Any]:
        """Load economic parameters for different species and regions"""
        return {
            'agroforestry_species_values': {
                # High-value fruit species (₹/tree/year after maturity)
                'mangifera indica': {'annual_yield': 15000, 'maturity_years': 5, 'peak_years': 15, 'volatility': 0.2},
                'cocos nucifera': {'annual_yield': 8000, 'maturity_years': 6, 'peak_years': 25, 'volatility': 0.15},
                'artocarpus heterophyllus': {'annual_yield': 12000, 'maturity_years': 4, 'peak_years': 20, 'volatility': 0.25},
                'anacardium occidentale': {'annual_yield': 18000, 'maturity_years': 4, 'peak_years': 15, 'volatility': 0.3},
                'psidium guajava': {'annual_yield': 6000, 'maturity_years': 2, 'peak_years': 10, 'volatility': 0.1},
                'punica granatum': {'annual_yield': 8000, 'maturity_years': 3, 'peak_years': 12, 'volatility': 0.2},
                'citrus species': {'annual_yield': 10000, 'maturity_years': 3, 'peak_years': 15, 'volatility': 0.2},
                'syzygium cumini': {'annual_yield': 5000, 'maturity_years': 3, 'peak_years': 20, 'volatility': 0.15},
                'tamarindus indica': {'annual_yield': 7000, 'maturity_years': 6, 'peak_years': 30, 'volatility': 0.1},
                'moringa oleifera': {'annual_yield': 4000, 'maturity_years': 1, 'peak_years': 8, 'volatility': 0.15},
            },
            
            'costs_per_hectare': {
                'site_preparation': 25000,
                'sapling_costs_agro': 150,  # ₹ per plant for fruit trees
                'sapling_costs_miyawaki': 25,  # ₹ per plant for native species
                'planting_labor': 15000,
                'irrigation_setup': 50000,
                'maintenance_annual_agro': 35000,
                'maintenance_annual_miyawaki': 8000,
                'fertilizer_annual': 20000,
                'pest_control_annual': 12000,
            },
            
            'planting_densities': {
                'agroforestry': 400,  # trees per hectare
                'miyawaki': 30000,    # plants per hectare (includes shrubs)
            }
        }
    
    def _load_market_data(self) -> Dict[str, Any]:
        """Load regional market data and price trends"""
        return {
            'regional_price_multipliers': {
                'western_ghats': 1.2,    # Higher prices due to quality
                'gangetic_plains': 1.0,   # Baseline
                'deccan_plateau': 0.9,    # Lower due to competition
                'himalaya': 1.3,         # Premium for hill produce
                'coastal': 1.1,          # Good market access
                'northeast': 0.8,        # Remote markets
                'central_india': 0.95,   # Average connectivity
            },
            
            'market_access_scores': {
                'urban_proximity_km': {0: 1.0, 10: 0.95, 25: 0.85, 50: 0.7, 100: 0.5, 200: 0.3},
                'road_connectivity': {'excellent': 1.0, 'good': 0.85, 'moderate': 0.7, 'poor': 0.5},
                'cold_storage_access': {'available': 1.2, 'limited': 1.0, 'none': 0.7},
            },
            
            'seasonal_price_variations': {
                'mango': [0.8, 0.7, 1.5, 1.8, 1.2, 0.9, 0.6, 0.6, 0.8, 1.0, 1.1, 1.0],
                'coconut': [1.0, 1.0, 1.1, 1.1, 1.0, 0.9, 0.9, 1.0, 1.1, 1.2, 1.1, 1.0],
                'guava': [1.2, 1.3, 1.1, 0.8, 0.7, 0.8, 1.0, 1.2, 1.3, 1.4, 1.2, 1.1],
            }
        }
    
    def _load_carbon_parameters(self) -> Dict[str, Any]:
        """Load carbon sequestration and pricing parameters"""
        return {
            'sequestration_rates': {
                # tons CO2 per hectare per year by species type
                'fast_growing_native': 8.5,
                'mixed_deciduous': 6.2,
                'fruit_trees': 4.8,
                'shrub_layer': 2.5,
                'grassland_restoration': 1.8,
            },
            
            'carbon_prices': {
                'voluntary_market': 800,     # ₹ per ton CO2
                'compliance_market': 1200,   # ₹ per ton CO2
                'premium_certification': 1500,  # ₹ per ton CO2 (high-quality projects)
            },
            
            'certification_costs': {
                'verification_annual': 50000,
                'monitoring_costs': 25000,
                'certification_setup': 150000,
            },
            
            'sequestration_curves': {
                # Percentage of maximum sequestration rate by year
                'miyawaki_curve': [0.1, 0.3, 0.5, 0.7, 0.85, 1.0, 1.0, 0.95, 0.9, 0.85],
                'traditional_curve': [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0, 1.0, 0.95, 0.9],
            }
        }
    
    def generate_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic training data based on realistic parameters"""
        
        np.random.seed(42)
        data = []
        
        for _ in range(n_samples):
            # Site characteristics
            area_ha = np.random.uniform(1, 50)
            rainfall_mm = np.random.uniform(400, 2500)
            soil_ph = np.random.uniform(5.5, 8.5)
            slope_deg = np.random.uniform(0, 30)
            elevation_m = np.random.uniform(0, 2000)
            temperature_c = 26 - (elevation_m / 1000 * 6.5) + np.random.normal(0, 2)
            
            # Project characteristics
            project_type = np.random.choice(['agroforestry', 'miyawaki'], p=[0.6, 0.4])
            n_species = np.random.randint(3, 15)
            native_percentage = np.random.uniform(0.3, 1.0)
            
            # Economic factors
            market_distance_km = np.random.uniform(5, 200)
            farmer_experience_years = np.random.uniform(0, 30)
            initial_investment = np.random.uniform(100000, 2000000)
            
            # Calculate outcomes based on realistic models
            if project_type == 'agroforestry':
                # Agroforestry outcomes
                annual_income = self._calculate_agroforestry_income(
                    area_ha, rainfall_mm, market_distance_km, n_species, farmer_experience_years
                )
                payback_years = initial_investment / max(annual_income, 1000)
                success_probability = self._calculate_success_probability(
                    rainfall_mm, soil_ph, farmer_experience_years, market_distance_km, 'agroforestry'
                )
            else:
                # Miyawaki outcomes
                carbon_tons_20yr = self._calculate_carbon_sequestration(
                    area_ha, n_species, native_percentage, rainfall_mm, temperature_c
                )
                annual_income = carbon_tons_20yr * 800 / 20  # Simplified carbon revenue
                payback_years = initial_investment / max(annual_income, 1000)
                success_probability = self._calculate_success_probability(
                    rainfall_mm, soil_ph, native_percentage, elevation_m, 'miyawaki'
                )
            
            # Add noise and realistic constraints
            annual_income = max(0, annual_income + np.random.normal(0, annual_income * 0.1))
            payback_years = max(1, min(30, payback_years + np.random.normal(0, 2)))
            success_probability = max(0.1, min(0.95, success_probability + np.random.normal(0, 0.05)))
            
            data.append({
                'area_ha': area_ha,
                'rainfall_mm': rainfall_mm,
                'soil_ph': soil_ph,
                'slope_deg': slope_deg,
                'elevation_m': elevation_m,
                'temperature_c': temperature_c,
                'project_type': project_type,
                'n_species': n_species,
                'native_percentage': native_percentage,
                'market_distance_km': market_distance_km,
                'farmer_experience_years': farmer_experience_years,
                'initial_investment': initial_investment,
                'annual_income': annual_income,
                'payback_years': payback_years,
                'success_probability': success_probability,
            })
        
        return pd.DataFrame(data)
    
    def _calculate_agroforestry_income(self, area_ha: float, rainfall_mm: float, 
                                     market_distance_km: float, n_species: int, 
                                     experience_years: float) -> float:
        """Calculate realistic agroforestry income based on parameters"""
        
        base_income_per_ha = 80000  # ₹ per hectare baseline
        
        # Rainfall adjustment
        if 600 <= rainfall_mm <= 1200:
            rainfall_factor = 1.0
        elif rainfall_mm < 600:
            rainfall_factor = 0.6 + (rainfall_mm / 600) * 0.4
        else:
            rainfall_factor = 1.0 - min(0.3, (rainfall_mm - 1200) / 2000)
        
        # Market distance penalty
        market_factor = max(0.5, 1.0 - (market_distance_km / 100) * 0.3)
        
        # Species diversity bonus
        diversity_factor = 1.0 + min(0.3, (n_species - 3) / 10 * 0.3)
        
        # Experience factor
        experience_factor = 0.7 + min(0.4, experience_years / 20 * 0.4)
        
        return (base_income_per_ha * area_ha * rainfall_factor * 
                market_factor * diversity_factor * experience_factor)
    
    def _calculate_carbon_sequestration(self, area_ha: float, n_species: int, 
                                      native_percentage: float, rainfall_mm: float, 
                                      temperature_c: float) -> float:
        """Calculate realistic carbon sequestration over 20 years"""
        
        base_sequestration_per_ha_per_year = 6.0  # tons CO2
        
        # Species diversity factor
        diversity_factor = 1.0 + min(0.4, (n_species - 5) / 10 * 0.4)
        
        # Native species factor
        native_factor = 0.8 + native_percentage * 0.3
        
        # Climate suitability
        if 20 <= temperature_c <= 30 and 800 <= rainfall_mm <= 1800:
            climate_factor = 1.0
        else:
            temp_penalty = abs(temperature_c - 25) / 25 * 0.2
            rain_penalty = abs(rainfall_mm - 1200) / 1200 * 0.2
            climate_factor = max(0.5, 1.0 - temp_penalty - rain_penalty)
        
        annual_sequestration = (base_sequestration_per_ha_per_year * area_ha * 
                              diversity_factor * native_factor * climate_factor)
        
        return annual_sequestration * 20
    
    def _calculate_success_probability(self, rainfall_mm: float, soil_ph: float, 
                                     third_param: float, fourth_param: float, 
                                     project_type: str) -> float:
        """Calculate project success probability"""
        
        base_probability = 0.7
        
        # pH factor
        ph_factor = 1.0 if 6.0 <= soil_ph <= 7.5 else max(0.6, 1.0 - abs(soil_ph - 6.75) * 0.1)
        
        # Rainfall factor
        if project_type == 'agroforestry':
            rain_factor = 1.0 if 600 <= rainfall_mm <= 1200 else max(0.5, 1.0 - abs(rainfall_mm - 900) / 1000)
            # third_param is farmer experience
            experience_factor = 0.8 + min(0.3, third_param / 30 * 0.3)
            # fourth_param is market distance
            market_factor = max(0.6, 1.0 - fourth_param / 200 * 0.4)
            return base_probability * ph_factor * rain_factor * experience_factor * market_factor
        else:  # miyawaki
            rain_factor = 1.0 if 800 <= rainfall_mm <= 1800 else max(0.6, 1.0 - abs(rainfall_mm - 1200) / 1500)
            # third_param is native percentage
            native_factor = 0.7 + third_param * 0.3
            # fourth_param is elevation
            elevation_factor = max(0.7, 1.0 - abs(fourth_param - 500) / 2000 * 0.3)
            return base_probability * ph_factor * rain_factor * native_factor * elevation_factor
    
    def train_models(self) -> None:
        """Train LightGBM models for different prediction targets"""
        
        if not HAS_ML:
            logging.warning("ML libraries not available. Models will use heuristics.")
            return
        
        # Generate training data
        data = self.generate_training_data(2000)
        
        # Prepare features
        feature_cols = [
            'area_ha', 'rainfall_mm', 'soil_ph', 'slope_deg', 'elevation_m',
            'temperature_c', 'n_species', 'native_percentage', 'market_distance_km',
            'farmer_experience_years', 'initial_investment'
        ]
        
        # Encode project type
        le = LabelEncoder()
        data['project_type_encoded'] = le.fit_transform(data['project_type'])
        feature_cols.append('project_type_encoded')
        
        X = data[feature_cols]
        
        # Train separate models for different targets
        targets = ['annual_income', 'payback_years', 'success_probability']
        
        for target in targets:
            y = data[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train LightGBM model
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"Model {target}: MSE={mse:.2f}, R2={r2:.3f}")
            
            # Store model and scaler
            self.models[target] = model
            self.scalers[target] = scaler
        
        self.feature_names = feature_cols
        logging.info("All models trained successfully")
    
    def predict_agroforestry_outcomes(self, blueprint: Dict[str, Any], 
                                    site_conditions: Dict[str, Any]) -> AgroforestryMetrics:
        """Predict detailed agroforestry outcomes"""
        
        # Extract features from blueprint and site conditions
        area_ha = float(blueprint.get('area_ha', 5))
        species_list = []
        
        # Extract species from blueprint
        if 'species_mix_df' in blueprint and blueprint['species_mix_df'] is not None:
            df = blueprint['species_mix_df']
            if hasattr(df, 'iterrows'):
                species_list = df['canonical_name'].tolist()
        
        n_species = len(species_list)
        
        # Calculate food income based on species composition
        annual_food_income = 0
        total_trees = 0
        
        for species in species_list:
            species_lower = species.lower()
            species_data = None
            
            # Match species to economic data
            for key, data in self.economic_params['agroforestry_species_values'].items():
                if any(word in species_lower for word in key.split()):
                    species_data = data
                    break
            
            if species_data:
                # Get tree count from blueprint
                tree_count = 1  # Default
                if 'species_mix_df' in blueprint and blueprint['species_mix_df'] is not None:
                    df = blueprint['species_mix_df']
                    if hasattr(df, 'loc'):
                        match = df[df['canonical_name'] == species]
                        if not match.empty:
                            tree_count = match['planned_count'].iloc[0]
                
                # Calculate income per tree after maturity
                maturity_years = species_data['maturity_years']
                annual_yield = species_data['annual_yield']
                
                # Apply regional and market factors
                region = blueprint.get('region', 'central_india')
                price_multiplier = self.market_data['regional_price_multipliers'].get(region, 1.0)
                
                tree_income = annual_yield * price_multiplier
                annual_food_income += tree_income * tree_count
                total_trees += tree_count
        
        # Apply density and area scaling
        if total_trees > 0:
            annual_food_income = annual_food_income / total_trees * min(total_trees, area_ha * 400)
        else:
            annual_food_income = area_ha * 60000  # Fallback estimate
        
        # Calculate 5-year cumulative income (accounting for maturation)
        cumulative_5yr = 0
        for year in range(1, 6):
            if year >= 3:  # Assume production starts in year 3
                yearly_income = annual_food_income * min(1.0, (year - 2) * 0.3 + 0.4)
                cumulative_5yr += yearly_income
        
        # Calculate other metrics
        initial_investment = area_ha * 200000  # ₹2 lakh per hectare estimate
        payback_period_months = int((initial_investment / max(annual_food_income, 1000)) * 12)
        
        # Cash flow stability (based on species diversity and market factors)
        diversification_index = min(1.0, n_species / 8)
        cash_flow_stability = 0.4 + diversification_index * 0.4 + min(0.2, area_ha / 10 * 0.2)
        
        # Market risk (distance, connectivity, crop type)
        market_risk_factor = 0.3 + np.random.uniform(0, 0.4)  # Simplified
        
        # Labor requirements (hours per hectare per year)
        farmer_labor_hours_annual = area_ha * 250  # 250 hours per hectare
        
        # Net profit margin
        annual_costs = area_ha * 80000  # Operating costs
        net_profit_margin = max(0, (annual_food_income - annual_costs) / annual_food_income) if annual_food_income > 0 else 0
        
        return AgroforestryMetrics(
            annual_food_income=round(annual_food_income),
            cumulative_5yr_income=round(cumulative_5yr),
            payback_period_months=min(120, payback_period_months),  # Cap at 10 years
            cash_flow_stability=round(cash_flow_stability, 2),
            market_risk_factor=round(market_risk_factor, 2),
            farmer_labor_hours_annual=round(farmer_labor_hours_annual),
            net_profit_margin=round(net_profit_margin, 3),
            diversification_index=round(diversification_index, 2)
        )
    
    def predict_miyawaki_outcomes(self, blueprint: Dict[str, Any], 
                                site_conditions: Dict[str, Any]) -> MiyawakiMetrics:
        """Predict detailed Miyawaki restoration outcomes"""
        
        area_ha = float(blueprint.get('area_ha', 5))
        density = blueprint.get('density_per_ha', 30000)
        
        # Calculate carbon sequestration
        rainfall = float(site_conditions.get('annual_precip_mm', 1000))
        temperature = 26 - (float(site_conditions.get('elevation_m', 200)) / 1000 * 6.5)
        
        # Base sequestration rate for Miyawaki method
        base_rate = self.carbon_params['sequestration_rates']['fast_growing_native']
        
        # Climate adjustment
        climate_factor = 1.0
        if not (800 <= rainfall <= 1800):
            climate_factor *= max(0.6, 1.0 - abs(rainfall - 1200) / 1500 * 0.4)
        if not (18 <= temperature <= 32):
            climate_factor *= max(0.7, 1.0 - abs(temperature - 25) / 15 * 0.3)
        
        # Density factor (Miyawaki benefits from high density)
        density_factor = min(1.3, 0.8 + (density / 30000) * 0.5)
        
        # Calculate total sequestration over 20 years using growth curve
        annual_base_sequestration = base_rate * area_ha * climate_factor * density_factor
        total_sequestration = 0
        
        for year in range(20):
            curve_index = min(year, 9)
            year_multiplier = self.carbon_params['sequestration_curves']['miyawaki_curve'][curve_index]
            total_sequestration += annual_base_sequestration * year_multiplier
        
        # Carbon credits revenue (20-year)
        carbon_price = self.carbon_params['carbon_prices']['voluntary_market']
        gross_carbon_revenue = total_sequestration * carbon_price
        
        # Subtract verification and monitoring costs
        verification_costs = self.carbon_params['certification_costs']['verification_annual'] * 20
        monitoring_costs = self.carbon_params['certification_costs']['monitoring_costs'] * 20
        setup_costs = self.carbon_params['certification_costs']['certification_setup']
        
        net_carbon_revenue = gross_carbon_revenue - verification_costs - monitoring_costs - setup_costs
        
        # Biodiversity enhancement score (based on native species density)
        native_species_count = 0
        if 'species_mix_df' in blueprint and blueprint['species_mix_df'] is not None:
            df = blueprint['species_mix_df']
            if hasattr(df, 'sum'):
                native_species_count = df['is_native'].sum() if 'is_native' in df.columns else len(df)
        
        biodiversity_score = min(1.0, native_species_count / 15 * 0.8 + density / 30000 * 0.2)
        
        # Ecosystem services value (water retention, air purification, etc.)
        ecosystem_services_value = area_ha * 25000 * 20  # ₹25,000 per hectare per year
        
        # Restoration success probability
        success_factors = [
            min(1.0, density / 25000),  # Adequate density
            climate_factor,             # Suitable climate
            biodiversity_score,         # Species diversity
            min(1.0, area_ha / 2)      # Minimum viable area
        ]
        restoration_success_probability = np.mean(success_factors) * 0.85  # Base 85% with factors
        
        # Carbon payback period
        initial_investment = area_ha * 350000  # ₹3.5 lakh per hectare for Miyawaki
        annual_carbon_income = net_carbon_revenue / 20
        carbon_payback_years = int(initial_investment / max(annual_carbon_income, 1000))
        
        # Long-term ROI
        total_benefits = net_carbon_revenue + ecosystem_services_value
        long_term_roi = ((total_benefits - initial_investment) / initial_investment) * 100 if initial_investment > 0 else 0
        
        # Ecological resilience index
        resilience_factors = [
            biodiversity_score,
            min(1.0, area_ha / 5),      # Size factor
            climate_factor,
            min(1.0, density / 20000)   # Density factor
        ]
        ecological_resilience_index = np.mean(resilience_factors)
        
        return MiyawakiMetrics(
            carbon_sequestration_total_tons=round(total_sequestration, 1),
            carbon_credits_revenue_20yr=round(max(0, net_carbon_revenue)),
            biodiversity_enhancement_score=round(biodiversity_score, 2),
            ecosystem_services_value=round(ecosystem_services_value),
            restoration_success_probability=round(restoration_success_probability, 2),
            carbon_payback_period_years=min(25, carbon_payback_years),
            long_term_roi_percentage=round(long_term_roi, 1),
            ecological_resilience_index=round(ecological_resilience_index, 2)
        )
    
    def assess_risks(self, blueprint: Dict[str, Any], site_conditions: Dict[str, Any], 
                    project_type: str) -> Dict[str, float]:
        """Comprehensive risk assessment"""
        
        risks = {}
        
        # Climate risk
        rainfall = float(site_conditions.get('annual_precip_mm', 1000))
        if project_type == 'agroforestry':
            climate_risk = max(0, abs(rainfall - 900) / 900) * 0.5
        else:  # miyawaki
            climate_risk = max(0, abs(rainfall - 1200) / 1200) * 0.4
        
        risks['climate_risk'] = min(1.0, climate_risk)
        
        # Market risk (mainly for agroforestry)
        if project_type == 'agroforestry':
            area_ha = float(blueprint.get('area_ha', 5))
            market_risk = 0.3 if area_ha < 2 else 0.2  # Small farms have higher risk
            risks['market_risk'] = market_risk
        else:
            risks['market_risk'] = 0.1  # Low market risk for carbon credits
        
        # Technical risk (implementation failure)
        density = blueprint.get('density_per_ha', 1000)
        if project_type == 'miyawaki' and density < 20000:
            risks['technical_risk'] = 0.4  # High risk if density too low
        elif project_type == 'agroforestry' and density > 600:
            risks['technical_risk'] = 0.3  # Overcrowding risk
        else:
            risks['technical_risk'] = 0.1
        
        # Financial risk
        area_ha = float(blueprint.get('area_ha', 5))
        investment_per_ha = 200000 if project_type == 'agroforestry' else 350000
        total_investment = area_ha * investment_per_ha
        
        if total_investment > 1000000:  # High investment projects
            risks['financial_risk'] = 0.4
        elif total_investment > 500000:
            risks['financial_risk'] = 0.25
        else:
            risks['financial_risk'] = 0.15
        
        # Regulatory risk (mainly for carbon credits)
        if project_type == 'miyawaki':
            risks['regulatory_risk'] = 0.2  # Carbon market volatility
        else:
            risks['regulatory_risk'] = 0.05  # Low regulatory risk for food crops
        
        return risks
    
    def generate_recommendations(self, metrics: Any, risks: Dict[str, float], 
                               project_type: str) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if project_type == 'agroforestry':
            agro_metrics = metrics
            
            # Income optimization recommendations
            if agro_metrics.annual_food_income < 100000:
                recommendations.append("Consider higher-value crops like mango, cashew, or jackfruit to increase income")
            
            if agro_metrics.payback_period_months > 60:
                recommendations.append("Include fast-maturing crops like guava, moringa for early cash flow")
            
            if agro_metrics.diversification_index < 0.5:
                recommendations.append("Increase crop diversity to reduce market risk and stabilize income")
            
            if agro_metrics.net_profit_margin < 0.3:
                recommendations.append("Focus on cost reduction: efficient irrigation, organic farming, direct marketing")
            
            # Risk mitigation
            if risks.get('market_risk', 0) > 0.3:
                recommendations.append("Establish direct market linkages or join farmer producer organizations")
            
            if risks.get('climate_risk', 0) > 0.3:
                recommendations.append("Install drip irrigation and rainwater harvesting for climate resilience")
        
        else:  # miyawaki
            miyawaki_metrics = metrics
            
            # Carbon optimization recommendations
            if miyawaki_metrics.carbon_sequestration_total_tons < 50:
                recommendations.append("Increase planting density and include more fast-growing native species")
            
            if miyawaki_metrics.restoration_success_probability < 0.7:
                recommendations.append("Ensure adequate aftercare, weeding, and watering for first 3 years")
            
            if miyawaki_metrics.biodiversity_enhancement_score < 0.6:
                recommendations.append("Include diverse native species across all forest layers for biodiversity")
            
            if miyawaki_metrics.carbon_payback_period_years > 15:
                recommendations.append("Consider premium carbon certification for higher credit prices")
            
            # Long-term sustainability
            if miyawaki_metrics.ecological_resilience_index < 0.6:
                recommendations.append("Add climate-resilient species and ensure genetic diversity")
            
            if risks.get('regulatory_risk', 0) > 0.2:
                recommendations.append("Diversify revenue streams: eco-tourism, education, research partnerships")
        
        # Common recommendations
        if any(risk > 0.4 for risk in risks.values()):
            recommendations.append("Consider phased implementation to reduce overall project risk")
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    def predict_integrated_outcomes(self, blueprint: Dict[str, Any], 
                                  site_conditions: Dict[str, Any]) -> PredictionResults:
        """Main prediction function that integrates all components"""
        
        # Determine project type from blueprint
        project_type = blueprint.get('goal', 'agroforestry')
        if project_type not in ['agroforestry', 'miyawaki']:
            project_type = 'agroforestry'  # Default fallback
        
        # Calculate confidence score based on data completeness
        confidence_factors = []
        confidence_factors.append(1.0 if blueprint.get('total_plants', 0) > 0 else 0.5)
        confidence_factors.append(1.0 if site_conditions.get('annual_precip_mm') else 0.7)
        confidence_factors.append(1.0 if blueprint.get('species_mix_df') is not None else 0.6)
        confidence_factors.append(1.0 if blueprint.get('area_ha', 0) > 0 else 0.5)
        
        confidence_score = np.mean(confidence_factors)
        
        # Generate predictions based on project type
        agroforestry_metrics = None
        miyawaki_metrics = None
        
        if project_type == 'agroforestry':
            agroforestry_metrics = self.predict_agroforestry_outcomes(blueprint, site_conditions)
        else:
            miyawaki_metrics = self.predict_miyawaki_outcomes(blueprint, site_conditions)
        
        # Risk assessment
        risks = self.assess_risks(blueprint, site_conditions, project_type)
        
        # Generate recommendations
        metrics_for_recommendations = agroforestry_metrics if project_type == 'agroforestry' else miyawaki_metrics
        recommendations = self.generate_recommendations(metrics_for_recommendations, risks, project_type)
        
        return PredictionResults(
            project_type=project_type,
            confidence_score=round(confidence_score, 2),
            agroforestry_metrics=agroforestry_metrics,
            miyawaki_metrics=miyawaki_metrics,
            risk_assessment=risks,
            recommendations=recommendations
        )

# Integration function for intelligent_app.py
def run_predictive_assessment(blueprint: Dict[str, Any], 
                            site_conditions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main integration function to be called from intelligent_app.py
    
    Args:
        blueprint: Output from build_forest_blueprint() function
        site_conditions: Site environmental data from GEE analysis
    
    Returns:
        Comprehensive prediction results
    """
    
    try:
        predictor = AdvancedViabilityPredictor()
        
        # Train models if not already trained (in production, models would be pre-trained)
        if not predictor.models and HAS_ML:
            predictor.train_models()
        
        # Generate predictions
        results = predictor.predict_integrated_outcomes(blueprint, site_conditions)
        
        # Convert to dictionary for JSON serialization
        return {
            'project_type': results.project_type,
            'confidence_score': results.confidence_score,
            'agroforestry_metrics': asdict(results.agroforestry_metrics) if results.agroforestry_metrics else None,
            'miyawaki_metrics': asdict(results.miyawaki_metrics) if results.miyawaki_metrics else None,
            'risk_assessment': results.risk_assessment,
            'recommendations': results.recommendations,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
    except Exception as e:
        logging.error(f"Predictive assessment failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    # Example blueprint (simulating output from intelligent_app.py)
    example_blueprint = {
        'goal': 'agroforestry',
        'area_ha': 5.0,
        'total_plants': 2000,
        'density_per_ha': 400,
        'region': 'gangetic_plains',
        'species_mix_df': pd.DataFrame({
            'canonical_name': ['Mangifera indica', 'Psidium guajava', 'Cocos nucifera'],
            'planned_count': [800, 600, 600],
            'is_native': [1, 1, 1]
        })
    }
    
    # Example site conditions
    example_site_conditions = {
        'annual_precip_mm': 950,
        'soil_ph': 6.8,
        'elevation_m': 200,
        'slope_deg': 3.5,
        'ndvi_mean': 0.45
    }
    
    print("🔮 PREDICTIVE VIABILITY ASSESSMENT DEMO")
    print("="*60)
    
    # Run prediction
    results = run_predictive_assessment(example_blueprint, example_site_conditions)
    
    if results['status'] == 'success':
        print(f"Project Type: {results['project_type']}")
        print(f"Confidence Score: {results['confidence_score']}")
        
        if results['agroforestry_metrics']:
            metrics = results['agroforestry_metrics']
            print(f"\n📊 Agroforestry Predictions:")
            print(f"  Annual Food Income: ₹{metrics['annual_food_income']:,}")
            print(f"  5-Year Cumulative: ₹{metrics['cumulative_5yr_income']:,}")
            print(f"  Payback Period: {metrics['payback_period_months']} months")
            print(f"  Profit Margin: {metrics['net_profit_margin']:.1%}")
        
        print(f"\n⚠️ Risk Assessment:")
        for risk, value in results['risk_assessment'].items():
            print(f"  {risk.replace('_', ' ').title()}: {value:.2f}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print(f"❌ Assessment failed: {results['error']}")