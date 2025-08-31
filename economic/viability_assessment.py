# economic/viability_assessment.py
from __future__ import annotations
from typing import List, Dict, Any

class EconomicViabilityAssessor:
    """
    Calculates costs, revenues, and key financial metrics for a restoration project.
    """

    def __init__(self, cost_db: Dict[str, Any], revenue_db: Dict[str, Any]):
        """
        Initializes the assessor with cost and revenue data.
        
        Args:
            cost_db: A dictionary containing cost parameters like site prep, labor, etc.
            revenue_db: A dictionary mapping species names to their revenue data,
                        fetched from the knowledge base.
        """
        self.cost_db = cost_db
        self.revenue_db = revenue_db

    def estimate_lifecycle_costs(self, area_ha: float, species_list: List[str], project_duration_years: int) -> Dict[str, float]:
        """
        Estimates total costs over the project lifecycle.
        """
        # Costs per hectare
        site_prep_cost = self.cost_db.get('site_preparation', 15000)
        
        # Calculate average sapling cost from the provided revenue_db (as a proxy)
        # or a dedicated cost_db entry.
        avg_sapling_price = 50 # Default price
        sapling_costs = self.cost_db.get('sapling_costs', {})
        if sapling_costs:
            avg_sapling_price = sum(sapling_costs.get(s, 50) for s in species_list) / len(species_list) if species_list else 50
        
        saplings_per_ha = self.cost_db.get('saplings_per_ha', 1000)
        planting_labor_cost = self.cost_db.get('planting_labor', 10000)
        maintenance_cost_annual = self.cost_db.get('maintenance_annual', 5000)

        initial_cost = (site_prep_cost + (avg_sapling_price * saplings_per_ha) + planting_labor_cost) * area_ha
        recurring_cost = maintenance_cost_annual * area_ha * project_duration_years
        total_cost = initial_cost + recurring_cost

        return {
            "initial_investment": round(initial_cost, 2),
            "total_maintenance_cost": round(recurring_cost, 2),
            "total_project_cost": round(total_cost, 2)
        }

    def project_revenues(self, area_ha: float, species_list: List[str], project_duration_years: int) -> Dict[str, Any]:
        """
        Projects revenues from timber, NTFPs, and carbon credits.
        """
        revenues: Dict[str, float] = {"timber": 0, "ntfp": 0, "carbon_credits": 0}
        timeline: List[Dict[str, Any]] = []

        for species_name in species_list:
            data = self.revenue_db.get(species_name)
            if data:
                product_type = data.get('type', 'Unknown')
                revenue = data.get('yield_per_ha', 0) * data.get('price', 0) * area_ha

                if product_type == 'Timber':
                    harvest_year = data.get('harvest_year', project_duration_years)
                    if harvest_year <= project_duration_years:
                        revenues['timber'] += revenue
                        timeline.append({"year": harvest_year, "event": f"Timber harvest: {species_name}", "revenue": revenue})
                elif product_type in ('NTFP', 'Fruit', 'Crop'):
                    start_year = data.get('start_year', 3)
                    # Assume annual revenue
                    annual_revenue = revenue / area_ha # Price is often per unit, yield is total units
                    total_ntfp_revenue = annual_revenue * (project_duration_years - start_year) * area_ha
                    revenues['ntfp'] += total_ntfp_revenue
                    timeline.append({"year": start_year, "event": f"NTFP income starts: {species_name}", "annual_revenue": annual_revenue})

        # Carbon credits (simplified model)
        carbon_price = self.cost_db.get('carbon_price_per_ton', 500)
        sequestration_rate = self.cost_db.get('avg_sequestration_per_ha_yr', 4) # in tons
        total_carbon_revenue = sequestration_rate * area_ha * project_duration_years * carbon_price
        revenues['carbon_credits'] = total_carbon_revenue

        total_revenue = sum(revenues.values())
        
        return {
            "total_projected_revenue": round(total_revenue, 2),
            "revenue_breakdown": {k: round(v, 2) for k, v in revenues.items()},
            "revenue_timeline": sorted(timeline, key=lambda x: x['year'])
        }

    def run_assessment(self, area_ha: float, species_list: List[str], project_duration_years: int = 20) -> Dict[str, Any]:
        """
        Runs a full assessment and calculates ROI and other key financial metrics.
        """
        costs = self.estimate_lifecycle_costs(area_ha, species_list, project_duration_years)
        revenues = self.project_revenues(area_ha, species_list, project_duration_years)
        
        total_cost = costs.get('total_project_cost', 0)
        total_revenue = revenues.get('total_projected_revenue', 0)
        
        if total_cost > 0:
            net_profit = total_revenue - total_cost
            roi = (net_profit / total_cost) * 100
        else:
            net_profit = total_revenue
            roi = float('inf') if total_revenue > 0 else 0

        return {
            "costs": costs,
            "revenues": revenues,
            "financial_summary": {
                "net_profit": round(net_profit, 2),
                "return_on_investment_pct": round(roi, 2),
            }
        }