# add_water_traits_to_database.py
"""
Script to add water-related traits to your existing Manthan database.
This prepares the database for the enhanced water management system.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def add_water_traits_to_database(db_path="data/manthan.db"):
    """
    Add water-related traits to species in the database based on existing environmental data
    """
    
    print("💧 ADDING WATER MANAGEMENT TRAITS TO MANTHAN DATABASE")
    print("="*65)
    
    conn = sqlite3.connect(db_path)
    
    # Get species with existing environmental traits
    query = """
    SELECT DISTINCT 
        s.species_key,
        s.canonical_name,
        s.family,
        MAX(CASE WHEN st.trait_name = 'drought_tolerance' THEN st.trait_value END) as drought_tolerance,
        MAX(CASE WHEN st.trait_name = 'min_rainfall_mm' THEN CAST(st.trait_value AS REAL) END) as min_rainfall,
        MAX(CASE WHEN st.trait_name = 'max_rainfall_mm' THEN CAST(st.trait_value AS REAL) END) as max_rainfall,
        MAX(CASE WHEN st.trait_name = 'optimal_rainfall_mm' THEN CAST(st.trait_value AS REAL) END) as optimal_rainfall
    FROM species s
    JOIN species_traits st ON s.species_key = st.species_key
    WHERE st.trait_name IN ('drought_tolerance', 'min_rainfall_mm', 'max_rainfall_mm', 'optimal_rainfall_mm')
    GROUP BY s.species_key, s.canonical_name, s.family
    HAVING COUNT(DISTINCT st.trait_name) >= 3
    """
    
    species_df = pd.read_sql_query(query, conn)
    print(f"📊 Found {len(species_df)} species with environmental traits")
    
    # Mapping drought tolerance and rainfall to water need categories
    water_need_map = {}
    
    for _, row in species_df.iterrows():
        species_key = row['species_key']
        drought_tol = row['drought_tolerance']
        min_rain = row['min_rainfall'] or 0
        max_rain = row['max_rainfall'] or 2000
        opt_rain = row['optimal_rainfall'] or (min_rain + max_rain) / 2
        
        # Determine water need category
        if drought_tol == 'High' and opt_rain < 800:
            water_need = 'Low'  # Drought-tolerant, low rainfall preference
        elif drought_tol == 'High' and opt_rain < 1200:
            water_need = 'Moderate'  # Drought-tolerant but moderate rainfall
        elif opt_rain > 2000:
            water_need = 'High'  # High rainfall preference
        elif opt_rain > 1500:
            water_need = 'Moderate'  # Moderate-high rainfall
        elif opt_rain < 600:
            water_need = 'Low'  # Low rainfall adapted
        else:
            water_need = 'Moderate'  # Default moderate
        
        water_need_map[species_key] = water_need
    
    # Create water-related trait records
    water_trait_records = []
    
    for species_key, water_need in water_need_map.items():
        water_trait_records.append({
            'species_key': species_key,
            'trait_name': 'water_need',
            'trait_value': water_need,
            'source': 'inferred_from_drought_rainfall'
        })
    
    print(f"🌊 Prepared {len(water_trait_records)} water need trait records")
    
    # Add family-based water management traits
    family_water_traits = {
        'Fabaceae': 'Moderate',      # Legumes - generally moderate water needs
        'Moraceae': 'Moderate',      # Fig family - moderate water needs  
        'Anacardiaceae': 'Low',      # Mango family - often drought-tolerant
        'Euphorbiaceae': 'Low',      # Spurge family - often low water needs
        'Combretaceae': 'Moderate',  # Terminalia family - moderate needs
        'Meliaceae': 'Low',          # Neem family - drought-tolerant
        'Rutaceae': 'Low',           # Citrus family - often drought-tolerant
        'Poaceae': 'Low',            # Grass family - often drought-adapted
        'Malvaceae': 'Moderate',     # Mallow family - moderate needs
    }
    
    # Get species without water traits but with family info
    family_query = """
    SELECT DISTINCT s.species_key, s.family
    FROM species s
    LEFT JOIN species_traits st ON s.species_key = st.species_key 
        AND st.trait_name = 'water_need'
    WHERE st.species_key IS NULL
      AND s.family IS NOT NULL
    """
    
    family_df = pd.read_sql_query(family_query, conn)
    print(f"👨‍👩‍👧‍👦 Found {len(family_df)} species without water traits, adding family-based estimates")
    
    for _, row in family_df.iterrows():
        family = row['family']
        if family in family_water_traits:
            water_trait_records.append({
                'species_key': row['species_key'],
                'trait_name': 'water_need',
                'trait_value': family_water_traits[family],
                'source': 'family_pattern'
            })
    
    # Insert water traits into database
    if water_trait_records:
        cursor = conn.cursor()
        
        insert_sql = """
        INSERT OR REPLACE INTO species_traits (species_key, trait_name, trait_value, source)
        VALUES (?, ?, ?, ?)
        """
        
        insert_data = [
            (record['species_key'], record['trait_name'], record['trait_value'], record['source'])
            for record in water_trait_records
        ]
        
        try:
            cursor.executemany(insert_sql, insert_data)
            conn.commit()
            print(f"✅ Successfully added {len(insert_data)} water trait records")
        except Exception as e:
            conn.rollback()
            print(f"❌ Failed to insert water traits: {e}")
    
    # Verify the additions
    verification_query = """
    SELECT trait_value, COUNT(*) as count
    FROM species_traits
    WHERE trait_name = 'water_need'
    GROUP BY trait_value
    ORDER BY count DESC
    """
    
    verification_df = pd.read_sql_query(verification_query, conn)
    
    print(f"\n📊 Water Need Distribution:")
    total_species_with_water_traits = verification_df['count'].sum()
    for _, row in verification_df.iterrows():
        percentage = (row['count'] / total_species_with_water_traits) * 100
        print(f"  {row['trait_value']}: {row['count']:,} species ({percentage:.1f}%)")
    
    print(f"\n🎉 WATER TRAITS INTEGRATION COMPLETE!")
    print(f"Total species with water traits: {total_species_with_water_traits:,}")
    
    conn.close()
    
    return True

def test_water_integration():
    """Test the water management system with sample data"""
    
    print("\n🧪 TESTING WATER MANAGEMENT INTEGRATION")
    print("="*50)
    
    try:
        from enhanced_water_management_system import EnhancedWaterManagementSystem
        
        water_system = EnhancedWaterManagementSystem()
        
        # Test site conditions
        site_conditions = {
            'annual_precip_mm': 650,
            'slope_deg': 8.0,
            'ndvi_mean': 0.4,
            'elevation_m': 200,
            'soil_ph': 7.2
        }
        
        # Sample species list
        species_list = [
            'Azadirachta indica',
            'Acacia nilotica', 
            'Prosopis cineraria',
            'Terminalia arjuna',
            'Dalbergia sissoo'
        ]
        
        print(f"🌱 Testing with species: {', '.join(species_list)}")
        print(f"🌍 Site rainfall: {site_conditions['annual_precip_mm']}mm")
        
        # Water balance analysis
        balance = water_system.calculate_comprehensive_water_balance(
            species_list, site_conditions, planting_density=400
        )
        
        print(f"\n💧 Water Balance Results:")
        print(f"  Effective rainfall: {balance['effective_rainfall_mm']:.0f}mm")
        print(f"  Average species demand: {balance['average_species_demand_mm']:.0f}mm")
        print(f"  Water efficiency ratio: {balance['water_efficiency_ratio']:.2f}")
        print(f"  Water stress level: {balance['water_stress_level']}")
        
        if balance['deficit_per_hectare_m3'] > 0:
            print(f"  Water deficit: {balance['deficit_per_hectare_m3']:.0f} m³/ha/year")
            
            # Test water harvesting recommendations
            harvesting = water_system.design_water_harvesting_system(
                site_conditions, 5.0, balance['deficit_per_hectare_m3'] * 5
            )
            
            print(f"\n🏗️ Water Harvesting Recommendations:")
            for i, rec in enumerate(harvesting, 1):
                print(f"  {i}. {rec.structure_type}: {rec.capacity_m3:.0f} m³, ₹{rec.cost_estimate_inr:,.0f}")
        
        # Test water efficiency scoring
        efficiency_scores = water_system.calculate_water_efficiency_scores(
            species_list, site_conditions['annual_precip_mm']
        )
        
        print(f"\n🌊 Species Water Efficiency:")
        for species, score in sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {species}: {score:.2f}")
        
        print(f"\n✅ Water management integration test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Could not import water management system: {e}")
    except Exception as e:
        print(f"❌ Water management test failed: {e}")

def main():
    """Main function"""
    
    print("🌊 MANTHAN WATER MANAGEMENT INTEGRATION")
    print("Adding comprehensive water management to your restoration system")
    print("="*80)
    
    db_path = "data/manthan.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        print("Please ensure your manthan.db exists first.")
        return
    
    # Step 1: Add water traits to database
    print("\n⚡ Step 1: Adding water management traits to database...")
    
    try:
        success = add_water_traits_to_database(db_path)
        if not success:
            print("❌ Failed to add water traits")
            return
    except Exception as e:
        print(f"❌ Error adding water traits: {e}")
        return
    
    # Step 2: Test water management integration
    print("\n⚡ Step 2: Testing water management system...")
    test_water_integration()
    
    print(f"\n🎉 WATER INTEGRATION COMPLETE!")
    print(f"Your Manthan system now includes:")
    print(f"  💧 Water efficiency scoring for species selection")
    print(f"  🌊 Comprehensive water balance analysis")
    print(f"  🏗️ Water harvesting structure recommendations") 
    print(f"  📊 Seasonal water demand forecasting")
    print(f"  🚰 Irrigation requirement calculations")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Run: streamlit run water_integrated_intelligent_app.py")
    print(f"2. Test with different water priority levels (Low/Balanced/High)")
    print(f"3. Explore water harvesting recommendations for your sites")

if __name__ == "__main__":
    main()