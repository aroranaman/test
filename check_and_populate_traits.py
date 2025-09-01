# check_and_populate_traits.py
"""
Ultra-fast trait population checker and runner.
This script will analyze your existing database and populate missing quantitative traits
in under 2 minutes using lookup tables instead of slow GEE calls.
"""

import sqlite3
import pandas as pd
from pathlib import Path

def quick_database_analysis(db_path="data/manthan.db"):
    """Quick analysis of what's in your database"""
    
    print("🔍 QUICK DATABASE ANALYSIS")
    print("="*50)
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Basic counts
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM species")
        species_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM occurrences")
        occurrence_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM species_traits")
        traits_count = cursor.fetchone()[0]
        
        print(f"📊 Species: {species_count:,}")
        print(f"📍 Occurrences: {occurrence_count:,}")
        print(f"🔬 Total traits: {traits_count:,}")
        
        # Check existing trait types
        cursor.execute("""
            SELECT trait_name, COUNT(*) as count 
            FROM species_traits 
            GROUP BY trait_name 
            ORDER BY count DESC
            LIMIT 10
        """)
        
        traits = cursor.fetchall()
        print(f"\n📋 Current trait types:")
        for trait_name, count in traits:
            print(f"  {trait_name}: {count:,} species")
        
        # Check for quantitative environmental traits
        quantitative_traits = ['min_rainfall_mm', 'max_rainfall_mm', 'min_ph', 'max_ph', 'min_temp_c', 'max_temp_c']
        
        cursor.execute(f"""
            SELECT COUNT(DISTINCT species_key) 
            FROM species_traits 
            WHERE trait_name IN ({','.join(['?' for _ in quantitative_traits])})
        """, quantitative_traits)
        
        quant_species = cursor.fetchone()[0]
        
        print(f"\n🌡️ Species with quantitative environmental traits: {quant_species}")
        print(f"🔧 Species needing enhancement: {species_count - quant_species}")
        
        # Show sample species that need traits
        cursor.execute("""
            SELECT s.canonical_name, s.family 
            FROM species s
            LEFT JOIN species_traits st ON s.species_key = st.species_key 
                AND st.trait_name = 'min_rainfall_mm'
            WHERE st.trait_value IS NULL
            LIMIT 5
        """)
        
        sample_species = cursor.fetchall()
        if sample_species:
            print(f"\n📋 Sample species needing quantitative traits:")
            for name, family in sample_species:
                print(f"  • {name} ({family})")
        
        enhancement_needed = (species_count - quant_species) > 0
        
        if enhancement_needed:
            print(f"\n✅ READY FOR FAST ENHANCEMENT!")
            print(f"💨 Fast population will add environmental traits in < 2 minutes")
        else:
            print(f"\n✅ All species already have quantitative traits!")
        
        return enhancement_needed
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False
    finally:
        conn.close()

def show_example_output():
    """Show what the enhancement will add"""
    
    print(f"\n📋 ENHANCEMENT PREVIEW")
    print(f"The fast population will add these quantitative traits:")
    print(f"")
    print(f"🌡️ Temperature Ranges:")
    print(f"  • min_temp_c, max_temp_c, optimal_temp_c")
    print(f"")  
    print(f"🌧️ Rainfall Ranges:")
    print(f"  • min_rainfall_mm, max_rainfall_mm, optimal_rainfall_mm")
    print(f"")
    print(f"🧪 Soil pH Ranges:")
    print(f"  • min_ph, max_ph, optimal_ph")
    print(f"")
    print(f"⛰️ Elevation Ranges:")
    print(f"  • min_elevation_m, max_elevation_m")
    print(f"")
    print(f"🏜️ Derived Traits:")
    print(f"  • drought_tolerance (High/Moderate/Low)")
    print(f"  • soil_type_preference (Acidic/Neutral/Alkaline)")
    print(f"")
    print(f"Example for Azadirachta indica (Neem):")
    print(f"  min_rainfall_mm: 400")
    print(f"  max_rainfall_mm: 1200") 
    print(f"  optimal_rainfall_mm: 800")
    print(f"  min_temp_c: 18")
    print(f"  max_temp_c: 38")
    print(f"  min_ph: 6.2")
    print(f"  max_ph: 8.5")
    print(f"  drought_tolerance: High")

def main():
    """Main function"""
    
    print("⚡ MANTHAN FAST TRAIT ENHANCEMENT")
    print("Ultra-fast quantitative trait population (< 2 minutes)")
    print("="*70)
    
    # Analyze database
    needs_enhancement = quick_database_analysis()
    
    if not needs_enhancement:
        print(f"\n🎉 Your database is already ready for intelligent recommendations!")
        print(f"You can run your intelligent_app.py now.")
        return
    
    # Show what will be added
    show_example_output()
    
    print(f"\n🚀 FAST ENHANCEMENT OPTIONS:")
    print(f"1. Ultra-fast lookup method (< 2 minutes)")
    print(f"   Uses known species data + family patterns + regional inference")
    print(f"   ✅ Recommended for immediate use")
    print(f"")
    print(f"2. Slow GEE method (30-60 minutes)")  
    print(f"   Uses Google Earth Engine for precise environmental data")
    print(f"   🐌 More accurate but very slow")
    
    choice = input(f"\n🤔 Choose method (1 for fast, 2 for slow, q to quit): ").strip()
    
    if choice == '1':
        print(f"\n🏃‍♂️ Running fast trait population...")
        try:
            from fast_trait_population import main as run_fast_population
            run_fast_population()
        except ImportError:
            print(f"❌ fast_trait_population.py not found in current directory")
            print(f"Please ensure the file is available and try again.")
    
    elif choice == '2':
        print(f"\n🐌 Running slow but accurate GEE method...")
        try:
            from run_climate_envelope_enhancement import main as run_slow_population
            run_slow_population()
        except ImportError:
            print(f"❌ run_climate_envelope_enhancement.py not found")
            print(f"Please ensure the GEE enhancement files are available.")
    
    elif choice.lower() == 'q':
        print(f"👋 Exiting without changes.")
    
    else:
        print(f"❌ Invalid choice. Please run again and choose 1, 2, or q.")

if __name__ == "__main__":
    main()