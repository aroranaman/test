# quick_species_test.py
"""
Quick test to see if we can find compatible species in your database
using the site conditions from your app (612mm rain, pH 8.25)
"""

import sqlite3
import pandas as pd

def test_direct_environmental_matching():
    """Test direct environmental matching with your site conditions"""
    
    print("🧪 TESTING DIRECT ENVIRONMENTAL MATCHING")
    print("Site conditions: 612mm rainfall, pH 8.25")
    print("="*60)
    
    conn = sqlite3.connect("data/manthan.db")
    
    # Query for species that can handle your site conditions
    query = """
    SELECT DISTINCT 
        s.canonical_name,
        s.family,
        st_rain_min.trait_value as min_rain,
        st_rain_max.trait_value as max_rain,
        st_ph_min.trait_value as min_ph,
        st_ph_max.trait_value as max_ph,
        st_drought.trait_value as drought_tolerance
    FROM species s
    JOIN species_traits st_rain_min ON s.species_key = st_rain_min.species_key 
        AND st_rain_min.trait_name = 'min_rainfall_mm'
    JOIN species_traits st_rain_max ON s.species_key = st_rain_max.species_key 
        AND st_rain_max.trait_name = 'max_rainfall_mm'  
    JOIN species_traits st_ph_min ON s.species_key = st_ph_min.species_key 
        AND st_ph_min.trait_name = 'min_ph'
    JOIN species_traits st_ph_max ON s.species_key = st_ph_max.species_key 
        AND st_ph_max.trait_name = 'max_ph'
    LEFT JOIN species_traits st_drought ON s.species_key = st_drought.species_key 
        AND st_drought.trait_name = 'drought_tolerance'
    WHERE 
        CAST(st_rain_min.trait_value AS REAL) <= 612
        AND CAST(st_rain_max.trait_value AS REAL) >= 612  
        AND CAST(st_ph_min.trait_value AS REAL) <= 8.25
        AND CAST(st_ph_max.trait_value AS REAL) >= 8.25
    ORDER BY s.canonical_name
    LIMIT 20
    """
    
    try:
        results_df = pd.read_sql_query(query, conn)
        
        print(f"🎯 Found {len(results_df)} compatible species:")
        print()
        
        if len(results_df) > 0:
            for _, row in results_df.iterrows():
                name = row['canonical_name']
                family = row['family'] or 'Unknown'
                min_rain = row['min_rain']
                max_rain = row['max_rain'] 
                min_ph = row['min_ph']
                max_ph = row['max_ph']
                drought = row['drought_tolerance'] or 'Unknown'
                
                print(f"✅ {name} ({family})")
                print(f"   Rainfall: {min_rain}-{max_rain}mm (site: 612mm)")
                print(f"   pH: {min_ph}-{max_ph} (site: 8.25)")
                print(f"   Drought tolerance: {drought}")
                print()
        else:
            print("❌ No compatible species found!")
            print("This suggests an issue with the trait data or query logic.")
            
            # Let's check what's in the database
            print("\n🔍 Checking sample trait data...")
            
            sample_query = """
            SELECT s.canonical_name,
                   st.trait_name,
                   st.trait_value
            FROM species s
            JOIN species_traits st ON s.species_key = st.species_key
            WHERE st.trait_name IN ('min_rainfall_mm', 'max_rainfall_mm', 'min_ph', 'max_ph')
            LIMIT 20
            """
            
            sample_df = pd.read_sql_query(sample_query, conn)
            print(sample_df.to_string())
        
        conn.close()
        return len(results_df) > 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        conn.close()
        return False

def test_semantic_embeddings_simple():
    """Simple test of semantic embeddings"""
    
    print("\n🧠 TESTING SEMANTIC EMBEDDINGS (SIMPLE)")
    print("="*50)
    
    try:
        from pathlib import Path
        import sys
        
        # Try to import the recommender
        sys.path.insert(0, str(Path.cwd()))
        
        try:
            from manthan_core.recommender.intelligent_recommender import IntelligentRecommender
            print("✅ IntelligentRecommender imported successfully")
        except ImportError as e:
            print(f"❌ Could not import IntelligentRecommender: {e}")
            return False
        
        # Check embeddings directory
        embeddings_dir = Path("data/embeddings")
        if not embeddings_dir.exists():
            print(f"❌ Embeddings directory not found: {embeddings_dir}")
            return False
        
        required_files = ["faiss.index", "metadata.pkl"]
        for filename in required_files:
            filepath = embeddings_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"✅ {filename}: {size:,} bytes")
            else:
                print(f"❌ Missing: {filename}")
                return False
        
        # Try to create recommender
        try:
            recommender = IntelligentRecommender(embeddings_dir)
            print("✅ Recommender initialized")
        except Exception as e:
            print(f"❌ Recommender initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test simple query
        try:
            results = recommender.recommend_species("trees", k=10)
            print(f"✅ Test query returned {len(results)} results")
            
            if len(results) > 0:
                print("Sample results:")
                for species in results[:5]:
                    print(f"  • {species}")
                return True
            else:
                print("❌ No results returned from semantic search")
                return False
                
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Semantic test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 QUICK MANTHAN DIAGNOSTICS")
    print("Testing with your exact site conditions")
    print("="*60)
    
    # Test environmental matching
    env_works = test_direct_environmental_matching()
    
    # Test semantic search
    semantic_works = test_semantic_embeddings_simple()
    
    print("\n📊 QUICK DIAGNOSIS RESULTS")
    print("="*40)
    print(f"Environmental filtering: {'✅ WORKS' if env_works else '❌ BROKEN'}")
    print(f"Semantic search: {'✅ WORKS' if semantic_works else '❌ BROKEN'}")
    
    if not semantic_works:
        print("\n🎯 MAIN ISSUE: Semantic search is not working")
        print("This is why you see 'Semantic recommender returned no items'")
        print("\nPossible fixes:")
        print("1. Check that data/embeddings/ directory has the right files")
        print("2. Verify IntelligentRecommender class can be imported")
        print("3. Test the FAISS index is not corrupted")
    
    elif not env_works:
        print("\n🎯 MAIN ISSUE: Environmental filtering is not working")
        print("The quantitative traits may not be in the expected format")
    
    else:
        print("\n🎯 Both systems work - issue is in intelligent_app.py integration")
        print("The app logic may not be connecting the components properly")