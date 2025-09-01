# test_patch.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

try:
    from manthan_core.recommender.intelligent_recommender import IntelligentRecommender
    print("✓ Import successful!")
    
    # Test it
    recommender = IntelligentRecommender(Path("./data/embeddings"))
    species = recommender.recommend_species("drought tolerant trees", k=5)
    print(f"✓ Got {len(species)} recommendations:", species)
    print("\n✅ Patch is working! You can now run your Streamlit app.")
    
except Exception as e:
    print(f"✗ Error: {e}")
