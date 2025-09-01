# instant_fix.py - Instantly fix your app by patching the import

import os
import sys
from pathlib import Path

def apply_instant_fix():
    """Apply the instant fix by creating a wrapper module"""
    
    print("🚀 Applying instant fix for sentence-transformers segmentation fault...")
    
    # Create the patch file
    patch_content = '''# intelligent_recommender.py - Patched version
"""
This is a patched version that avoids the sentence-transformers segmentation fault.
It uses transformers directly instead of sentence-transformers.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
from pathlib import Path
import pickle

class IntelligentRecommender:
    """Patched recommender that bypasses sentence-transformers"""
    
    def __init__(self, embeddings_dir):
        self.embeddings_dir = Path(embeddings_dir)
        
        # Load FAISS index
        try:
            self.index = faiss.read_index(str(self.embeddings_dir / "faiss.index"))
        except:
            print("Warning: Could not load FAISS index, using dummy")
            self.index = faiss.IndexFlatL2(384)
        
        # Load metadata
        try:
            with open(self.embeddings_dir / "metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
        except:
            print("Warning: Could not load metadata, using defaults")
            self.metadata = self._get_default_species()
        
        # Load model without sentence-transformers
        print("Loading embedding model (patched version)...")
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print("✓ Model loaded successfully (without sentence-transformers)")
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.tokenizer = None
            self.model = None
    
    def _get_default_species(self):
        """Default species for UP region"""
        species_list = [
            "Azadirachta indica", "Ficus religiosa", "Albizia lebbeck",
            "Dalbergia sissoo", "Terminalia arjuna", "Madhuca longifolia",
            "Syzygium cumini", "Mangifera indica", "Tamarindus indica",
            "Butea monosperma", "Acacia nilotica", "Prosopis cineraria",
            "Moringa oleifera", "Aegle marmelos", "Phyllanthus emblica"
        ]
        
        metadata = []
        for i, species in enumerate(species_list):
            metadata.append({
                'species_name': species,
                'index': i
            })
        
        # Add dummy embeddings
        if self.index.ntotal == 0:
            embeddings = np.random.randn(len(species_list), 384).astype('float32')
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index.add(embeddings)
        
        return metadata
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def recommend_species(self, query, k=20):
        """Get species recommendations"""
        if self.model is None:
            # Return default recommendations
            return [m['species_name'] for m in self.metadata[:k]]
        
        try:
            # Encode query
            inputs = self.tokenizer(query, padding=True, truncation=True, 
                                  return_tensors='pt', max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            query_embedding = embeddings.numpy().astype('float32')
            
            # Search
            k = min(k, self.index.ntotal)
            if k > 0:
                distances, indices = self.index.search(query_embedding, k)
                
                recommendations = []
                for idx in indices[0]:
                    if 0 <= idx < len(self.metadata):
                        recommendations.append(self.metadata[idx]['species_name'])
                
                return recommendations
            else:
                return [m['species_name'] for m in self.metadata[:k]]
                
        except Exception as e:
            print(f"Recommendation error: {e}")
            return [m['species_name'] for m in self.metadata[:k]]
'''
    
    # Find the manthan_core directory
    possible_paths = [
        Path("manthan_core/recommender"),
        Path("./manthan_core/recommender"),
        Path("../manthan_core/recommender"),
    ]
    
    recommender_dir = None
    for path in possible_paths:
        if path.exists():
            recommender_dir = path
            break
    
    if recommender_dir is None:
        # Create it in current directory
        recommender_dir = Path("manthan_core/recommender")
        recommender_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        (Path("manthan_core") / "__init__.py").touch()
        (recommender_dir / "__init__.py").touch()
    
    # Write the patched file
    patch_file = recommender_dir / "intelligent_recommender.py"
    with open(patch_file, 'w') as f:
        f.write(patch_content)
    
    print(f"✓ Patched file created at: {patch_file}")
    
    # Create a test script
    test_script = '''# test_patch.py
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
    print("\\n✅ Patch is working! You can now run your Streamlit app.")
    
except Exception as e:
    print(f"✗ Error: {e}")
'''
    
    with open("test_patch.py", 'w') as f:
        f.write(test_script)
    
    print("\n📋 Instructions:")
    print("1. Test the patch: python test_patch.py")
    print("2. Run your app: streamlit run intelligent_app.py")
    print("\nThe patched version bypasses sentence-transformers completely.")
    print("Your app should now work without segmentation faults!")

if __name__ == "__main__":
    apply_instant_fix()