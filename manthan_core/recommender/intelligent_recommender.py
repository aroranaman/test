# manthan_core/recommender/intelligent_recommender.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class IntelligentRecommender:
    """Uses a FAISS index and semantic embeddings for smart species recommendations."""

    def __init__(self, embeddings_dir: Path):
        self.embeddings_dir = embeddings_dir
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Load the FAISS index and the metadata
        self.index = faiss.read_index(str(self.embeddings_dir / "faiss.index"))
        with open(self.embeddings_dir / "items.json", 'r') as f:
            self.metadata = json.load(f)
        
        self.vectors = np.load(self.embeddings_dir / "embeddings.npy")

    def recommend_species(self, query_text: str, k: int = 10) -> list[str]:
        """Finds the top k most similar species for a given text query."""
        # Convert the user's query into a numerical fingerprint
        query_vector = self.model.encode([query_text], normalize_embeddings=True)
        
        # Search the FAISS index for the most similar species
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        
        # Return the names of the recommended species
        recommended_species = [self.metadata[i]['name'] for i in indices[0]]
        return recommended_species