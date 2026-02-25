"""Embedding service for query and document embedding.

In production, this would use actual embedding models like:
- OpenAI's text-embedding-ada-002
- Sentence Transformers
- Custom fine-tuned models
"""

import logging
from typing import List, Optional, Dict, Any
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and caching embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding service.
        
        In production, this would load actual models.
        For now, we'll use a deterministic function that simulates embeddings.
        """
        self.model_name = model_name
        self.dimension = 8  # Match our dummy embeddings
        self.cache = {}
        logger.info(f"Initialized embedding service with model: {model_name}")
    
    def _simulate_embedding(self, text: str) -> List[float]:
        """Simulate embedding generation.
        
        In production, replace this with actual model inference.
        This uses a simple hash-based approach for demonstration.
        """
        import hashlib
        
        # Create a deterministic but seemingly random embedding
        hash_obj = hashlib.sha256(text.encode())
        hex_digest = hash_obj.hexdigest()
        
        # Convert hash to numbers between -1 and 1
        embedding = []
        for i in range(self.dimension):
            # Take chunks of the hash and convert to float
            chunk = hex_digest[i*8:(i+1)*8]
            if chunk:
                # Convert to float between -1 and 1
                val = int(chunk, 16) / (2**32)  # Normalize to 0-1
                val = val * 2 - 1  # Shift to -1 to 1
                embedding.append(val)
            else:
                embedding.append(0.0)
        
        return embedding
    
    @lru_cache(maxsize=1000)
    def embed(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for text with caching."""
        if use_cache and text in self.cache:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return self.cache[text]
        
        # In production: response = self.model.encode(text)
        embedding = self._simulate_embedding(text)
        
        if use_cache:
            self.cache[text] = embedding
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        return [self.embed(text) for text in texts]
    
    def compute_similarity_matrix(self, embeddings1: List[List[float]], 
                                  embeddings2: List[List[float]]) -> np.ndarray:
        """Compute similarity matrix between two sets of embeddings."""
        e1 = np.array(embeddings1)
        e2 = np.array(embeddings2)
        
        # Normalize
        e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
        
        return np.dot(e1, e2.T)


# Global embedding service
embedding_service = EmbeddingService()
