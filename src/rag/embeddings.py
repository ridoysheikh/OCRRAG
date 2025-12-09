"""
Document Embeddings Module
Creates embeddings for document chunks for semantic search.
"""

from sentence_transformers import SentenceTransformer
from typing import Optional
import numpy as np


class EmbeddingModel:
    """Wrapper for embedding models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name or path.
                       Default is a fast, lightweight model.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            NumPy array of embeddings (n_texts, dimension).
        """
        return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Args:
        text: Text to split.
        chunk_size: Target size of each chunk in characters.
        overlap: Number of overlapping characters between chunks.
        
    Returns:
        List of text chunks.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within last 100 chars
            last_period = text.rfind('.', end - 100, end)
            last_newline = text.rfind('\n', end - 100, end)
            break_point = max(last_period, last_newline)
            
            if break_point > start:
                end = break_point + 1
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return [c for c in chunks if c]  # Filter empty chunks
