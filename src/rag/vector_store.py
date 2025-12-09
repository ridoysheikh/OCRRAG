"""
Vector Store Module
ChromaDB-based storage for document embeddings with metadata.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

from .embeddings import EmbeddingModel, chunk_text

load_dotenv()


@dataclass
class SearchResult:
    """Represents a search result with citation info."""
    text: str
    filename: str
    page_number: int
    chunk_index: int
    score: float
    
    def citation(self) -> str:
        """Format as citation."""
        snippet = self.text[:100] + "..." if len(self.text) > 100 else self.text
        return f'[Source: {self.filename}, Page {self.page_number}, "{snippet}"]'


class VectorStore:
    """ChromaDB vector store for document chunks."""
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "documents"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_dir: Directory to persist ChromaDB data.
            collection_name: Name of the collection.
        """
        persist_dir = persist_dir or os.getenv('CHROMA_PERSIST_DIR', './data/chroma')
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = EmbeddingModel()
    
    def add_document(
        self,
        filename: str,
        pages: list[dict],  # [{"page_number": int, "text": str}]
        chunk_size: int = 500,
        overlap: int = 50
    ) -> int:
        """
        Add a document to the vector store.
        
        Args:
            filename: Name of the source file.
            pages: List of page dicts with page_number and text.
            chunk_size: Size of text chunks.
            overlap: Overlap between chunks.
            
        Returns:
            Number of chunks added.
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for page in pages:
            page_num = page["page_number"]
            text = page["text"]
            
            chunks = chunk_text(text, chunk_size, overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}__p{page_num}__c{chunk_idx}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "text": chunk  # Store full text for retrieval
                })
                all_ids.append(chunk_id)
        
        if all_chunks:
            embeddings = self.embedding_model.embed(all_chunks).tolist()
            
            self.collection.add(
                ids=all_ids,
                embeddings=embeddings,
                metadatas=all_metadatas,
                documents=all_chunks
            )
        
        return len(all_chunks)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filename_filter: Optional[str] = None
    ) -> list[SearchResult]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query.
            n_results: Number of results to return.
            filename_filter: Optional filename to filter by.
            
        Returns:
            List of SearchResult objects.
        """
        query_embedding = self.embedding_model.embed_single(query).tolist()
        
        where_filter = None
        if filename_filter:
            where_filter = {"filename": filename_filter}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (cosine)
                score = 1 - distance
                
                search_results.append(SearchResult(
                    text=doc,
                    filename=metadata['filename'],
                    page_number=metadata['page_number'],
                    chunk_index=metadata['chunk_index'],
                    score=score
                ))
        
        return search_results
    
    def delete_document(self, filename: str) -> int:
        """Delete all chunks for a document."""
        # Get all IDs for this document
        results = self.collection.get(
            where={"filename": filename},
            include=[]
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        
        return 0
    
    def list_documents(self) -> list[str]:
        """List all unique document filenames."""
        results = self.collection.get(include=["metadatas"])
        
        filenames = set()
        for metadata in results['metadatas']:
            filenames.add(metadata['filename'])
        
        return sorted(list(filenames))
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "total_chunks": self.collection.count(),
            "documents": self.list_documents()
        }
