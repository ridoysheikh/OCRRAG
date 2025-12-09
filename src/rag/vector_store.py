"""
Vector Store Module
PostgreSQL + pgvector storage for document embeddings with metadata.
"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from pgvector.psycopg2 import register_vector
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
    """PostgreSQL + pgvector store for document chunks."""
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        table_name: str = "document_chunks"
    ):
        """
        Initialize vector store.
        
        Args:
            connection_string: PostgreSQL connection string.
            table_name: Name of the table for storing chunks.
        """
        self.connection_string = connection_string or os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:postgres@localhost:5432/ocrrag'
        )
        self.table_name = table_name
        self.embedding_model = EmbeddingModel()
        
        self._init_db()
    
    def _get_connection(self):
        """Get a database connection."""
        conn = psycopg2.connect(self.connection_string)
        register_vector(conn)
        return conn
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create table for document chunks
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        chunk_id VARCHAR(512) UNIQUE NOT NULL,
                        filename VARCHAR(255) NOT NULL,
                        page_number INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector({self.embedding_model.dimension}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for vector similarity search (use HNSW for better performance)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                    ON {self.table_name} 
                    USING hnsw (embedding vector_cosine_ops)
                """)
                
                # Create index for filename filtering
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_filename_idx 
                    ON {self.table_name} (filename)
                """)
                
            conn.commit()
    
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
        all_data = []
        
        for page in pages:
            page_num = page["page_number"]
            text = page["text"]
            
            chunks = chunk_text(text, chunk_size, overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}__p{page_num}__c{chunk_idx}"
                embedding = self.embedding_model.embed_single(chunk)
                
                all_data.append((
                    chunk_id,
                    filename,
                    page_num,
                    chunk_idx,
                    chunk,
                    embedding.tolist()
                ))
        
        if all_data:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {self.table_name} 
                        (chunk_id, filename, page_number, chunk_index, text, embedding)
                        VALUES %s
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding
                        """,
                        all_data,
                        template="(%s, %s, %s, %s, %s, %s::vector)"
                    )
                conn.commit()
        
        return len(all_data)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filename_filter: Optional[str] = None
    ) -> list[SearchResult]:
        """
        Search for relevant document chunks using cosine similarity.
        
        Args:
            query: Search query.
            n_results: Number of results to return.
            filename_filter: Optional filename to filter by.
            
        Returns:
            List of SearchResult objects.
        """
        query_embedding = self.embedding_model.embed_single(query).tolist()
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if filename_filter:
                    cur.execute(f"""
                        SELECT 
                            text,
                            filename,
                            page_number,
                            chunk_index,
                            1 - (embedding <=> %s::vector) as score
                        FROM {self.table_name}
                        WHERE filename = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, filename_filter, query_embedding, n_results))
                else:
                    cur.execute(f"""
                        SELECT 
                            text,
                            filename,
                            page_number,
                            chunk_index,
                            1 - (embedding <=> %s::vector) as score
                        FROM {self.table_name}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, query_embedding, n_results))
                
                rows = cur.fetchall()
        
        return [
            SearchResult(
                text=row['text'],
                filename=row['filename'],
                page_number=row['page_number'],
                chunk_index=row['chunk_index'],
                score=float(row['score'])
            )
            for row in rows
        ]
    
    def delete_document(self, filename: str) -> int:
        """Delete all chunks for a document."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE filename = %s",
                    (filename,)
                )
                deleted = cur.rowcount
            conn.commit()
        
        return deleted
    
    def list_documents(self) -> list[str]:
        """List all unique document filenames."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT DISTINCT filename FROM {self.table_name} ORDER BY filename")
                rows = cur.fetchall()
        
        return [row[0] for row in rows]
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                total_chunks = cur.fetchone()[0]
        
        return {
            "total_chunks": total_chunks,
            "documents": self.list_documents()
        }
