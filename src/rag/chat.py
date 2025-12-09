"""
RAG Chat Module
Retrieval-Augmented Generation with citations and quote verification.
"""

from openai import OpenAI
from typing import Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

from .vector_store import VectorStore, SearchResult
from ..utils.quote_verify import verify_quotes_in_response, remove_unverified_quotes

load_dotenv()


@dataclass
class ChatResponse:
    """RAG chat response with citations."""
    answer: str
    citations: list[dict]
    sources_used: list[SearchResult]
    quote_verification: dict
    refused: bool = False
    refusal_reason: Optional[str] = None


SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions based ONLY on the provided source documents.

CRITICAL RULES:
1. ONLY use information from the provided sources. Never use external knowledge.
2. If the sources don't contain relevant information, say "I cannot find information about this in the provided documents."
3. Always cite your sources using the format: [Source: filename, Page X]
4. When quoting text, use exact quotes from the sources.
5. Be concise and accurate.

You will receive context from documents in this format:
---
Source: [filename], Page [number]
[text content]
---

Base your answer ONLY on these sources."""


class RAGChat:
    """RAG-based chat with citations and verification."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize RAG chat.
        
        Args:
            vector_store: VectorStore instance for retrieval.
            openai_api_key: OpenAI API key.
            model: OpenAI model to use.
        """
        self.vector_store = vector_store or VectorStore()
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model
    
    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results as context for LLM."""
        context_parts = []
        
        for result in results:
            context_parts.append(
                f"---\nSource: {result.filename}, Page {result.page_number}\n{result.text}\n---"
            )
        
        return "\n\n".join(context_parts)
    
    def _build_sources_for_verification(self, results: list[SearchResult]) -> list[dict]:
        """Build source list for quote verification."""
        return [
            {
                "text": r.text,
                "filename": r.filename,
                "page_number": r.page_number
            }
            for r in results
        ]
    
    def chat(
        self,
        query: str,
        n_sources: int = 5,
        min_relevance: float = 0.3,
        verify_quotes: bool = True,
        filename_filter: Optional[str] = None
    ) -> ChatResponse:
        """
        Answer a query using RAG with citations.
        
        Args:
            query: User's question.
            n_sources: Number of source chunks to retrieve.
            min_relevance: Minimum relevance score to include source.
            verify_quotes: Whether to verify quotes in response.
            filename_filter: Optional filename to restrict search to.
            
        Returns:
            ChatResponse with answer, citations, and verification.
        """
        # Retrieve relevant documents
        results = self.vector_store.search(
            query=query,
            n_results=n_sources,
            filename_filter=filename_filter
        )
        
        # Filter by relevance
        relevant_results = [r for r in results if r.score >= min_relevance]
        
        # If no relevant sources, refuse to answer
        if not relevant_results:
            return ChatResponse(
                answer="I cannot find relevant information in the provided documents to answer this question.",
                citations=[],
                sources_used=[],
                quote_verification={"status": "no_sources"},
                refused=True,
                refusal_reason="No relevant sources found"
            )
        
        # Build context
        context = self._format_context(relevant_results)
        
        # Generate response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context from documents:\n\n{context}\n\nQuestion: {query}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,  # Low temperature for factual responses
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Build citations
        citations = []
        for result in relevant_results:
            citations.append({
                "filename": result.filename,
                "page_number": result.page_number,
                "snippet": result.text[:150] + "..." if len(result.text) > 150 else result.text,
                "relevance_score": result.score
            })
        
        # Verify quotes if enabled
        quote_verification = {"status": "skipped"}
        if verify_quotes:
            sources_for_verify = self._build_sources_for_verification(relevant_results)
            quote_verification = verify_quotes_in_response(answer, sources_for_verify)
            
            # Remove unverified quotes
            if not quote_verification["all_verified"]:
                answer = remove_unverified_quotes(answer, quote_verification["unverified"])
                answer += "\n\n⚠️ Note: Some quoted text could not be verified against sources and was removed."
        
        return ChatResponse(
            answer=answer,
            citations=citations,
            sources_used=relevant_results,
            quote_verification=quote_verification
        )
    
    def add_document_from_ocr(self, ocr_result: dict) -> int:
        """
        Add a document from OCR results to the vector store.
        
        Args:
            ocr_result: Dict from DocumentOCR.to_dict()
            
        Returns:
            Number of chunks added.
        """
        pages = [
            {"page_number": p["page_number"], "text": p["text"]}
            for p in ocr_result["pages"]
        ]
        
        return self.vector_store.add_document(
            filename=ocr_result["filename"],
            pages=pages
        )
