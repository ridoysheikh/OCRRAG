"""
OCR RAG API - FastAPI Web Interface
REST API for PDF OCR and RAG chat.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import shutil
from pathlib import Path
import os
from dotenv import load_dotenv

from .ocr import get_ocr_processor, LocalOCR
from .rag import RAGChat, VectorStore

load_dotenv()

app = FastAPI(
    title="OCR RAG API",
    description="PDF OCR and RAG Chat with Citations",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

vector_store = VectorStore()
rag_chat = RAGChat(vector_store=vector_store)


# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    filename_filter: Optional[str] = None
    n_sources: int = 5
    verify_quotes: bool = True


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    quote_verification: dict
    refused: bool


class UploadResponse(BaseModel):
    filename: str
    pages_extracted: int
    chunks_indexed: int
    message: str


class DocumentInfo(BaseModel):
    filename: str


class StatsResponse(BaseModel):
    total_chunks: int
    documents: list[str]


# Endpoints
@app.get("/")
async def root():
    return {
        "service": "OCR RAG API",
        "version": "0.1.0",
        "endpoints": {
            "POST /upload": "Upload and process PDF",
            "POST /chat": "RAG chat with citations",
            "GET /documents": "List indexed documents",
            "DELETE /documents/{filename}": "Delete a document"
        }
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    use_textract: bool = False
):
    """
    Upload and process a PDF document.
    
    - Extracts text using OCR (Textract or local)
    - Indexes document chunks in vector store
    - Returns processing stats
    """
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff')):
        raise HTTPException(400, "Only PDF and image files are supported")
    
    # Save uploaded file
    filepath = UPLOAD_DIR / file.filename
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # OCR processing
        if use_textract:
            ocr = get_ocr_processor(use_textract=True)
        else:
            ocr = LocalOCR()
        
        result = ocr.extract_from_file(str(filepath))
        
        # Save OCR result
        result.save("./data/processed")
        
        # Index in vector store
        pages = [
            {"page_number": p.page_number, "text": p.text}
            for p in result.pages
        ]
        chunks = vector_store.add_document(
            filename=result.filename,
            pages=pages
        )
        
        return UploadResponse(
            filename=result.filename,
            pages_extracted=result.total_pages,
            chunks_indexed=chunks,
            message="Document processed and indexed successfully"
        )
    
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG chat with citations.
    
    - Retrieves relevant document chunks
    - Generates answer with citations
    - Verifies quotes against sources
    - Refuses if no relevant sources found
    """
    response = rag_chat.chat(
        query=request.query,
        n_sources=request.n_sources,
        verify_quotes=request.verify_quotes,
        filename_filter=request.filename_filter
    )
    
    return ChatResponse(
        answer=response.answer,
        citations=response.citations,
        quote_verification=response.quote_verification,
        refused=response.refused
    )


@app.get("/documents", response_model=StatsResponse)
async def list_documents():
    """List all indexed documents and stats."""
    stats = vector_store.get_stats()
    return StatsResponse(
        total_chunks=stats["total_chunks"],
        documents=stats["documents"]
    )


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from the index."""
    deleted = vector_store.delete_document(filename)
    
    if deleted == 0:
        raise HTTPException(404, f"Document not found: {filename}")
    
    return {
        "message": f"Deleted {deleted} chunks for {filename}",
        "filename": filename
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
