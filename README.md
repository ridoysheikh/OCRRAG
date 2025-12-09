# OCR RAG - PDF Document Intelligence with Citations

A production-ready PDF Retrieval-Augmented Generation (RAG) system featuring AWS Textract OCR,
semantic search with PostgreSQL + pgvector, and verifiable citations with source attribution.

## Overview

This application solves a critical problem in document intelligence: answering questions
about PDF documents while providing verifiable citations. Unlike generic chatbots that
hallucinate information, this system:

1. **Extracts text from any PDF** - Including scanned documents and images using AWS Textract
2. **Indexes content semantically** - Using sentence embeddings stored in PostgreSQL + pgvector
3. **Answers with citations** - Every response includes file name, page number, and exact snippet
4. **Verifies quotes** - Any quoted text is validated against source documents before output
5. **Refuses gracefully** - If no relevant source exists, the system declines to answer

This is ideal for legal document review, compliance research, academic research assistants,
customer support knowledge bases, and any application requiring traceable AI responses.

## Key Features

### 🔍 AWS Textract OCR Integration
- Processes scanned PDFs, photographs of documents, and images
- Extracts text with high accuracy using Amazon's ML-powered OCR
- Preserves page-level metadata for accurate citations
- Fallback to PyPDF2 for text-based PDFs (no AWS costs for digital documents)

### 💬 RAG Chat with Verifiable Citations
- Semantic search retrieves the most relevant document chunks
- Every answer includes structured citations: `[Source: file.pdf, Page X, "snippet..."]`
- Configurable relevance threshold to control answer quality
- Supports filtering queries to specific documents

### ✅ Quote Verification System
- Extracts all quoted text from AI responses
- Validates each quote against source documents using fuzzy matching
- Removes or flags unverified quotes before showing to users
- Prevents AI hallucinations from appearing as cited facts

### 🗄️ PostgreSQL + pgvector Storage
- Production-ready vector database using PostgreSQL
- HNSW indexing for fast approximate nearest neighbor search
- Single database for vectors, metadata, and future user/tenant data
- Easy backup, scaling, and integration with existing infrastructure

## Project Structure

```
OCRRAG/
├── src/
│   ├── ocr/
│   │   ├── __init__.py
│   │   └── textract_ocr.py      # AWS Textract + local PDF extraction
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Sentence transformer embeddings + chunking
│   │   ├── vector_store.py      # PostgreSQL + pgvector operations
│   │   └── chat.py              # RAG chat engine with citation generation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── quote_verify.py      # Quote extraction and verification
│   ├── main.py                  # CLI entry point
│   └── api.py                   # FastAPI REST endpoints
├── data/
│   ├── uploads/                 # Uploaded PDF storage
│   └── processed/               # OCR results as JSON
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore
└── README.md
```

## Technical Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   PDF/Image │────▶│ AWS Textract │────▶│ Page-level Text │
└─────────────┘     │   (or local) │     │   + Metadata    │
                    └──────────────┘     └────────┬────────┘
                                                  │
                    ┌──────────────┐              ▼
                    │  Sentence    │     ┌─────────────────┐
                    │ Transformers │◀────│  Text Chunking  │
                    └──────┬───────┘     │  (500 char +    │
                           │             │   50 overlap)   │
                           ▼             └─────────────────┘
                    ┌──────────────┐
                    │  PostgreSQL  │
                    │  + pgvector  │
                    │  (HNSW idx)  │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ User Query   │   │  Semantic    │   │   OpenAI     │
│              │──▶│   Search     │──▶│   GPT-4o     │
└──────────────┘   │ (top-k docs) │   │  (generate)  │
                   └──────────────┘   └──────┬───────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │    Quote     │
                                      │ Verification │
                                      └──────┬───────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │   Response   │
                                      │ + Citations  │
                                      └──────────────┘
```

## Quick Start

### 1. Set up PostgreSQL with pgvector

```bash
# Using Docker (recommended for development)
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ocrrag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. Install Python Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials:
# - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for Textract)
# - OPENAI_API_KEY (for chat completions)
# - DATABASE_URL (PostgreSQL connection string)
```

### 4. Run the Application

**CLI Mode:**
```bash
# Process a PDF document
python -m src.main process --file /path/to/document.pdf

# Start interactive chat
python -m src.main chat

# Single query
python -m src.main chat --query "What are the key findings?"

# List indexed documents
python -m src.main list
```

**API Mode:**
```bash
uvicorn src.api:app --reload --port 8000
```

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload and process a PDF document |
| POST | `/chat` | Send a query and receive cited answer |
| GET | `/documents` | List all indexed documents |
| DELETE | `/documents/{filename}` | Remove a document from index |
| GET | `/health` | Health check endpoint |

### Example API Usage

```bash
# Upload a document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main conclusion?", "verify_quotes": true}'
```

## Citation Format

All responses include structured citations:

```
Based on the document, the main finding is that sales increased by 25%.
[Source: quarterly_report.pdf, Page 3, "sales growth of 25% compared to..."]
```

If no relevant source is found, the system responds:
```
I cannot find relevant information in the provided documents to answer this question.
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key for Textract | For scanned PDFs |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | For scanned PDFs |
| `AWS_REGION` | AWS region (default: us-east-1) | For scanned PDFs |
| `UPLOAD_DIR` | Directory for uploaded files | No (default: ./data/uploads) |
| `PROCESSED_DIR` | Directory for OCR results | No (default: ./data/processed) |

## Dependencies

- **boto3** - AWS SDK for Textract OCR
- **psycopg2-binary** - PostgreSQL adapter
- **pgvector** - Vector similarity search extension
- **sentence-transformers** - Text embeddings (all-MiniLM-L6-v2)
- **openai** - GPT-4o chat completions
- **fastapi** - REST API framework
- **PyPDF2** - Fallback PDF text extraction
- **pdf2image** - PDF to image conversion for Textract

