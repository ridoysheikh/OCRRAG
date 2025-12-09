# OCR RAG Demo - Minimal Implementation

A minimal demonstration of PDF RAG (Retrieval-Augmented Generation) with OCR capabilities.

## Features

- **AWS Textract OCR**: Extract text from scanned PDFs/images
- **RAG Chat with Citations**: Query documents with file name + page number + snippet
- **Quote Verification**: Validates quotes against source text before output
- **Searchable Document Storage**: ChromaDB vector store for semantic search

## Project Structure

```
OCRRAG/
├── src/
│   ├── ocr/
│   │   └── textract_ocr.py      # AWS Textract integration
│   ├── rag/
│   │   ├── embeddings.py        # Document embeddings
│   │   ├── vector_store.py      # ChromaDB storage
│   │   └── chat.py              # RAG chat with citations
│   ├── utils/
│   │   └── quote_verify.py      # Quote verification
│   └── main.py                  # Demo entry point
├── data/
│   ├── uploads/                 # Uploaded PDFs
│   └── processed/               # Processed text with metadata
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your AWS and OpenAI credentials
   ```

3. **Run demo**:
   ```bash
   python -m src.main
   ```

## API Endpoints (Optional)

```bash
uvicorn src.api:app --reload
```

- `POST /upload` - Upload PDF for OCR processing
- `POST /chat` - RAG chat with citations
- `GET /documents` - List processed documents

## Environment Variables

- `AWS_ACCESS_KEY_ID` - AWS credentials for Textract
- `AWS_SECRET_ACCESS_KEY` - AWS credentials
- `AWS_REGION` - AWS region (default: us-east-1)
- `OPENAI_API_KEY` - OpenAI API key for embeddings/chat

## Citation Format

All responses include citations in format:
```
[Source: filename.pdf, Page X, "exact quote snippet..."]
```

If no relevant source found, the system refuses to answer.
