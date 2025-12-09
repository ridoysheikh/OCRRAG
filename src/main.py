"""
OCR RAG Demo - Main Entry Point
Demonstrates PDF OCR and RAG chat with citations.
"""

import argparse
from pathlib import Path
import json
from dotenv import load_dotenv

from .ocr import get_ocr_processor, LocalOCR
from .rag import RAGChat, VectorStore

load_dotenv()


def process_document(filepath: str, use_textract: bool = False) -> dict:
    """Process a document with OCR and add to vector store."""
    print(f"📄 Processing: {filepath}")
    
    # Get OCR processor
    if use_textract:
        print("   Using AWS Textract OCR...")
        ocr = get_ocr_processor(use_textract=True)
    else:
        print("   Using local PDF extraction (for text-based PDFs)...")
        ocr = LocalOCR()
    
    # Extract text
    result = ocr.extract_from_file(filepath)
    print(f"   ✓ Extracted {result.total_pages} pages")
    
    # Save OCR result
    output_path = result.save("./data/processed")
    print(f"   ✓ Saved OCR result to: {output_path}")
    
    return result.to_dict()


def add_to_vector_store(ocr_result: dict, vector_store: VectorStore) -> int:
    """Add OCR result to vector store."""
    pages = [
        {"page_number": p["page_number"], "text": p["text"]}
        for p in ocr_result["pages"]
    ]
    
    chunks = vector_store.add_document(
        filename=ocr_result["filename"],
        pages=pages
    )
    print(f"   ✓ Added {chunks} chunks to vector store")
    return chunks


def interactive_chat(rag: RAGChat):
    """Interactive chat loop."""
    print("\n" + "="*60)
    print("💬 RAG Chat with Citations")
    print("="*60)
    print("Ask questions about your documents. Type 'quit' to exit.\n")
    
    while True:
        query = input("You: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        print("\n🔍 Searching documents...")
        response = rag.chat(query)
        
        if response.refused:
            print(f"\n❌ {response.answer}")
            print(f"   Reason: {response.refusal_reason}")
        else:
            print(f"\n📝 Answer:\n{response.answer}")
            
            print("\n📚 Citations:")
            for i, citation in enumerate(response.citations, 1):
                print(f"   [{i}] {citation['filename']}, Page {citation['page_number']}")
                print(f"       Relevance: {citation['relevance_score']:.2f}")
                print(f"       Snippet: \"{citation['snippet'][:80]}...\"")
            
            if response.quote_verification.get("status") != "skipped":
                print(f"\n✅ Quote Verification: {response.quote_verification['status']}")
                if response.quote_verification.get("unverified"):
                    print(f"   ⚠️ Unverified quotes: {len(response.quote_verification['unverified'])}")
        
        print("\n" + "-"*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="OCR RAG Demo")
    parser.add_argument(
        "command",
        choices=["process", "chat", "list", "demo"],
        help="Command to run"
    )
    parser.add_argument(
        "--file", "-f",
        help="PDF file to process"
    )
    parser.add_argument(
        "--textract",
        action="store_true",
        help="Use AWS Textract (requires AWS credentials)"
    )
    parser.add_argument(
        "--query", "-q",
        help="Query for single-shot chat mode"
    )
    
    args = parser.parse_args()
    
    # Initialize vector store
    vector_store = VectorStore()
    rag = RAGChat(vector_store=vector_store)
    
    if args.command == "process":
        if not args.file:
            print("❌ Error: --file is required for process command")
            return
        
        ocr_result = process_document(args.file, use_textract=args.textract)
        add_to_vector_store(ocr_result, vector_store)
        print("\n✅ Document processed and indexed!")
    
    elif args.command == "chat":
        if args.query:
            # Single query mode
            response = rag.chat(args.query)
            print(json.dumps({
                "answer": response.answer,
                "citations": response.citations,
                "quote_verification": response.quote_verification
            }, indent=2))
        else:
            # Interactive mode
            interactive_chat(rag)
    
    elif args.command == "list":
        stats = vector_store.get_stats()
        print("\n📊 Vector Store Stats:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Documents: {', '.join(stats['documents']) or 'None'}")
    
    elif args.command == "demo":
        print("="*60)
        print("🚀 OCR RAG Demo")
        print("="*60)
        print("""
This demo shows:
1. PDF OCR (AWS Textract or local extraction)
2. Vector storage with ChromaDB
3. RAG chat with citations (file + page + snippet)
4. Quote verification against sources

Quick Start:
  # Process a PDF
  python -m src.main process --file document.pdf
  
  # Start interactive chat
  python -m src.main chat
  
  # Single query
  python -m src.main chat --query "What is the main topic?"
  
  # List indexed documents
  python -m src.main list

For AWS Textract, add --textract flag and configure .env
        """)


if __name__ == "__main__":
    main()
