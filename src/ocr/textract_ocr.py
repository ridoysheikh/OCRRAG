"""
AWS Textract OCR Module
Extracts text from scanned PDFs and images with page-level metadata.
"""

import boto3
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import json
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PageContent:
    """Represents extracted text from a single page."""
    page_number: int
    text: str
    confidence: float
    blocks: list = field(default_factory=list)


@dataclass
class DocumentOCR:
    """Represents OCR results for an entire document."""
    filename: str
    filepath: str
    pages: list[PageContent] = field(default_factory=list)
    total_pages: int = 0
    
    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "total_pages": self.total_pages,
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "confidence": p.confidence
                }
                for p in self.pages
            ]
        }
    
    def save(self, output_dir: str) -> str:
        """Save OCR results to JSON file."""
        output_path = Path(output_dir) / f"{Path(self.filename).stem}_ocr.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(output_path)


class TextractOCR:
    """AWS Textract OCR processor for PDFs and images."""
    
    def __init__(
        self,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        region: str = "us-east-1"
    ):
        self.client = boto3.client(
            'textract',
            aws_access_key_id=aws_access_key or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region or os.getenv('AWS_REGION', 'us-east-1')
        )
    
    def extract_from_file(self, filepath: str) -> DocumentOCR:
        """
        Extract text from a local PDF or image file.
        For multi-page PDFs, uses async API with S3.
        For single images, uses sync API.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        if suffix in ['.jpg', '.jpeg', '.png', '.tiff']:
            return self._extract_from_image(filepath)
        elif suffix == '.pdf':
            return self._extract_from_pdf(filepath)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _extract_from_image(self, filepath: Path) -> DocumentOCR:
        """Extract text from a single image using sync API."""
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        
        response = self.client.detect_document_text(
            Document={'Bytes': image_bytes}
        )
        
        text_blocks = []
        full_text = []
        total_confidence = 0
        block_count = 0
        
        for block in response.get('Blocks', []):
            if block['BlockType'] == 'LINE':
                text_blocks.append({
                    'text': block['Text'],
                    'confidence': block['Confidence'],
                    'geometry': block['Geometry']
                })
                full_text.append(block['Text'])
                total_confidence += block['Confidence']
                block_count += 1
        
        avg_confidence = total_confidence / block_count if block_count > 0 else 0
        
        page = PageContent(
            page_number=1,
            text='\n'.join(full_text),
            confidence=avg_confidence,
            blocks=text_blocks
        )
        
        return DocumentOCR(
            filename=filepath.name,
            filepath=str(filepath),
            pages=[page],
            total_pages=1
        )
    
    def _extract_from_pdf(self, filepath: Path) -> DocumentOCR:
        """
        Extract text from PDF using Textract.
        For demo, we use analyze_document with each page converted to image.
        In production, use async StartDocumentTextDetection with S3.
        """
        # For minimal demo - process PDF page by page
        # This requires pdf2image to convert PDF pages to images
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(filepath)
            pages = []
            
            for page_num, image in enumerate(images, start=1):
                # Convert PIL image to bytes
                import io
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                
                response = self.client.detect_document_text(
                    Document={'Bytes': img_bytes}
                )
                
                text_blocks = []
                full_text = []
                total_confidence = 0
                block_count = 0
                
                for block in response.get('Blocks', []):
                    if block['BlockType'] == 'LINE':
                        text_blocks.append({
                            'text': block['Text'],
                            'confidence': block['Confidence']
                        })
                        full_text.append(block['Text'])
                        total_confidence += block['Confidence']
                        block_count += 1
                
                avg_confidence = total_confidence / block_count if block_count > 0 else 0
                
                pages.append(PageContent(
                    page_number=page_num,
                    text='\n'.join(full_text),
                    confidence=avg_confidence,
                    blocks=text_blocks
                ))
            
            return DocumentOCR(
                filename=filepath.name,
                filepath=str(filepath),
                pages=pages,
                total_pages=len(pages)
            )
            
        except ImportError:
            raise ImportError("pdf2image is required for PDF processing. Install with: pip install pdf2image")


class LocalOCR:
    """
    Fallback OCR using PyPDF2 for text-based PDFs (non-scanned).
    Use this when Textract is not available or for testing.
    """
    
    def extract_from_file(self, filepath: str) -> DocumentOCR:
        """Extract text from a text-based PDF."""
        from PyPDF2 import PdfReader
        
        filepath = Path(filepath)
        reader = PdfReader(filepath)
        pages = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageContent(
                page_number=page_num,
                text=text,
                confidence=100.0 if text else 0.0
            ))
        
        return DocumentOCR(
            filename=filepath.name,
            filepath=str(filepath),
            pages=pages,
            total_pages=len(pages)
        )


def get_ocr_processor(use_textract: bool = True) -> TextractOCR | LocalOCR:
    """Factory function to get the appropriate OCR processor."""
    if use_textract:
        return TextractOCR()
    return LocalOCR()
