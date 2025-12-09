"""OCR Module"""
from .textract_ocr import TextractOCR, LocalOCR, DocumentOCR, PageContent, get_ocr_processor

__all__ = ['TextractOCR', 'LocalOCR', 'DocumentOCR', 'PageContent', 'get_ocr_processor']
