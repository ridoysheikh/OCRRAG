"""
Quote Verification Module
Validates that any quote in the response matches source text.
"""

from difflib import SequenceMatcher
from typing import Optional
import re


def extract_quotes(text: str) -> list[str]:
    """
    Extract quoted text from a response.
    
    Matches text within double quotes or single quotes.
    """
    # Match double quotes
    double_quotes = re.findall(r'"([^"]+)"', text)
    # Match single quotes (but avoid contractions)
    single_quotes = re.findall(r"'([^']{10,})'", text)  # Min 10 chars to avoid contractions
    
    return double_quotes + single_quotes


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase, remove extra whitespace
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def find_quote_in_source(
    quote: str,
    source_texts: list[dict],  # [{"text": str, "filename": str, "page_number": int}]
    threshold: float = 0.85
) -> Optional[dict]:
    """
    Find if a quote exists in source texts.
    
    Args:
        quote: The quote to verify.
        source_texts: List of source text dicts with metadata.
        threshold: Minimum similarity ratio (0-1) to consider a match.
        
    Returns:
        Source dict if found, None otherwise.
    """
    normalized_quote = normalize_text(quote)
    
    for source in source_texts:
        normalized_source = normalize_text(source["text"])
        
        # Check if quote is substring (exact match)
        if normalized_quote in normalized_source:
            return source
        
        # Check fuzzy match using sliding window
        quote_len = len(normalized_quote)
        
        for i in range(len(normalized_source) - quote_len + 1):
            window = normalized_source[i:i + quote_len]
            ratio = SequenceMatcher(None, normalized_quote, window).ratio()
            
            if ratio >= threshold:
                return source
    
    return None


def verify_quotes_in_response(
    response: str,
    source_texts: list[dict],
    threshold: float = 0.85
) -> dict:
    """
    Verify all quotes in a response against source texts.
    
    Args:
        response: The generated response text.
        source_texts: List of source text dicts.
        threshold: Minimum similarity for match.
        
    Returns:
        Dict with verified quotes, unverified quotes, and overall status.
    """
    quotes = extract_quotes(response)
    
    if not quotes:
        return {
            "status": "no_quotes",
            "verified": [],
            "unverified": [],
            "all_verified": True
        }
    
    verified = []
    unverified = []
    
    for quote in quotes:
        source = find_quote_in_source(quote, source_texts, threshold)
        
        if source:
            verified.append({
                "quote": quote,
                "source_file": source["filename"],
                "source_page": source["page_number"]
            })
        else:
            unverified.append(quote)
    
    return {
        "status": "verified" if not unverified else "partial" if verified else "unverified",
        "verified": verified,
        "unverified": unverified,
        "all_verified": len(unverified) == 0
    }


def remove_unverified_quotes(response: str, unverified_quotes: list[str]) -> str:
    """
    Remove unverified quotes from response or mark them.
    
    Args:
        response: Original response.
        unverified_quotes: List of unverified quote strings.
        
    Returns:
        Response with unverified quotes marked.
    """
    modified = response
    
    for quote in unverified_quotes:
        # Mark unverified quotes
        modified = modified.replace(
            f'"{quote}"',
            f'[UNVERIFIED QUOTE REMOVED]'
        )
        modified = modified.replace(
            f"'{quote}'",
            f'[UNVERIFIED QUOTE REMOVED]'
        )
    
    return modified
