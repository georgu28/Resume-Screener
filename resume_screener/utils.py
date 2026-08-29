"""
Utility functions for MDST Resume Screener
"""

import os
import logging
from typing import List, Dict, Optional
import re
from resume_screener.config import LOGGING_CONFIG

def setup_logging(level: str = LOGGING_CONFIG['level']) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level (str): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format=LOGGING_CONFIG['format']
    )
    return logging.getLogger(__name__)

def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and is readable.
    
    Args:
        file_path (str): Path to validate
        
    Returns:
        bool: True if file exists and is readable
    """
    return os.path.exists(file_path) and os.access(file_path, os.R_OK)

def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', ' ', text)
    
    # Remove email addresses (keep for separate extraction)
    # text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_years_experience(text: str) -> Optional[int]:
    """
    Extract years of experience from resume text.
    
    Args:
        text (str): Resume text
        
    Returns:
        Optional[int]: Years of experience if found
    """
    # Patterns for years of experience
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
        r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*in\s*\w+',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                return int(matches[0])
            except ValueError:
                continue
    
    return None

def extract_education_level(text: str) -> Optional[str]:
    """
    Extract highest education level from resume text.
    
    Args:
        text (str): Resume text
        
    Returns:
        Optional[str]: Education level if found
    """
    education_levels = {
        'phd': ['phd', 'ph.d', 'doctorate', 'doctoral'],
        'masters': ['masters', 'master', 'm.s', 'mba', 'ma', 'ms'],
        'bachelors': ['bachelors', 'bachelor', 'b.s', 'ba', 'bs', 'b.a'],
        'associates': ['associates', 'associate', 'aa', 'as'],
        'high_school': ['high school', 'diploma', 'ged']
    }
    
    text_lower = text.lower()
    
    for level, keywords in education_levels.items():
        for keyword in keywords:
            if keyword in text_lower:
                return level
    
    return None

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard similarity.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to sets of words
    words1 = set(clean_text(text1).split())
    words2 = set(clean_text(text2).split())
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def format_phone_number(phone: str) -> str:
    """
    Format phone number to standard format.
    
    Args:
        phone (str): Raw phone number
        
    Returns:
        str: Formatted phone number
    """
    if not phone:
        return ""
    
    # Remove all non-digits
    digits = re.sub(r'\D', '', phone)
    
    # Format as (XXX) XXX-XXXX
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    
    return phone  # Return original if can't format

def get_file_extension(file_path: str) -> str:
    """
    Get file extension from file path.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        str: File extension (without dot)
    """
    return os.path.splitext(file_path)[1][1:].lower()

def create_directory_if_not_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def batch_process_pdfs(pdf_directory: str, processor_func) -> Dict[str, any]:
    """
    Process all PDFs in a directory with a given function.
    
    Args:
        pdf_directory (str): Directory containing PDFs
        processor_func: Function to process each PDF
        
    Returns:
        Dict[str, any]: Results mapped by filename
    """
    results = {}
    
    if not os.path.exists(pdf_directory):
        return results
    
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(pdf_directory, filename)
            try:
                result = processor_func(file_path)
                results[filename] = result
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                results[filename] = None
    
    return results
