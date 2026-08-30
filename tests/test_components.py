"""
Test script for MDST Resume Screener components
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path so the resume_screener package imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from resume_screener.parser import read_pdf, get_details, get_sections
from resume_screener.classifier import ResumeClassifier
from resume_screener.semantic import SemanticMatcher
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_parser():
    """Test the parser functionality."""
    print("Testing parser...")
    
    # Test with a sample PDF (if available)
    pdf_files = list(Path("pdfs").glob("*.pdf"))
    if pdf_files:
        test_pdf = pdf_files[0]
        print(f"Testing with: {test_pdf}")
        
        try:
            # Test PDF reading
            text = read_pdf(str(test_pdf))
            print(f"✓ PDF text extracted: {len(text)} characters")
            
            # Test detail extraction
            details, cleaned_text = get_details(text)
            print(f"✓ Details extracted: {details}")
            
            # Test section parsing
            sections = get_sections(text)
            print(f"✓ Sections found: {list(sections.keys())}")
            
        except Exception as e:
            print(f"✗ Parser test failed: {e}")
    else:
        print("No PDF files found in pdfs/ directory")

def test_classifier():
    """Test the resume category classifier."""
    print("\nTesting classifier...")
    
    try:
        # Initialize classifier
        classifier = ResumeClassifier()
        print("✓ Classifier initialized")
        
        # Test with sample text
        sample_text = "Python developer with 5 years experience in machine learning and data science"
        category = classifier.predict_text(sample_text)
        print(f"✓ Text classification: {category}")
        
        # Test categories
        categories = classifier.get_categories()
        print(f"✓ Available categories: {categories}")
        
    except Exception as e:
        print(f"✗ Classifier test failed: {e}")

def test_semantic_matcher():
    """Test the semantic matcher."""
    print("\nTesting semantic matcher...")
    
    try:
        # Initialize matcher
        matcher = SemanticMatcher()
        print("✓ Semantic matcher initialized")
        
        # Test with sample texts
        matcher.sentences = [
            "Python developer with machine learning experience",
            "Software engineer position requiring Python and ML skills"
        ]
        
        similarities = matcher.calculate_similarities()
        if similarities is not None:
            print(f"✓ Similarity calculated: {similarities[1]:.3f}")
        else:
            print("✗ Similarity calculation failed")
            
    except Exception as e:
        print(f"✗ Semantic matcher test failed: {e}")

def test_integration():
    """Test integration between components."""
    print("\nTesting integration...")
    
    pdf_files = list(Path("pdfs").glob("*.pdf"))
    if pdf_files:
        test_pdf = pdf_files[0]
        
        try:
            # Test full pipeline
            classifier = ResumeClassifier()
            category = classifier.predict_pdf(str(test_pdf))
            print(f"✓ Full PDF classification: {category}")
            
            # Test probabilities
            probs = classifier.get_prediction_probabilities(str(test_pdf))
            top_category = list(probs.keys())[0]
            top_prob = list(probs.values())[0]
            print(f"✓ Top prediction: {top_category} ({top_prob:.3f})")
            
        except Exception as e:
            print(f"✗ Integration test failed: {e}")
    else:
        print("No PDF files available for integration test")

if __name__ == "__main__":
    print("MDST Resume Screener - Component Tests")
    print("=" * 50)
    
    test_parser()
    test_classifier()
    test_semantic_matcher()
    test_integration()
    
    print("\n" + "=" * 50)
    print("Testing complete!")
