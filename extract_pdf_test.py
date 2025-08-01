#!/usr/bin/env python3
"""
PDF Text Extraction Test Script
================================
This script tests different PDF extraction methods on the first 10 pages
of "Outwitting the Devil" to determine which produces the cleanest output.

We'll test three methods:
1. PyPDF2 - Simple and fast, good for basic PDFs
2. pdfplumber - Better at handling complex layouts
3. pymupdf (fitz) - High quality extraction with formatting preservation
"""

import os
import sys
from pathlib import Path

# Configuration
PDF_PATH = "Hill_Napoleon_-_Outwitting_the_devil.pdf"
OUTPUT_DIR = "extraction_results"
PAGES_TO_EXTRACT = 10

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}/")

def extract_with_pypdf2():
    """
    Extract text using PyPDF2
    - Pros: Fast, pure Python, no external dependencies
    - Cons: May struggle with complex layouts
    """
    try:
        from PyPDF2 import PdfReader
        print("\n1. Testing PyPDF2 extraction...")
        
        # Open the PDF
        reader = PdfReader(PDF_PATH)
        total_pages = len(reader.pages)
        print(f"   Total pages in PDF: {total_pages}")
        
        # Extract first N pages
        extracted_text = []
        pages_to_process = min(PAGES_TO_EXTRACT, total_pages)
        
        for page_num in range(pages_to_process):
            page = reader.pages[page_num]
            text = page.extract_text()
            extracted_text.append(f"\n{'='*50}\nPAGE {page_num + 1}\n{'='*50}\n{text}")
            print(f"   ✓ Extracted page {page_num + 1}")
        
        # Save to file
        output_path = os.path.join(OUTPUT_DIR, "pypdf2_extraction.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(extracted_text))
        
        print(f"   ✓ Saved to: {output_path}")
        return True
        
    except ImportError:
        print("   ✗ PyPDF2 not installed. Run: pip install PyPDF2")
        return False
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False

def extract_with_pdfplumber():
    """
    Extract text using pdfplumber
    - Pros: Better layout detection, can extract tables
    - Cons: Slower than PyPDF2
    """
    try:
        import pdfplumber
        print("\n2. Testing pdfplumber extraction...")
        
        # Open the PDF
        with pdfplumber.open(PDF_PATH) as pdf:
            total_pages = len(pdf.pages)
            print(f"   Total pages in PDF: {total_pages}")
            
            # Extract first N pages
            extracted_text = []
            pages_to_process = min(PAGES_TO_EXTRACT, total_pages)
            
            for page_num in range(pages_to_process):
                page = pdf.pages[page_num]
                text = page.extract_text()
                extracted_text.append(f"\n{'='*50}\nPAGE {page_num + 1}\n{'='*50}\n{text}")
                print(f"   ✓ Extracted page {page_num + 1}")
        
        # Save to file
        output_path = os.path.join(OUTPUT_DIR, "pdfplumber_extraction.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(extracted_text))
        
        print(f"   ✓ Saved to: {output_path}")
        return True
        
    except ImportError:
        print("   ✗ pdfplumber not installed. Run: pip install pdfplumber")
        return False
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False

def extract_with_pymupdf():
    """
    Extract text using PyMuPDF (fitz)
    - Pros: High quality, preserves formatting, fast
    - Cons: Larger dependency
    """
    try:
        import fitz  # PyMuPDF
        print("\n3. Testing PyMuPDF extraction...")
        
        # Open the PDF
        doc = fitz.open(PDF_PATH)
        total_pages = doc.page_count
        print(f"   Total pages in PDF: {total_pages}")
        
        # Extract first N pages
        extracted_text = []
        pages_to_process = min(PAGES_TO_EXTRACT, total_pages)
        
        for page_num in range(pages_to_process):
            page = doc[page_num]
            text = page.get_text()
            extracted_text.append(f"\n{'='*50}\nPAGE {page_num + 1}\n{'='*50}\n{text}")
            print(f"   ✓ Extracted page {page_num + 1}")
        
        doc.close()
        
        # Save to file
        output_path = os.path.join(OUTPUT_DIR, "pymupdf_extraction.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(extracted_text))
        
        print(f"   ✓ Saved to: {output_path}")
        return True
        
    except ImportError:
        print("   ✗ PyMuPDF not installed. Run: pip install pymupdf")
        return False
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False

def analyze_extraction_quality():
    """
    Analyze the extracted text files for common issues:
    - Page numbers
    - Headers/footers
    - Excessive whitespace
    - Special characters
    """
    print("\n" + "="*60)
    print("QUALITY ANALYSIS")
    print("="*60)
    
    # Check each extraction result
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith('.txt'):
            filepath = os.path.join(OUTPUT_DIR, filename)
            print(f"\nAnalyzing: {filename}")
            print("-" * 40)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Get first 500 chars as sample
            sample = content[:500].strip()
            
            # Check for common issues
            issues = []
            
            # Check for page numbers (common patterns)
            import re
            if re.search(r'\n\s*\d{1,3}\s*\n', content[:2000]):
                issues.append("Possible standalone page numbers detected")
            
            # Check for repeated headers/footers
            lines = content.split('\n')[:50]  # First 50 lines
            line_counts = {}
            for line in lines:
                line_stripped = line.strip()
                if 5 < len(line_stripped) < 100:  # Reasonable header/footer length
                    line_counts[line_stripped] = line_counts.get(line_stripped, 0) + 1
            
            repeated_lines = [line for line, count in line_counts.items() if count > 2]
            if repeated_lines:
                issues.append(f"Possible headers/footers: {repeated_lines[:2]}")
            
            # Check for excessive whitespace
            double_newlines = content.count('\n\n\n')
            if double_newlines > 10:
                issues.append(f"Excessive whitespace detected ({double_newlines} occurrences)")
            
            # Print findings
            print("Sample text:")
            print(sample)
            print("\nIssues found:")
            if issues:
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("  - No major issues detected")

def main():
    """Main execution function"""
    print("PDF Text Extraction Quality Test")
    print("================================")
    print(f"PDF file: {PDF_PATH}")
    print(f"Pages to extract: {PAGES_TO_EXTRACT}")
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"\n✗ Error: PDF file not found: {PDF_PATH}")
        sys.exit(1)
    
    # Create output directory
    ensure_output_dir()
    
    # Test each extraction method
    methods_tested = 0
    if extract_with_pypdf2():
        methods_tested += 1
    if extract_with_pdfplumber():
        methods_tested += 1
    if extract_with_pymupdf():
        methods_tested += 1
    
    # Analyze results if any methods succeeded
    if methods_tested > 0:
        analyze_extraction_quality()
        print(f"\n✓ Extraction complete! Check the '{OUTPUT_DIR}' directory for results.")
        print("\nNext steps:")
        print("1. Review each extraction file to see which has the cleanest output")
        print("2. Check for headers, footers, and page numbers that need removal")
        print("3. Choose the best extraction method for the full PDF")
    else:
        print("\n✗ No extraction methods available. Please install at least one:")
        print("  pip install PyPDF2 pdfplumber pymupdf")

if __name__ == "__main__":
    main()