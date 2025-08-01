#!/usr/bin/env python3
"""
Full PDF Extraction with Cleanup
================================
This script extracts the complete text from "Outwitting the Devil"
using pdfplumber (our best performer) and applies all cleanup steps.
"""

import os
import re
import pdfplumber
from tqdm import tqdm

# Configuration
PDF_PATH = "Hill_Napoleon_-_Outwitting_the_devil.pdf"
OUTPUT_DIR = "processed_text"
CLEANED_TEXT_FILE = "outwitting_the_devil_cleaned.txt"

def extract_with_cleanup(pdf_path, output_path):
    """
    Extract text from PDF with comprehensive cleanup
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the cleaned text
    """
    
    print("Starting PDF extraction and cleanup...")
    print(f"Input: {pdf_path}")
    print(f"Output: {output_path}")
    print("-" * 60)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract text from all pages
    extracted_pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages to process: {total_pages}")
        
        # Process each page with progress bar
        for page_num in tqdm(range(total_pages), desc="Extracting pages"):
            page = pdf.pages[page_num]
            text = page.extract_text()
            
            if text:  # Only add non-empty pages
                # Store page number with text for reference
                extracted_pages.append({
                    'page_num': page_num + 1,
                    'text': text
                })
    
    print(f"\nExtracted {len(extracted_pages)} non-empty pages")
    
    # Apply cleanup steps
    print("\nApplying cleanup steps...")
    
    cleaned_pages = []
    for page_data in tqdm(extracted_pages, desc="Cleaning pages"):
        cleaned_text = clean_page_text(page_data['text'])
        if cleaned_text.strip():  # Only keep pages with content after cleaning
            cleaned_pages.append({
                'page_num': page_data['page_num'],
                'text': cleaned_text
            })
    
    # Join all pages with clear separation
    full_text = []
    for page_data in cleaned_pages:
        # Add page marker for reference (can be removed later if not needed)
        full_text.append(f"\n[PAGE {page_data['page_num']}]\n")
        full_text.append(page_data['text'])
    
    final_text = ''.join(full_text)
    
    # Additional global cleanup
    final_text = global_cleanup(final_text)
    
    # Save the cleaned text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print(f"Pages processed: {len(extracted_pages)}")
    print(f"Pages with content: {len(cleaned_pages)}")
    print(f"Total characters: {len(final_text):,}")
    print(f"Total words (approx): {len(final_text.split()):,}")
    print(f"Output saved to: {output_path}")
    
    return final_text

def clean_page_text(text):
    """
    Clean individual page text
    
    Args:
        text: Raw text from a page
        
    Returns:
        Cleaned text
    """
    
    # Remove standalone page numbers
    # Look for lines that are just numbers
    lines = text.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip lines that are just page numbers
        if stripped.isdigit() and 1 <= int(stripped) <= 500:
            # Check if this is isolated (surrounded by empty lines or at boundaries)
            prev_empty = i == 0 or lines[i-1].strip() == ''
            next_empty = i == len(lines)-1 or lines[i+1].strip() == ''
            if prev_empty and next_empty:
                continue  # Skip this line
        
        # Skip common headers/footers
        if stripped in ['OUTWITTING THE DEVIL', 'NAPOLEON HILL']:
            # Check if it's isolated
            prev_empty = i == 0 or lines[i-1].strip() == ''
            next_empty = i == len(lines)-1 or lines[i+1].strip() == ''
            if prev_empty and next_empty:
                continue
        
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Join hyphenated words at line breaks
    # Look for word- at end of line followed by lowercase start
    text = re.sub(r'(\w+)-\n(\w)', r'\1\2', text)
    
    # Normalize whitespace
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove weird artifacts
    text = text.replace('™', '')
    text = text.replace('®', '')
    text = text.replace('©', '')
    
    return text.strip()

def global_cleanup(text):
    """
    Apply global cleanup to the entire text
    
    Args:
        text: Full concatenated text
        
    Returns:
        Cleaned text
    """
    
    # Remove any remaining excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # Clean up quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove any remaining special characters that might cause issues
    # Keep only standard ASCII and common unicode
    text = re.sub(r'[^\x00-\x7F\u2018-\u201D\u2014\u2026]+', '', text)
    
    return text.strip()

def save_sample(text, sample_path, num_chars=5000):
    """
    Save a sample of the cleaned text for quick review
    
    Args:
        text: Full cleaned text
        sample_path: Path to save the sample
        num_chars: Number of characters to include in sample
    """
    sample = text[:num_chars]
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write("SAMPLE OF CLEANED TEXT (First 5000 characters)\n")
        f.write("=" * 60 + "\n\n")
        f.write(sample)
        f.write("\n\n" + "=" * 60)
        f.write("\n[... continued in full file ...]")
    print(f"Sample saved to: {sample_path}")

def main():
    """Main execution function"""
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found: {PDF_PATH}")
        return
    
    # Set output path
    output_path = os.path.join(OUTPUT_DIR, CLEANED_TEXT_FILE)
    
    # Extract and clean the PDF
    cleaned_text = extract_with_cleanup(PDF_PATH, output_path)
    
    # Save a sample for quick review
    sample_path = os.path.join(OUTPUT_DIR, "sample_cleaned_text.txt")
    save_sample(cleaned_text, sample_path)
    
    print("\n✅ Extraction and cleanup complete!")
    print("\nNext steps:")
    print("1. Review the sample file to verify quality")
    print("2. If satisfied, proceed with chunking strategy")
    print("3. If issues found, adjust cleanup rules and re-run")

if __name__ == "__main__":
    main()