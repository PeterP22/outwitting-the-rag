#!/usr/bin/env python3
"""
Simple Text Chunking with Metadata
==================================
This script chunks the cleaned text from "Outwitting the Devil" into
overlapping segments with metadata, treating the book as continuous text.

Chunking Strategy:
- Size: 500 words per chunk
- Overlap: 50 words (10%)
- Metadata: page tracking, chunk position, content preview
"""

import os
import json
import re
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

# Configuration
INPUT_FILE = "processed_text/outwitting_the_devil_cleaned.txt"
OUTPUT_DIR = "processed_text"
OUTPUT_FILE = "book_chunks.jsonl"
CHUNK_SIZE = 500  # words
OVERLAP_SIZE = 50  # words (10% of chunk size)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    chunk_id: str
    text: str
    word_count: int
    char_count: int
    start_page: int
    end_page: int
    position: int  # Position in the overall sequence
    total_chunks: int
    preview: str  # First 150 chars for quick reference
    timestamp: str
    

def chunk_text_with_overlap(words: List[str], chunk_size: int, overlap_size: int) -> List[List[str]]:
    """
    Split words into overlapping chunks
    
    Args:
        words: List of words
        chunk_size: Number of words per chunk
        overlap_size: Number of overlapping words
    
    Returns:
        List of word lists (each chunk is a list of words)
    """
    if len(words) <= chunk_size:
        return [words]
    
    chunks = []
    stride = chunk_size - overlap_size
    
    for i in range(0, len(words), stride):
        chunk = words[i:i + chunk_size]
        
        # Don't create tiny final chunks
        if len(chunk) < chunk_size // 3 and chunks:
            # Extend the previous chunk instead
            chunks[-1].extend(chunk)
        else:
            chunks.append(chunk)
        
        # Stop if we've processed all words
        if i + chunk_size >= len(words):
            break
    
    return chunks


def create_chunks(input_file: str, output_file: str):
    """
    Main function to create chunks from the cleaned text
    
    Args:
        input_file: Path to cleaned text file
        output_file: Path to output JSONL file
    """
    print(f"Reading cleaned text from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    print(f"Total text length: {len(full_text):,} characters")
    
    # Split text by page markers to track pages
    page_sections = re.split(r'(\[PAGE\s+\d+\])', full_text)
    
    # Build word list with page tracking
    all_words = []
    word_to_page = {}  # Map word index to page number
    current_page = 1  # Default page
    
    for section in page_sections:
        # Check if this is a page marker
        page_match = re.match(r'\[PAGE\s+(\d+)\]', section)
        if page_match:
            current_page = int(page_match.group(1))
        else:
            # This is content
            words = section.split()
            start_idx = len(all_words)
            all_words.extend(words)
            
            # Map each word to its page
            for i in range(start_idx, len(all_words)):
                word_to_page[i] = current_page
    
    print(f"Total words: {len(all_words):,}")
    
    # Create chunks
    chunks_data = chunk_text_with_overlap(all_words, CHUNK_SIZE, OVERLAP_SIZE)
    print(f"Created {len(chunks_data)} chunks")
    
    # Convert to TextChunk objects
    all_chunks = []
    word_index = 0
    
    for position, chunk_words in enumerate(tqdm(chunks_data, desc="Processing chunks")):
        chunk_text = ' '.join(chunk_words)
        
        # Determine page range
        chunk_start_idx = word_index
        chunk_end_idx = min(word_index + len(chunk_words) - 1, len(all_words) - 1)
        
        start_page = word_to_page.get(chunk_start_idx, 1)
        end_page = word_to_page.get(chunk_end_idx, start_page)
        
        # Create preview (first 150 chars)
        preview = chunk_text[:150]
        if len(chunk_text) > 150:
            # Find last complete word
            last_space = preview.rfind(' ')
            if last_space > 100:
                preview = preview[:last_space] + "..."
            else:
                preview = preview + "..."
        
        # Create chunk object
        chunk = TextChunk(
            chunk_id=f"chunk_{position:04d}",
            text=chunk_text,
            word_count=len(chunk_words),
            char_count=len(chunk_text),
            start_page=start_page,
            end_page=end_page,
            position=position,
            total_chunks=len(chunks_data),
            preview=preview,
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )
        
        all_chunks.append(chunk)
        
        # Update word index for next chunk (accounting for overlap)
        word_index += CHUNK_SIZE - OVERLAP_SIZE
    
    # Save chunks to JSONL file
    output_path = os.path.join(OUTPUT_DIR, output_file)
    print(f"\nSaving {len(all_chunks)} chunks to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            json_line = json.dumps(asdict(chunk), ensure_ascii=False)
            f.write(json_line + '\n')
    
    # Print statistics
    print("\n" + "="*60)
    print("CHUNKING COMPLETE")
    print("="*60)
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Average chunk size: {sum(c.word_count for c in all_chunks) / len(all_chunks):.1f} words")
    print(f"Chunk size setting: {CHUNK_SIZE} words")
    print(f"Overlap size: {OVERLAP_SIZE} words ({OVERLAP_SIZE/CHUNK_SIZE*100:.0f}%)")
    print(f"Page range: {all_chunks[0].start_page} - {all_chunks[-1].end_page}")
    
    # Save a sample for review
    sample_path = os.path.join(OUTPUT_DIR, "sample_chunks.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        # Save first 3 and last 2 chunks as sample
        sample_chunks = [asdict(chunk) for chunk in all_chunks[:3] + all_chunks[-2:]]
        json.dump(sample_chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSample chunks saved to: {sample_path}")
    
    return all_chunks


def validate_chunks(chunks: List[TextChunk]):
    """
    Validate the chunks for common issues
    
    Args:
        chunks: List of TextChunk objects
    """
    print("\nValidating chunks...")
    
    issues = []
    
    # Check for empty chunks
    empty_chunks = [c for c in chunks if not c.text.strip()]
    if empty_chunks:
        issues.append(f"Found {len(empty_chunks)} empty chunks")
    
    # Check for very small chunks
    small_chunks = [c for c in chunks if c.word_count < CHUNK_SIZE // 3]
    if small_chunks:
        issues.append(f"Found {len(small_chunks)} chunks with < {CHUNK_SIZE//3} words")
    
    # Check for chunks with same text (duplicates)
    seen_texts = set()
    duplicates = 0
    for chunk in chunks:
        if chunk.text in seen_texts:
            duplicates += 1
        seen_texts.add(chunk.text)
    if duplicates:
        issues.append(f"Found {duplicates} duplicate chunks")
    
    # Verify overlap is working
    if len(chunks) > 1:
        # Check first two consecutive chunks
        words1 = chunks[0].text.split()[-OVERLAP_SIZE:]
        words2 = chunks[1].text.split()[:OVERLAP_SIZE]
        
        # Calculate overlap similarity
        overlap_match = sum(1 for w1, w2 in zip(words1, words2) if w1 == w2)
        overlap_ratio = overlap_match / OVERLAP_SIZE
        
        if overlap_ratio < 0.8:  # Allow some flexibility for edge cases
            issues.append(f"Overlap verification shows only {overlap_ratio:.0%} match (expected ~100%)")
    
    # Check page progression
    page_jumps = 0
    for i in range(1, len(chunks)):
        if chunks[i].start_page < chunks[i-1].end_page - 5:  # Allow some backward movement
            page_jumps += 1
    if page_jumps:
        issues.append(f"Found {page_jumps} suspicious page number jumps")
    
    if issues:
        print("⚠️  Validation issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ All chunks validated successfully!")
    
    # Print some positive statistics
    print(f"\n📊 Chunk Statistics:")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Words per chunk: min={min(c.word_count for c in chunks)}, "
          f"max={max(c.word_count for c in chunks)}, "
          f"avg={sum(c.word_count for c in chunks)/len(chunks):.0f}")
    print(f"   - Page coverage: {chunks[0].start_page} to {chunks[-1].end_page}")


def main():
    """Main execution function"""
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        print("Please run extract_full_pdf.py first to generate the cleaned text.")
        return
    
    # Create chunks
    chunks = create_chunks(INPUT_FILE, OUTPUT_FILE)
    
    # Validate chunks
    validate_chunks(chunks)
    
    print("\n✅ Chunking complete!")
    print("\nNext steps:")
    print("1. Review the sample chunks in 'sample_chunks.json'")
    print("2. Check the full chunks in 'book_chunks.jsonl'")
    print("3. Proceed with generating embeddings for each chunk")


if __name__ == "__main__":
    main()