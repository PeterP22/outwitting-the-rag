#!/usr/bin/env python3
"""
Text Chunking with Metadata
===========================
This script chunks the cleaned text from "Outwitting the Devil" into
overlapping segments with rich metadata for RAG retrieval.

Chunking Strategy:
- Size: 500 words per chunk
- Overlap: 50 words (10%)
- Metadata: chapter, page, chunk_id, position, content preview
"""

import os
import json
import re
from typing import List, Dict, Tuple
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
    chapter: str
    chapter_title: str
    position_in_chapter: int
    total_chunks_in_chapter: int
    preview: str  # First 100 chars for quick reference
    timestamp: str
    

def extract_chapters(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract chapter information from the text
    
    Returns:
        List of tuples: (chapter_number, chapter_title, chapter_text)
    """
    # First, let's skip the table of contents
    # Look for where the actual content starts (after "About The Napoleon Hill Foundation" or similar)
    toc_end_markers = [
        "Share Your Stories",
        "About The Napoleon Hill Foundation",
        "Index .",
        "ANNOTATED BY"
    ]
    
    content_start = 0
    for marker in toc_end_markers:
        pos = text.find(marker)
        if pos > 0:
            # Find the next page marker after this
            next_page = text.find("[PAGE", pos)
            if next_page > 0:
                content_start = max(content_start, next_page)
    
    # If we found a content start, use it
    if content_start > 0:
        text = text[content_start:]
        print(f"Skipped {content_start} characters of front matter")
    
    # Now look for actual chapter content
    # Real chapters usually have more substantial text after them
    # Pattern: Chapter N followed by title, then actual content
    chapter_pattern = r'Chapter\s+(\d+)[:\s]+([^\.]+?)(?:\s*\.+\s*\d+)?\s*\n'
    
    chapters = []
    
    # Find all potential chapter headings
    chapter_matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))
    
    # Filter out table of contents entries (they have page numbers and little content after)
    real_chapters = []
    for match in chapter_matches:
        # Check if there's substantial content after this match
        start_pos = match.end()
        next_100_chars = text[start_pos:start_pos + 100].strip()
        
        # Skip if it looks like a TOC entry (next line is another chapter or very short)
        if not re.match(r'^Chapter\s+\d+', next_100_chars) and len(next_100_chars) > 20:
            real_chapters.append(match)
    
    if not real_chapters:
        # If no chapters found, treat entire text as one chapter
        print("No chapter markers found. Treating entire text as single chapter.")
        return [("1", "Full Text", text)]
    
    # Extract each chapter
    for i, match in enumerate(real_chapters):
        chapter_num = match.group(1)
        chapter_title = match.group(2).strip()
        
        # Clean up chapter title (remove dots and page numbers)
        chapter_title = re.sub(r'\.+\s*\d+\s*$', '', chapter_title).strip()
        
        # Get text from this chapter start to next chapter (or end)
        start_pos = match.start()
        if i < len(real_chapters) - 1:
            end_pos = real_chapters[i + 1].start()
        else:
            end_pos = len(text)
        
        chapter_text = text[start_pos:end_pos]
        
        # Only include if chapter has substantial content
        if len(chapter_text.strip()) > 100:
            chapters.append((chapter_num, chapter_title, chapter_text))
    
    # Also check for content before first chapter
    if real_chapters and real_chapters[0].start() > 500:  # Significant content before Chapter 1
        pre_chapter_text = text[:real_chapters[0].start()]
        if len(pre_chapter_text.strip()) > 100:
            chapters.insert(0, ("0", "Introduction/Preface", pre_chapter_text))
    
    return chapters


def extract_page_info(text: str) -> Tuple[int, str]:
    """
    Extract page number from text if it contains a page marker
    
    Returns:
        Tuple of (page_number, text_without_marker)
    """
    # Look for our page markers: [PAGE n]
    page_pattern = r'\[PAGE\s+(\d+)\]'
    
    match = re.search(page_pattern, text)
    if match:
        page_num = int(match.group(1))
        # Remove the page marker from text
        clean_text = re.sub(page_pattern, '', text).strip()
        return page_num, clean_text
    
    return 0, text


def chunk_text_with_overlap(text: str, chunk_size: int, overlap_size: int) -> List[List[str]]:
    """
    Split text into overlapping chunks based on word count
    
    Args:
        text: The text to chunk
        chunk_size: Number of words per chunk
        overlap_size: Number of overlapping words
    
    Returns:
        List of word lists (each chunk is a list of words)
    """
    # Split into words while preserving some punctuation info
    words = text.split()
    
    if len(words) <= chunk_size:
        return [words]
    
    chunks = []
    stride = chunk_size - overlap_size
    
    for i in range(0, len(words), stride):
        chunk = words[i:i + chunk_size]
        chunks.append(chunk)
        
        # If this is the last chunk and it's too small, merge with previous
        if len(chunk) < chunk_size // 2 and len(chunks) > 1:
            # Merge with previous chunk
            chunks[-2].extend(chunks[-1])
            chunks.pop()
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
    
    # Extract chapters
    print("\nExtracting chapters...")
    chapters = extract_chapters(full_text)
    print(f"Found {len(chapters)} chapters")
    
    # Process each chapter
    all_chunks = []
    chunk_id_counter = 0
    
    for chapter_num, chapter_title, chapter_text in tqdm(chapters, desc="Processing chapters"):
        # Split chapter text by page markers to track pages
        page_sections = re.split(r'(\[PAGE\s+\d+\])', chapter_text)
        
        # Reconstruct text with page tracking
        current_page = 0
        chapter_words = []
        page_map = {}  # Track which word index belongs to which page
        
        for section in page_sections:
            page_num, clean_section = extract_page_info(section)
            if page_num > 0:
                current_page = page_num
            
            words = clean_section.split()
            start_idx = len(chapter_words)
            chapter_words.extend(words)
            
            # Map word indices to pages
            for i in range(start_idx, len(chapter_words)):
                page_map[i] = current_page
        
        # Create chunks for this chapter
        if chapter_words:  # Only process if chapter has content
            chunks = chunk_text_with_overlap(' '.join(chapter_words), CHUNK_SIZE, OVERLAP_SIZE)
            
            # Process each chunk
            word_index = 0
            for position, chunk_words_list in enumerate(chunks):
                chunk_text = ' '.join(chunk_words_list)
                
                # Determine page range for this chunk
                chunk_start_idx = word_index
                chunk_end_idx = word_index + len(chunk_words_list) - 1
                
                start_page = page_map.get(chunk_start_idx, 0)
                end_page = page_map.get(chunk_end_idx, start_page)
                
                # Create chunk object
                chunk = TextChunk(
                    chunk_id=f"chunk_{chunk_id_counter:04d}",
                    text=chunk_text,
                    word_count=len(chunk_words_list),
                    char_count=len(chunk_text),
                    start_page=start_page,
                    end_page=end_page,
                    chapter=chapter_num,
                    chapter_title=chapter_title,
                    position_in_chapter=position,
                    total_chunks_in_chapter=len(chunks),
                    preview=chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                    timestamp=datetime.utcnow().isoformat()
                )
                
                all_chunks.append(chunk)
                chunk_id_counter += 1
                
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
    
    # Chapter statistics
    chapter_stats = {}
    for chunk in all_chunks:
        if chunk.chapter not in chapter_stats:
            chapter_stats[chunk.chapter] = 0
        chapter_stats[chunk.chapter] += 1
    
    print("\nChunks per chapter:")
    for chapter, count in sorted(chapter_stats.items(), key=lambda x: int(x[0]) if x[0].isdigit() else -1):
        chapter_info = next((c for c in chapters if c[0] == chapter), None)
        if chapter_info:
            print(f"  Chapter {chapter} ({chapter_info[1][:30]}...): {count} chunks")
    
    # Save a sample for review
    sample_path = os.path.join(OUTPUT_DIR, "sample_chunks.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        sample_chunks = [asdict(chunk) for chunk in all_chunks[:5]]
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
    small_chunks = [c for c in chunks if c.word_count < CHUNK_SIZE // 4]
    if small_chunks:
        issues.append(f"Found {len(small_chunks)} chunks with < {CHUNK_SIZE//4} words")
    
    # Check for chunks with no page info
    no_page_chunks = [c for c in chunks if c.start_page == 0]
    if no_page_chunks:
        issues.append(f"Found {len(no_page_chunks)} chunks with no page information")
    
    # Verify overlap is working
    if len(chunks) > 1:
        # Check if consecutive chunks have overlapping text
        overlap_works = False
        for i in range(len(chunks) - 1):
            if chunks[i].chapter == chunks[i+1].chapter:  # Same chapter
                words1 = chunks[i].text.split()[-OVERLAP_SIZE:]
                words2 = chunks[i+1].text.split()[:OVERLAP_SIZE]
                if words1 == words2:
                    overlap_works = True
                    break
        
        if not overlap_works:
            issues.append("Overlap verification failed - chunks may not be overlapping correctly")
    
    if issues:
        print("⚠️  Validation issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ All chunks validated successfully!")


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