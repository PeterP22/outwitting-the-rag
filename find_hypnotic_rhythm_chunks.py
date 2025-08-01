#!/usr/bin/env python3
"""
Find chunks that actually contain hypnotic rhythm definitions
"""

import json

# Load chunks
chunks = []
with open('processed_text/book_chunks.jsonl', 'r') as f:
    for line in f:
        chunks.append(json.loads(line))

# Search for chunks with "hypnotic rhythm" and related context
print("CHUNKS CONTAINING 'HYPNOTIC RHYTHM':")
print("="*60)

hr_chunks = []
for chunk in chunks:
    text_lower = chunk['text'].lower()
    if 'hypnotic rhythm' in text_lower:
        # Count occurrences
        count = text_lower.count('hypnotic rhythm')
        
        # Check for definition indicators
        has_definition_words = any(word in text_lower for word in 
            ['what is hypnotic', 'hypnotic rhythm is', 'hypnotic rhythm?', 
             'elementary description', 'universal law'])
        
        hr_chunks.append({
            'chunk_id': chunk['chunk_id'],
            'pages': f"{chunk['start_page']}-{chunk['end_page']}",
            'count': count,
            'has_definition': has_definition_words,
            'preview': chunk['text'][:200] + "..."
        })

# Sort by count and definition presence
hr_chunks.sort(key=lambda x: (x['has_definition'], x['count']), reverse=True)

# Print top candidates
print(f"Found {len(hr_chunks)} chunks containing 'hypnotic rhythm'")
print("\nTOP DEFINITION CANDIDATES:")
print("-"*60)

for i, chunk in enumerate(hr_chunks[:10]):
    print(f"\n{i+1}. {chunk['chunk_id']} (pages: {chunk['pages']})")
    print(f"   Count: {chunk['count']}, Has definition words: {chunk['has_definition']}")
    print(f"   Preview: {chunk['preview']}")

# Now let's specifically find the Q&A about "What is hypnotic rhythm?"
print("\n\nSPECIFIC Q&A SEARCH:")
print("="*60)

for chunk in chunks:
    if "what is hypnotic rhythm?" in chunk['text'].lower():
        print(f"\nFOUND Q&A in {chunk['chunk_id']} (pages: {chunk['start_page']}-{chunk['end_page']})")
        print("Full chunk text:")
        print("-"*40)
        print(chunk['text'])
        break