#!/usr/bin/env python3
"""
Build BM25 Index for Hybrid Search
==================================
Creates a BM25 index from the existing chunks for keyword-based retrieval.
"""

import json
import pickle
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text for BM25: lowercase, tokenize, remove stopwords"""
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [
        token for token in tokens 
        if token not in string.punctuation and token not in stop_words
    ]
    
    return tokens

def build_bm25_index():
    """Build BM25 index from chunks"""
    print("Loading chunks...")
    
    # Load chunks
    chunks = []
    with open('processed_text/book_chunks.jsonl', 'r') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Preprocess all chunks
    print("Preprocessing text for BM25...")
    tokenized_chunks = []
    for i, chunk in enumerate(chunks):
        if i % 20 == 0:
            print(f"  Processing chunk {i}/{len(chunks)}...")
        tokens = preprocess_text(chunk['text'])
        tokenized_chunks.append(tokens)
    
    # Build BM25 index
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Save the index and metadata
    print("Saving BM25 index...")
    bm25_data = {
        'bm25': bm25,
        'chunks': chunks,
        'tokenized_chunks': tokenized_chunks
    }
    
    with open('processed_text/bm25_index.pkl', 'wb') as f:
        pickle.dump(bm25_data, f)
    
    print("✓ BM25 index saved to processed_text/bm25_index.pkl")
    
    # Test the index
    print("\nTesting BM25 index...")
    test_queries = [
        "hypnotic rhythm",
        "six basic fears",
        "drifting",
        "definiteness of purpose"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_tokens = preprocess_text(query)
        scores = bm25.get_scores(query_tokens)
        
        # Get top 3 results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        
        for rank, idx in enumerate(top_indices):
            chunk = chunks[idx]
            score = scores[idx]
            preview = chunk['text'][:100] + "..."
            print(f"  {rank+1}. Score: {score:.2f} | Pages {chunk['start_page']}-{chunk['end_page']}")
            print(f"     {preview}")

if __name__ == "__main__":
    build_bm25_index()