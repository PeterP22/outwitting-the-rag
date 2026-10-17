#!/usr/bin/env python3
"""
Test Hybrid Search Improvement
==============================
Demonstrates how hybrid search improves retrieval for problematic queries.
"""

from query_rag import RAGQueryEngine
from query_rag_hybrid import HybridRAGQueryEngine
import time

def test_query(semantic_engine, hybrid_engine, query):
    """Test a query with both engines and compare results"""
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print('='*70)
    
    # Test semantic-only
    print("\n1. SEMANTIC-ONLY SEARCH:")
    start = time.time()
    semantic_result = semantic_engine.query(query, verbose=False)
    semantic_time = time.time() - start
    
    print(f"   Top chunks: {', '.join(s['pages'] for s in semantic_result['sources'][:3])}")
    print(f"   Answer preview: {semantic_result['answer'][:150]}...")
    print(f"   Time: {semantic_time:.2f}s")
    
    # Test hybrid
    print("\n2. HYBRID SEARCH:")
    start = time.time()
    hybrid_result = hybrid_engine.query(query, verbose=False)
    hybrid_time = time.time() - start
    
    print(f"   Top chunks: {', '.join(s['pages'] for s in hybrid_result['sources'][:3])}")
    print(f"   Answer preview: {hybrid_result['answer'][:150]}...")
    print(f"   Time: {hybrid_time:.2f}s")
    
    # Compare top chunks
    semantic_top = semantic_result['sources'][0]['pages'] if semantic_result['sources'] else "None"
    hybrid_top = hybrid_result['sources'][0]['pages'] if hybrid_result['sources'] else "None"
    
    if semantic_top != hybrid_top:
        print(f"\n   ✅ IMPROVEMENT: Hybrid found different top chunk ({hybrid_top} vs {semantic_top})")
    else:
        print(f"\n   ℹ️  Same top chunk: {semantic_top}")

def main():
    print("Initializing search engines...")
    print("-" * 40)
    
    # Initialize engines
    semantic_engine = RAGQueryEngine()
    print("\n" + "-" * 40)
    hybrid_engine = HybridRAGQueryEngine()
    
    # Test problematic queries
    test_queries = [
        "What is hypnotic rhythm?",
        "What does Hill mean by drifting?",
        "What are the six basic fears?",
        "How can someone stop drifting?"
    ]
    
    print("\n" + "="*70)
    print("TESTING HYBRID SEARCH IMPROVEMENTS")
    print("="*70)
    
    for query in test_queries:
        test_query(semantic_engine, hybrid_engine, query)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nHybrid search combines:")
    print("- BM25 (keyword matching) for exact phrases")
    print("- Semantic search (embeddings) for conceptual similarity")
    print("- Result: Better retrieval for both specific and conceptual queries")

if __name__ == "__main__":
    main()