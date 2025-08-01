#!/usr/bin/env python3
"""
Test the RAG system with multiple queries
"""

from query_rag import RAGQueryEngine
import json

# Test queries
test_queries = [
    "What is hypnotic rhythm?",
    "How can someone avoid drifting?",
    "What role does fear play according to the Devil?",
    "What are the principles of success mentioned in the book?",
    "How does the Devil control people?"
]

def main():
    print("Testing RAG System with Multiple Queries")
    print("="*60)
    
    # Initialize engine
    engine = RAGQueryEngine()
    
    results = []
    
    # Test each query
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*40)
        
        response = engine.query(query, top_k=3)
        
        print(f"Answer: {response['answer'][:200]}...")
        print(f"Sources: {', '.join(s['pages'] for s in response['sources'])}")
        print(f"Time: {response['metadata']['total_time']:.2f}s")
        
        results.append({
            'query': query,
            'answer_preview': response['answer'][:200],
            'sources': [s['pages'] for s in response['sources']],
            'time': response['metadata']['total_time']
        })
    
    # Save results
    with open('rag_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ Test complete! Results saved to rag_test_results.json")

if __name__ == "__main__":
    main()