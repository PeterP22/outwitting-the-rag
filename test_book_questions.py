#!/usr/bin/env python3
"""
Test RAG with key questions about Outwitting the Devil
"""

from query_rag import RAGQueryEngine
import time

# Key questions from Perplexity research
questions = [
    "What does Napoleon Hill mean by 'drifting' and how can I recognize if I am drifting?",
    "How does fear serve as a tool of the Devil according to the book?",
    "What are the six basic fears the Devil uses to control people?",
    "What practical steps can someone take to develop definiteness of purpose?",
    "How does hypnotic rhythm influence habits and what strategies can change negative patterns?",
    "Why is a lack of clear goals dangerous according to the Devil's confession?",
    "What role does the education system play in creating drifters?",
    "How can I distinguish between drifters and non-drifters in everyday life?",
    "What are the consequences of failing to develop courage and initiative?",
    "How does the book describe the power of beliefs in practical daily action?"
]

def test_question(engine, question, num):
    """Test a single question"""
    print(f"\n{'='*80}")
    print(f"Question {num}: {question}")
    print('='*80)
    
    start_time = time.time()
    response = engine.query(question, verbose=True)
    elapsed = time.time() - start_time
    
    print(f"\n📖 ANSWER:")
    print("-"*40)
    print(response['answer'])
    
    print(f"\n📚 Sources: {', '.join('Pages ' + s['pages'] for s in response['sources'])}")
    print(f"⏱️  Total time: {elapsed:.2f}s")
    
    # Add a short pause between queries to not overwhelm Ollama
    time.sleep(2)
    
    return response

def main():
    print("Testing RAG System with Key Book Questions")
    print("="*80)
    print(f"Testing {len(questions)} questions with top_k=8 chunks")
    
    # Initialize engine
    engine = RAGQueryEngine()
    
    # Test a subset of questions (you can test all if you have time)
    test_subset = questions[:5]  # Test first 5 questions
    
    results = []
    for i, question in enumerate(test_subset, 1):
        try:
            result = test_question(engine, question, i)
            results.append({
                'question': question,
                'answer': result['answer'],
                'sources': [s['pages'] for s in result['sources']],
                'chunks_used': result['metadata'].get('chunks_used', 0)
            })
        except Exception as e:
            print(f"\n❌ Error with question {i}: {str(e)}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Questions tested: {len(results)}/{len(test_subset)}")
    print(f"Average chunks used: {sum(r['chunks_used'] for r in results)/len(results):.1f}")
    
    # Quality check - do answers mention key concepts?
    key_concepts = ['drifting', 'fear', 'hypnotic rhythm', 'definiteness', 'purpose']
    for result in results:
        concepts_found = [c for c in key_concepts if c in result['answer'].lower()]
        if concepts_found:
            print(f"\n✅ Key concepts found: {', '.join(concepts_found)}")

if __name__ == "__main__":
    main()