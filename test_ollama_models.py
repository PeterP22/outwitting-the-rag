#!/usr/bin/env python3
"""
Test Ollama models for RAG quality
"""

import requests
import json
import time

# Test prompt that mimics our RAG use case
TEST_PROMPT = """Based on the following passages from "Outwitting the Devil" by Napoleon Hill:

[Pages 133-136]: "Q What is hypnotic rhythm? How do you use it to gain permanent mastery over human beings? A I will have to go back into time and space and give you a brief elementary description of how nature uses hypnotic rhythm. Otherwise you will not be able to understand my description of how I use this universal law to control human beings."

[Pages 158-160]: "Man, alone, has the power to establish his own rhythm of thought providing he exercises this privilege before hypnotic rhythm has forced upon him the influences of his environment."

Question: What is hypnotic rhythm according to the text?

Please provide a clear, concise answer based only on the provided passages, and mention the page numbers when citing."""

def test_model(model_name):
    """Test a specific Ollama model"""
    print(f"\nTesting {model_name}...")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_name,
                'prompt': TEST_PROMPT,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 200
                }
            }
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '')
            
            print(f"Response time: {response_time:.2f} seconds")
            print(f"Response length: {len(answer)} chars")
            print("\nAnswer:")
            print(answer[:500] + "..." if len(answer) > 500 else answer)
            
            # Simple quality checks
            quality_score = 0
            if "hypnotic rhythm" in answer.lower():
                quality_score += 1
            if any(page in answer for page in ["133", "136", "158", "160", "page"]):
                quality_score += 1
            if "nature" in answer.lower() or "universal law" in answer.lower():
                quality_score += 1
            if len(answer) > 50 and len(answer) < 400:  # Good length
                quality_score += 1
                
            print(f"\nQuality indicators: {quality_score}/4")
            
            return {
                'model': model_name,
                'time': response_time,
                'length': len(answer),
                'quality': quality_score,
                'answer': answer
            }
        else:
            print(f"Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        return None

def main():
    print("Testing Ollama models for RAG...")
    print("=" * 60)
    
    models = ['qwen3:8b', 'gemma3:latest']
    results = []
    
    for model in models:
        result = test_model(model)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for r in results:
        print(f"\n{r['model']}:")
        print(f"  - Response time: {r['time']:.2f}s")
        print(f"  - Quality score: {r['quality']}/4")
        print(f"  - Response length: {r['length']} chars")
    
    # Recommendation
    best = max(results, key=lambda x: (x['quality'], -x['time']))
    print(f"\n✅ Recommended model: {best['model']}")
    print(f"   (Best quality score with reasonable speed)")

if __name__ == "__main__":
    main()