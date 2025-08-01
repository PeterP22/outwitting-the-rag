#!/usr/bin/env python3
"""
Debug Query Process - Detailed Step-by-Step RAG Query
=====================================================
This script runs a single query through the RAG pipeline and logs
every step to a JSON file for learning purposes.
"""

import json
import time
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Configuration
QUERY = "What is hypnotic rhythm?"
COLLECTION_NAME = "outwitting_the_devil"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
QDRANT_PATH = "./qdrant_db"
TOP_K = 5
OUTPUT_FILE = "debug_query_process.json"


def run_query_with_logging(query: str):
    """
    Run a query through the RAG pipeline, logging each step
    """
    # Initialize process log
    process_log = {
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "query": query,
        "steps": []
    }
    
    # Step 1: Load embedding model
    step1_start = time.time()
    process_log["steps"].append({
        "step": 1,
        "name": "Load Embedding Model",
        "description": "Initialize the sentence transformer model for encoding",
        "start_time": time.time()
    })
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    process_log["steps"][-1].update({
        "end_time": time.time(),
        "duration_seconds": time.time() - step1_start,
        "model_info": {
            "name": EMBEDDING_MODEL,
            "embedding_dimension": model.get_sentence_embedding_dimension(),
            "max_sequence_length": model.max_seq_length
        }
    })
    
    # Step 2: Prepare query text
    step2_start = time.time()
    process_log["steps"].append({
        "step": 2,
        "name": "Prepare Query Text",
        "description": "Add e5 model prefix to query",
        "start_time": time.time()
    })
    
    # E5 models require "query: " prefix for queries
    prepared_query = f"query: {query}"
    
    process_log["steps"][-1].update({
        "end_time": time.time(),
        "duration_seconds": time.time() - step2_start,
        "original_query": query,
        "prepared_query": prepared_query,
        "explanation": "E5 models require 'query:' prefix for queries and 'passage:' for documents"
    })
    
    # Step 3: Generate query embedding
    step3_start = time.time()
    process_log["steps"].append({
        "step": 3,
        "name": "Generate Query Embedding",
        "description": "Convert query text to vector representation",
        "start_time": time.time()
    })
    
    query_embedding = model.encode(
        prepared_query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    process_log["steps"][-1].update({
        "end_time": time.time(),
        "duration_seconds": time.time() - step3_start,
        "embedding_shape": str(query_embedding.shape),
        "embedding_stats": {
            "min": float(np.min(query_embedding)),
            "max": float(np.max(query_embedding)),
            "mean": float(np.mean(query_embedding)),
            "norm": float(np.linalg.norm(query_embedding))
        },
        "first_10_values": query_embedding[:10].tolist()
    })
    
    # Step 4: Initialize Qdrant client
    step4_start = time.time()
    process_log["steps"].append({
        "step": 4,
        "name": "Initialize Vector Database",
        "description": "Connect to Qdrant vector database",
        "start_time": time.time()
    })
    
    client = QdrantClient(path=QDRANT_PATH)
    collection_info = client.get_collection(COLLECTION_NAME)
    
    process_log["steps"][-1].update({
        "end_time": time.time(),
        "duration_seconds": time.time() - step4_start,
        "database_path": QDRANT_PATH,
        "collection_name": COLLECTION_NAME,
        "collection_stats": {
            "total_points": collection_info.points_count,
            "vector_dimension": collection_info.config.params.vectors.size,
            "distance_metric": str(collection_info.config.params.vectors.distance)
        }
    })
    
    # Step 5: Perform vector search
    step5_start = time.time()
    process_log["steps"].append({
        "step": 5,
        "name": "Vector Similarity Search",
        "description": "Find most similar chunks using cosine similarity",
        "start_time": time.time()
    })
    
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=TOP_K,
        with_payload=True
    )
    
    process_log["steps"][-1].update({
        "end_time": time.time(),
        "duration_seconds": time.time() - step5_start,
        "search_params": {
            "top_k": TOP_K,
            "distance_metric": "cosine"
        },
        "num_results": len(search_results)
    })
    
    # Step 6: Process search results
    step6_start = time.time()
    process_log["steps"].append({
        "step": 6,
        "name": "Process Search Results",
        "description": "Extract and format retrieved chunks",
        "start_time": time.time()
    })
    
    retrieved_chunks = []
    for i, result in enumerate(search_results):
        chunk_data = {
            "rank": i + 1,
            "score": result.score,
            "chunk_id": result.payload['chunk_id'],
            "pages": f"{result.payload['start_page']}-{result.payload['end_page']}",
            "word_count": result.payload['word_count'],
            "preview": result.payload['preview'],
            "full_text": result.payload['text'][:500] + "..." if len(result.payload['text']) > 500 else result.payload['text']
        }
        retrieved_chunks.append(chunk_data)
    
    process_log["steps"][-1].update({
        "end_time": time.time(),
        "duration_seconds": time.time() - step6_start,
        "retrieved_chunks": retrieved_chunks
    })
    
    # Step 7: Analyze retrieval quality
    step7_start = time.time()
    process_log["steps"].append({
        "step": 7,
        "name": "Analyze Retrieval Quality",
        "description": "Check if retrieved chunks contain relevant information",
        "start_time": time.time()
    })
    
    # Simple keyword analysis
    query_keywords = query.lower().split()
    relevance_analysis = []
    
    for chunk in retrieved_chunks[:3]:  # Analyze top 3
        text_lower = chunk['full_text'].lower()
        keyword_counts = {
            keyword: text_lower.count(keyword) 
            for keyword in query_keywords
        }
        
        # Check for exact phrase
        exact_phrase_count = text_lower.count("hypnotic rhythm")
        
        relevance_analysis.append({
            "chunk_id": chunk['chunk_id'],
            "keyword_counts": keyword_counts,
            "exact_phrase_count": exact_phrase_count,
            "total_keyword_mentions": sum(keyword_counts.values())
        })
    
    process_log["steps"][-1].update({
        "end_time": time.time(),
        "duration_seconds": time.time() - step7_start,
        "query_keywords": query_keywords,
        "relevance_analysis": relevance_analysis
    })
    
    # Summary
    process_log["summary"] = {
        "total_duration_seconds": sum(step["duration_seconds"] for step in process_log["steps"]),
        "top_result": {
            "chunk_id": retrieved_chunks[0]["chunk_id"],
            "score": retrieved_chunks[0]["score"],
            "preview": retrieved_chunks[0]["preview"]
        },
        "relevance_assessment": "High" if relevance_analysis[0]["exact_phrase_count"] > 0 else "Medium" if relevance_analysis[0]["total_keyword_mentions"] > 2 else "Low"
    }
    
    return process_log


def main():
    """Main execution"""
    print(f"Running debug query: '{QUERY}'")
    print("="*60)
    
    # Run query with detailed logging
    process_log = run_query_with_logging(QUERY)
    
    # Save to JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(process_log, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Query process logged to: {OUTPUT_FILE}")
    
    # Print summary
    print("\nQUERY RESULTS SUMMARY:")
    print("-"*40)
    print(f"Query: {QUERY}")
    print(f"Total time: {process_log['summary']['total_duration_seconds']:.3f} seconds")
    print(f"\nTop result (score: {process_log['summary']['top_result']['score']:.4f}):")
    print(f"Chunk: {process_log['summary']['top_result']['chunk_id']}")
    print(f"Preview: {process_log['summary']['top_result']['preview']}")
    
    # Print relevance info
    top_relevance = process_log['steps'][6]['relevance_analysis'][0]
    print(f"\nRelevance check:")
    print(f"- Exact phrase 'hypnotic rhythm' count: {top_relevance['exact_phrase_count']}")
    print(f"- Individual keyword mentions: {top_relevance['keyword_counts']}")
    print(f"- Assessment: {process_log['summary']['relevance_assessment']}")
    
    # Print all top 3 previews for manual inspection
    print("\n" + "="*60)
    print("TOP 3 RETRIEVED CHUNKS (for manual inspection):")
    print("="*60)
    
    for chunk in process_log['steps'][5]['retrieved_chunks'][:3]:
        print(f"\n{chunk['rank']}. Chunk {chunk['chunk_id']} (score: {chunk['score']:.4f}, pages: {chunk['pages']})")
        print(f"Preview: {chunk['preview']}")
        print(f"Full text sample: {chunk['full_text'][:200]}...")


if __name__ == "__main__":
    main()