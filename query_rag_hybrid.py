#!/usr/bin/env python3
"""
Hybrid RAG Query Interface for Outwitting the Devil
===================================================
Combines BM25 (keyword) and semantic (vector) search for better retrieval.
"""

import os
import sys
import json
import time
import pickle
import requests
import argparse
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import warnings
warnings.filterwarnings("ignore")

# Configuration
COLLECTION_NAME = "outwitting_the_devil"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
QDRANT_PATH = "./qdrant_db"
BM25_INDEX_PATH = "./processed_text/bm25_index.pkl"
OLLAMA_MODEL = "gemma3:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Hybrid Search Parameters
HYBRID_ALPHA = 0.5  # Balance between BM25 (0) and semantic (1)
DEFAULT_TOP_K = 8
MAX_CONTEXT_LENGTH = 3000


class HybridRAGQueryEngine:
    """Hybrid RAG query engine combining BM25 and semantic search"""
    
    def __init__(self):
        """Initialize the RAG components"""
        print("Initializing Hybrid RAG Query Engine...")
        
        # Load embedding model
        print("Loading embedding model...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize Qdrant client
        print("Connecting to vector database...")
        self.qdrant = QdrantClient(path=QDRANT_PATH)
        
        # Load BM25 index
        print("Loading BM25 index...")
        with open(BM25_INDEX_PATH, 'rb') as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data['bm25']
            self.chunks = bm25_data['chunks']
            self.tokenized_chunks = bm25_data['tokenized_chunks']
        
        print(f"✓ Loaded {len(self.chunks)} chunks")
        
        # Stopwords for query preprocessing
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess query for BM25"""
        query = query.lower()
        tokens = word_tokenize(query)
        tokens = [
            token for token in tokens 
            if token not in string.punctuation and token not in self.stop_words
        ]
        return tokens
    
    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Perform BM25 keyword search"""
        query_tokens = self.preprocess_query(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices with scores
        top_indices = sorted(
            enumerate(scores), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [(idx, score) for idx, score in top_indices if score > 0]
    
    def semantic_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Perform semantic vector search"""
        # Prepare query for e5 model
        prepared_query = f"query: {query}"
        
        # Generate query embedding
        query_embedding = self.encoder.encode(
            prepared_query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in Qdrant
        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        
        return [(r.payload['chunk_id'], r.score) for r in results]
    
    def hybrid_search(self, query: str, top_k: int = DEFAULT_TOP_K, alpha: float = HYBRID_ALPHA) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and semantic results
        
        Args:
            query: User's question
            top_k: Number of final results to return
            alpha: Weight for semantic search (0=pure BM25, 1=pure semantic)
        """
        # Get results from both searches
        bm25_results = self.bm25_search(query, top_k=top_k*2)
        semantic_results = self.semantic_search(query, top_k=top_k*2)
        
        # Create score dictionaries
        bm25_scores = {}
        for idx, score in bm25_results:
            chunk_id = self.chunks[idx]['chunk_id']
            # Normalize BM25 scores to 0-1 range
            normalized_score = min(score / 10.0, 1.0)  # Empirical normalization
            bm25_scores[chunk_id] = normalized_score
        
        semantic_scores = dict(semantic_results)
        
        # Reciprocal Rank Fusion
        combined_scores = {}
        
        # Add BM25 scores
        for rank, (idx, _) in enumerate(bm25_results):
            chunk_id = self.chunks[idx]['chunk_id']
            combined_scores[chunk_id] = (1 - alpha) / (rank + 1)
        
        # Add semantic scores
        for rank, (chunk_id, _) in enumerate(semantic_results):
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += alpha / (rank + 1)
            else:
                combined_scores[chunk_id] = alpha / (rank + 1)
        
        # Sort by combined score
        sorted_chunks = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Build final results with metadata
        results = []
        for chunk_id, hybrid_score in sorted_chunks:
            # Find chunk data
            chunk_data = next((c for c in self.chunks if c['chunk_id'] == chunk_id), None)
            if chunk_data:
                results.append({
                    'chunk_id': chunk_id,
                    'text': chunk_data['text'],
                    'pages': f"{chunk_data['start_page']}-{chunk_data['end_page']}",
                    'hybrid_score': hybrid_score,
                    'bm25_score': bm25_scores.get(chunk_id, 0),
                    'semantic_score': semantic_scores.get(chunk_id, 0)
                })
        
        return results
    
    def generate_answer(self, query: str, chunks: List[Dict]) -> Tuple[str, Dict]:
        """Generate answer using Ollama"""
        # Build context from chunks
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(chunks):
            chunk_text = f"[Pages {chunk['pages']}]: {chunk['text']}"
            chunk_length = len(chunk_text)
            
            if total_length + chunk_length > MAX_CONTEXT_LENGTH and i > 0:
                break
                
            context_parts.append(chunk_text)
            total_length += chunk_length
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""Based on the following passages from "Outwitting the Devil" by Napoleon Hill:

{context}

Question: {query}

Instructions:
1. Answer based ONLY on the provided passages
2. Be clear and concise
3. Cite page numbers when referencing specific information
4. If the passages don't contain enough information to answer fully, say so

Answer:"""

        # Call Ollama
        start_time = time.time()
        
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    'model': OLLAMA_MODEL,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'max_tokens': 300
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                metadata = {
                    'model': OLLAMA_MODEL,
                    'response_time': time.time() - start_time,
                    'chunks_used': len(context_parts),
                    'context_length': total_length,
                    'search_type': 'hybrid'
                }
                
                return answer, metadata
            else:
                return f"Error: Ollama returned status {response.status_code}", {}
                
        except Exception as e:
            return f"Error: {str(e)}", {}
    
    def query(self, question: str, top_k: int = DEFAULT_TOP_K, alpha: float = HYBRID_ALPHA, verbose: bool = False) -> Dict:
        """
        Complete hybrid RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            alpha: Hybrid search weight (0=BM25, 1=semantic)
            verbose: Whether to print detailed info
        """
        start_time = time.time()
        
        # Perform hybrid search
        if verbose:
            print(f"\nPerforming hybrid search (alpha={alpha})...")
        
        chunks = self.hybrid_search(question, top_k, alpha)
        
        if not chunks:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'chunks': [],
                'metadata': {'error': 'No relevant chunks found'}
            }
        
        if verbose:
            print(f"Retrieved {len(chunks)} chunks via hybrid search")
            for i, chunk in enumerate(chunks):
                print(f"  {i+1}. Pages {chunk['pages']}")
                print(f"     Hybrid: {chunk['hybrid_score']:.3f} | "
                      f"BM25: {chunk['bm25_score']:.3f} | "
                      f"Semantic: {chunk['semantic_score']:.3f}")
        
        # Generate answer
        if verbose:
            print(f"\nGenerating answer with {OLLAMA_MODEL}...")
        
        answer, metadata = self.generate_answer(question, chunks)
        
        # Prepare response
        response = {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'pages': chunk['pages'],
                    'preview': chunk['text'][:150] + "...",
                    'scores': {
                        'hybrid': chunk['hybrid_score'],
                        'bm25': chunk['bm25_score'],
                        'semantic': chunk['semantic_score']
                    }
                }
                for chunk in chunks
            ],
            'metadata': {
                **metadata,
                'total_time': time.time() - start_time,
                'chunks_retrieved': len(chunks),
                'alpha': alpha
            }
        }
        
        return response


def compare_search_methods(engine: HybridRAGQueryEngine, query: str):
    """Compare pure semantic, pure BM25, and hybrid search"""
    print(f"\n{'='*60}")
    print(f"Comparing search methods for: '{query}'")
    print('='*60)
    
    # Pure semantic (alpha=1)
    print("\n1. PURE SEMANTIC SEARCH (alpha=1.0)")
    semantic_results = engine.query(query, alpha=1.0, verbose=True)
    
    # Pure BM25 (alpha=0)
    print("\n2. PURE BM25 SEARCH (alpha=0.0)")
    bm25_results = engine.query(query, alpha=0.0, verbose=True)
    
    # Hybrid (alpha=0.5)
    print("\n3. HYBRID SEARCH (alpha=0.5)")
    hybrid_results = engine.query(query, alpha=0.5, verbose=True)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for method, results in [("Semantic", semantic_results), 
                            ("BM25", bm25_results), 
                            ("Hybrid", hybrid_results)]:
        print(f"\n{method}:")
        print(f"  Top chunk pages: {results['sources'][0]['pages'] if results['sources'] else 'None'}")
        print(f"  Answer preview: {results['answer'][:100]}...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Hybrid RAG system with BM25 + semantic search')
    parser.add_argument('question', nargs='?', help='Question to ask')
    parser.add_argument('--alpha', type=float, default=0.5, help='Hybrid weight (0=BM25, 1=semantic)')
    parser.add_argument('--compare', action='store_true', help='Compare all search methods')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = HybridRAGQueryEngine()
    print("✓ Hybrid RAG engine ready!")
    
    if args.compare and args.question:
        # Compare mode
        compare_search_methods(engine, args.question)
    elif args.question:
        # Single query mode
        response = engine.query(args.question, alpha=args.alpha, verbose=args.verbose)
        print(f"\n📖 Answer: {response['answer']}")
        print(f"\n📚 Based on pages: {', '.join(s['pages'] for s in response['sources'])}")
    else:
        print("\nUsage:")
        print("  python query_rag_hybrid.py 'Your question here'")
        print("  python query_rag_hybrid.py 'Your question' --compare")
        print("  python query_rag_hybrid.py 'Your question' --alpha 0.7 --verbose")


if __name__ == "__main__":
    main()