#!/usr/bin/env python3
"""
RAG Query Interface for Outwitting the Devil
============================================
This script provides a complete RAG pipeline:
1. Takes user query
2. Retrieves relevant chunks
3. Generates answer using Ollama (Gemma3)
4. Includes source citations
"""

import os
import sys
import json
import time
import requests
import argparse
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import warnings
warnings.filterwarnings("ignore")

# Configuration
COLLECTION_NAME = "outwitting_the_devil"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
QDRANT_PATH = "./qdrant_db"
OLLAMA_MODEL = "gemma3:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

# RAG Parameters
DEFAULT_TOP_K = 8
SCORE_THRESHOLD = 0.98  # Keep chunks within 2% of top score
MAX_CONTEXT_LENGTH = 3000  # Characters to send to LLM


class RAGQueryEngine:
    """Main RAG query engine"""
    
    def __init__(self):
        """Initialize the RAG components"""
        print("Initializing RAG Query Engine...")
        
        # Load embedding model
        print("Loading embedding model...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize Qdrant client
        print("Connecting to vector database...")
        self.qdrant = QdrantClient(path=QDRANT_PATH)
        
        # Verify collection exists
        try:
            info = self.qdrant.get_collection(COLLECTION_NAME)
            print(f"✓ Connected to collection with {info.points_count} chunks")
        except Exception as e:
            print(f"✗ Error: Collection '{COLLECTION_NAME}' not found. Run generate_embeddings.py first.")
            sys.exit(1)
    
    def retrieve_chunks(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
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
            limit=top_k * 2,  # Get extra to filter
            with_payload=True
        )
        
        # Filter by score threshold
        if results:
            top_score = results[0].score
            threshold = top_score * SCORE_THRESHOLD
            filtered_results = [r for r in results if r.score >= threshold][:top_k]
        else:
            filtered_results = []
        
        # Format results
        chunks = []
        for result in filtered_results:
            chunks.append({
                'chunk_id': result.payload['chunk_id'],
                'text': result.payload['text'],
                'pages': f"{result.payload['start_page']}-{result.payload['end_page']}",
                'score': result.score
            })
        
        return chunks
    
    def generate_answer(self, query: str, chunks: List[Dict]) -> Tuple[str, Dict]:
        """
        Generate answer using Ollama
        
        Args:
            query: User's question
            chunks: Retrieved chunks
            
        Returns:
            Tuple of (answer, metadata)
        """
        # Build context from chunks
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(chunks):
            chunk_text = f"[Pages {chunk['pages']}]: {chunk['text']}"
            chunk_length = len(chunk_text)
            
            # Check if adding this chunk would exceed limit
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
                    'context_length': total_length
                }
                
                return answer, metadata
            else:
                return f"Error: Ollama returned status {response.status_code}", {}
                
        except requests.exceptions.Timeout:
            return "Error: Request to Ollama timed out", {}
        except Exception as e:
            return f"Error: {str(e)}", {}
    
    def query(self, question: str, top_k: int = DEFAULT_TOP_K, verbose: bool = False) -> Dict:
        """
        Complete RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            verbose: Whether to print detailed info
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Retrieve chunks
        if verbose:
            print(f"\nRetrieving top {top_k} chunks...")
        
        chunks = self.retrieve_chunks(question, top_k)
        
        if not chunks:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'chunks': [],
                'metadata': {'error': 'No relevant chunks found'}
            }
        
        if verbose:
            print(f"Retrieved {len(chunks)} relevant chunks")
            for i, chunk in enumerate(chunks):
                print(f"  {i+1}. Pages {chunk['pages']} (score: {chunk['score']:.4f})")
        
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
                    'preview': chunk['text'][:150] + "..."
                }
                for chunk in chunks
            ],
            'metadata': {
                **metadata,
                'total_time': time.time() - start_time,
                'chunks_retrieved': len(chunks)
            }
        }
        
        return response


def interactive_mode(engine: RAGQueryEngine):
    """Run in interactive mode"""
    print("\n" + "="*60)
    print("RAG Query Interface - Interactive Mode")
    print("="*60)
    print("Ask questions about 'Outwitting the Devil'")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for options")
    print("="*60)
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Your question: ").strip()
            
            # Check for commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            elif question.lower() == 'help':
                print("\nOptions:")
                print("  - Type any question about the book")
                print("  - Use --verbose before question for detailed info")
                print("  - Type 'quit' to exit")
                continue
            elif not question:
                continue
            
            # Check for verbose flag
            verbose = False
            if question.startswith("--verbose "):
                verbose = True
                question = question[10:]
            
            # Process query
            print("\n🔍 Searching for relevant information...")
            response = engine.query(question, verbose=verbose)
            
            # Display answer
            print("\n📖 Answer:")
            print("-" * 40)
            print(response['answer'])
            
            # Display sources
            print("\n📚 Sources:")
            for i, source in enumerate(response['sources']):
                print(f"  [{i+1}] Pages {source['pages']}")
            
            # Display timing
            print(f"\n⏱️  Response time: {response['metadata']['total_time']:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Query the Outwitting the Devil RAG system')
    parser.add_argument('question', nargs='?', help='Question to ask (if not provided, runs in interactive mode)')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K, help='Number of chunks to retrieve')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = RAGQueryEngine()
    print("✓ RAG engine ready!")
    
    # Run in appropriate mode
    if args.question:
        # Single query mode
        response = engine.query(args.question, top_k=args.top_k, verbose=args.verbose)
        
        if args.json:
            print(json.dumps(response, indent=2))
        else:
            print(f"\n📖 Answer: {response['answer']}")
            print(f"\n📚 Based on pages: {', '.join(s['pages'] for s in response['sources'])}")
    else:
        # Interactive mode
        interactive_mode(engine)


if __name__ == "__main__":
    main()