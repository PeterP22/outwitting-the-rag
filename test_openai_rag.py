#!/usr/bin/env python3
"""
Test RAG system with OpenAI API
===============================
This script tests the RAG system using OpenAI's API instead of Ollama.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configuration
COLLECTION_NAME = "outwitting_the_devil"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
QDRANT_PATH = "./qdrant_db"
TOP_K = 8
SCORE_THRESHOLD = 0.98

class OpenAIRAGQueryEngine:
    """RAG query engine using OpenAI API"""
    
    def __init__(self):
        """Initialize the RAG components"""
        print("Initializing OpenAI RAG Query Engine...")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("LLM_MODEL", "gpt-4")
        
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
            print(f"✗ Error: Collection '{COLLECTION_NAME}' not found.")
            raise
    
    def retrieve_chunks(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Retrieve relevant chunks for a query"""
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
            limit=top_k * 2,
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
    
    def generate_answer(self, query: str, chunks: List[Dict]) -> Dict:
        """Generate answer using OpenAI API"""
        # Build context from chunks
        context_parts = []
        total_length = 0
        max_context = 3000
        
        for i, chunk in enumerate(chunks):
            chunk_text = f"[Pages {chunk['pages']}]: {chunk['text']}"
            chunk_length = len(chunk_text)
            
            if total_length + chunk_length > max_context and i > 0:
                break
                
            context_parts.append(chunk_text)
            total_length += chunk_length
        
        context = "\n\n".join(context_parts)
        
        # Create messages for chat completion
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant answering questions about 'Outwitting the Devil' by Napoleon Hill. Answer based ONLY on the provided passages and cite page numbers when referencing specific information."
            },
            {
                "role": "user",
                "content": f"""Based on the following passages from "Outwitting the Devil":

{context}

Question: {query}

Please provide a clear and concise answer based only on these passages. Include page citations when referencing specific information."""
            }
        ]
        
        # Call OpenAI API
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            metadata = {
                'model': self.model,
                'response_time': time.time() - start_time,
                'chunks_used': len(context_parts),
                'context_length': total_length,
                'tokens_used': {
                    'prompt': response.usage.prompt_tokens,
                    'completion': response.usage.completion_tokens,
                    'total': response.usage.total_tokens
                }
            }
            
            return answer, metadata
            
        except Exception as e:
            return f"Error: {str(e)}", {}


def test_queries():
    """Test several queries using OpenAI API"""
    print("\n" + "="*60)
    print("Testing RAG with OpenAI API")
    print(f"Model: {os.getenv('LLM_MODEL')}")
    print("="*60)
    
    # Initialize engine
    engine = OpenAIRAGQueryEngine()
    
    # Test queries
    test_questions = [
        "What is hypnotic rhythm and how does it work?",
        "What are the six basic fears mentioned in the book?",
        "How does Napoleon Hill define 'drifting'?",
        "What is definiteness of purpose?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print('='*60)
        
        # Retrieve chunks
        print("\nRetrieving relevant chunks...")
        chunks = engine.retrieve_chunks(question)
        
        if not chunks:
            print("No relevant chunks found!")
            continue
        
        print(f"Retrieved {len(chunks)} chunks:")
        for j, chunk in enumerate(chunks):
            print(f"  {j+1}. Pages {chunk['pages']} (score: {chunk['score']:.4f})")
        
        # Generate answer
        print("\nGenerating answer with OpenAI...")
        answer, metadata = engine.generate_answer(question, chunks)
        
        print(f"\n📖 Answer:")
        print("-"*40)
        print(answer)
        print("-"*40)
        
        # Print metadata
        print(f"\n📊 Metadata:")
        print(f"  Model: {metadata.get('model', 'N/A')}")
        print(f"  Response time: {metadata.get('response_time', 0):.2f}s")
        print(f"  Chunks used: {metadata.get('chunks_used', 0)}")
        if 'tokens_used' in metadata:
            tokens = metadata['tokens_used']
            print(f"  Tokens: {tokens['prompt']} prompt + {tokens['completion']} completion = {tokens['total']} total")
        
        # Pause between queries to respect rate limits
        time.sleep(2)


def single_query(question: str):
    """Test a single query"""
    engine = OpenAIRAGQueryEngine()
    
    print(f"\n❓ Question: {question}")
    
    # Retrieve and generate
    chunks = engine.retrieve_chunks(question)
    answer, metadata = engine.generate_answer(question, chunks)
    
    print(f"\n📖 Answer: {answer}")
    print(f"\n📚 Sources: {', '.join('Pages ' + c['pages'] for c in chunks)}")
    print(f"⏱️  Time: {metadata.get('response_time', 0):.2f}s")
    
    if 'tokens_used' in metadata:
        tokens = metadata['tokens_used']
        print(f"🔢 Tokens: {tokens['total']} total ({tokens['prompt']} prompt, {tokens['completion']} completion)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single query mode
        question = " ".join(sys.argv[1:])
        single_query(question)
    else:
        # Test multiple queries
        test_queries()