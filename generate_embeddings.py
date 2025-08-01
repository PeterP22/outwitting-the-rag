#!/usr/bin/env python3
"""
Generate Embeddings and Store in Qdrant
=======================================
This script generates embeddings for our text chunks using the
intfloat/e5-large-v2 model and stores them in Qdrant vector database.

Model: intfloat/e5-large-v2
- Open-source, free to use
- 1024-dimensional embeddings
- Good performance for general English text
- ~1.3GB model size

Vector Store: Qdrant
- Lightweight, fast
- Simple Python API
- Local storage (no server needed for development)
"""

import os
import json
import time
from typing import List, Dict
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

# Embedding model imports
from sentence_transformers import SentenceTransformer

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

# Configuration
CHUNKS_FILE = "processed_text/book_chunks.jsonl"
COLLECTION_NAME = "outwitting_the_devil"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
VECTOR_SIZE = 1024  # e5-large-v2 produces 1024-dim vectors
BATCH_SIZE = 32  # Process chunks in batches for efficiency
QDRANT_PATH = "./qdrant_db"  # Local storage


def load_chunks(file_path: str) -> List[Dict]:
    """
    Load chunks from JSONL file
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line.strip())
            chunks.append(chunk)
    
    print(f"Loaded {len(chunks)} chunks from {file_path}")
    return chunks


def prepare_texts_for_e5(chunks: List[Dict]) -> List[str]:
    """
    Prepare texts for e5 model embedding
    
    E5 models expect a prefix:
    - "query: " for queries
    - "passage: " for documents
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of prepared texts
    """
    # For indexing, we use "passage: " prefix
    prepared_texts = []
    for chunk in chunks:
        # Add passage prefix for e5 model
        prepared_text = f"passage: {chunk['text']}"
        prepared_texts.append(prepared_text)
    
    return prepared_texts


def generate_embeddings(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts
    
    Args:
        texts: List of texts to embed
        model_name: Name of the sentence-transformer model
        batch_size: Batch size for processing
        
    Returns:
        Numpy array of embeddings
    """
    print(f"\nLoading embedding model: {model_name}")
    print("This may take a few minutes on first run as the model downloads...")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Verify model properties
    print(f"Model loaded successfully!")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"Max sequence length: {model.max_seq_length}")
    
    # Generate embeddings
    print(f"\nGenerating embeddings for {len(texts)} texts...")
    
    # Process in batches with progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Important for cosine similarity
    )
    
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    return embeddings


def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create or recreate Qdrant collection
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection
        vector_size: Dimension of vectors
    """
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(col.name == collection_name for col in collections)
    
    if exists:
        print(f"Collection '{collection_name}' already exists. Deleting...")
        client.delete_collection(collection_name)
    
    # Create collection
    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE  # Using cosine similarity
        )
    )
    print("Collection created successfully!")


def store_embeddings_in_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict],
    embeddings: np.ndarray
):
    """
    Store embeddings and metadata in Qdrant
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection
        chunks: List of chunk dictionaries
        embeddings: Numpy array of embeddings
    """
    print(f"\nStoring {len(chunks)} embeddings in Qdrant...")
    
    # Prepare points for Qdrant
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Create point with embedding and metadata
        point = PointStruct(
            id=i,  # Using index as ID
            vector=embedding.tolist(),
            payload={
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "preview": chunk["preview"],
                "word_count": chunk["word_count"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
                "position": chunk["position"],
                "timestamp": chunk["timestamp"]
            }
        )
        points.append(point)
    
    # Upload in batches
    batch_size = 100
    for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
    
    print("Successfully stored all embeddings!")


def verify_storage(client: QdrantClient, collection_name: str, sample_size: int = 5):
    """
    Verify that embeddings were stored correctly
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection
        sample_size: Number of samples to check
    """
    print(f"\nVerifying storage...")
    
    # Get collection info
    collection_info = client.get_collection(collection_name)
    print(f"Collection '{collection_name}':")
    print(f"  - Total points: {collection_info.points_count}")
    print(f"  - Vector size: {collection_info.config.params.vectors.size}")
    print(f"  - Distance metric: {collection_info.config.params.vectors.distance}")
    
    # Sample a few points
    print(f"\nSampling {sample_size} points:")
    points = client.retrieve(
        collection_name=collection_name,
        ids=list(range(sample_size)),
        with_payload=True,
        with_vectors=False  # Don't need vectors for verification
    )
    
    for point in points:
        print(f"\n  Point ID: {point.id}")
        print(f"  Chunk ID: {point.payload['chunk_id']}")
        print(f"  Pages: {point.payload['start_page']}-{point.payload['end_page']}")
        print(f"  Preview: {point.payload['preview'][:80]}...")


def test_search(client: QdrantClient, collection_name: str, query: str, top_k: int = 5):
    """
    Test search functionality
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection
        query: Search query
        top_k: Number of results to return
    """
    print(f"\n{'='*60}")
    print(f"TEST SEARCH: '{query}'")
    print('='*60)
    
    # Load model for query embedding
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Prepare query with e5 prefix
    prepared_query = f"query: {query}"
    
    # Generate query embedding
    query_embedding = model.encode(
        prepared_query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Search
    start_time = time.time()
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        with_payload=True
    )
    search_time = time.time() - start_time
    
    print(f"\nSearch completed in {search_time:.3f} seconds")
    print(f"Top {top_k} results:\n")
    
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.score:.4f}")
        print(f"   Chunk: {result.payload['chunk_id']}")
        print(f"   Pages: {result.payload['start_page']}-{result.payload['end_page']}")
        print(f"   Preview: {result.payload['preview']}")
        print()


def main():
    """Main execution function"""
    
    # Check if chunks file exists
    if not os.path.exists(CHUNKS_FILE):
        print(f"Error: Chunks file not found: {CHUNKS_FILE}")
        print("Please run chunk_text_simple.py first.")
        return
    
    # Load chunks
    chunks = load_chunks(CHUNKS_FILE)
    
    # Prepare texts for e5 model
    prepared_texts = prepare_texts_for_e5(chunks)
    
    # Generate embeddings
    embeddings = generate_embeddings(prepared_texts, EMBEDDING_MODEL, BATCH_SIZE)
    
    # Initialize Qdrant client (local storage)
    print(f"\nInitializing Qdrant client with local storage: {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH)
    
    # Setup collection
    setup_qdrant_collection(client, COLLECTION_NAME, VECTOR_SIZE)
    
    # Store embeddings
    store_embeddings_in_qdrant(client, COLLECTION_NAME, chunks, embeddings)
    
    # Verify storage
    verify_storage(client, COLLECTION_NAME)
    
    # Test searches
    test_queries = [
        "What is hypnotic rhythm?",
        "How can we avoid drifting?",
        "What role does fear play according to the Devil?",
        "What is the secret to freedom and success?"
    ]
    
    for query in test_queries:
        test_search(client, COLLECTION_NAME, query, top_k=3)
    
    print("\n✅ Embedding generation and storage complete!")
    print(f"\nVector database location: {QDRANT_PATH}")
    print("You can now use this for RAG queries!")


if __name__ == "__main__":
    main()