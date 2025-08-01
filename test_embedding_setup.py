#!/usr/bin/env python3
"""
Test script to verify embedding model and Qdrant setup
"""

import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

print("Testing embedding and vector store setup...")
print("=" * 50)

# Test 1: Check if model can be loaded
print("\n1. Testing embedding model...")
try:
    model_name = "intfloat/e5-large-v2"
    print(f"   Loading {model_name}...")
    print("   (This may take a few minutes on first run as model downloads)")
    
    model = SentenceTransformer(model_name)
    print(f"   ✓ Model loaded successfully!")
    print(f"   - Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"   - Max sequence length: {model.max_seq_length}")
    
    # Test embedding generation
    test_text = "passage: What is the secret to success?"
    embedding = model.encode(test_text)
    print(f"   ✓ Test embedding generated: shape {embedding.shape}")
    
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    sys.exit(1)

# Test 2: Check Qdrant
print("\n2. Testing Qdrant setup...")
try:
    client = QdrantClient(path="./test_qdrant")
    print("   ✓ Qdrant client initialized")
    
    # Try to create a test collection
    from qdrant_client.models import Distance, VectorParams
    
    test_collection = "test_collection"
    client.recreate_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print(f"   ✓ Test collection '{test_collection}' created")
    
    # Clean up
    client.delete_collection(test_collection)
    print("   ✓ Test collection deleted")
    
except Exception as e:
    print(f"   ✗ Error with Qdrant: {e}")
    sys.exit(1)

print("\n✅ All tests passed! Ready to generate embeddings.")
print("\nNote: The full embedding generation may take 5-10 minutes on first run.")