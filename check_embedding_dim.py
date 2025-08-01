#!/usr/bin/env python3
"""
Check the exact embedding dimension of e5-large-v2
"""

from sentence_transformers import SentenceTransformer

# Load model
model_name = "intfloat/e5-large-v2"
model = SentenceTransformer(model_name)

# Embed a dummy sentence
dummy_sentence = "passage: This is a test sentence."
embedding = model.encode(dummy_sentence)

print(f"Model: {model_name}")
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding dimension: {embedding.shape[0]}")
print(f"\nThis is the 'd' value to use in VectorParams(size=d)")

# Also check what the model reports
print(f"\nModel's reported dimension: {model.get_sentence_embedding_dimension()}")