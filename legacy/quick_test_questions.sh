#!/bin/bash
# Quick test script for key questions about Outwitting the Devil

echo "Testing RAG with Key Questions from the Book"
echo "============================================"
echo "Now using top_k=8 for better coverage"
echo ""

# Activate virtual environment
source venv/bin/activate

# Test key questions one by one
echo "1. Testing 'drifting' concept..."
python query_rag.py "What does Napoleon Hill mean by drifting and how can I recognize if I am drifting?" --verbose

echo -e "\n\n2. Testing 'six fears' concept..."
python query_rag.py "What are the six basic fears the Devil uses to control people?"

echo -e "\n\n3. Testing 'hypnotic rhythm' concept..."
python query_rag.py "How does hypnotic rhythm influence habits and what strategies can change negative patterns?"

echo -e "\n\n4. Testing 'definiteness of purpose'..."
python query_rag.py "What practical steps can someone take to develop definiteness of purpose?"

echo -e "\n\n5. Testing 'education system' critique..."
python query_rag.py "What role does the education system play in creating drifters?"