# Outwitting the Devil - RAG Application

A Retrieval-Augmented Generation (RAG) system for Napoleon Hill's "Outwitting the Devil" book, built as a learning project to understand AI engineering fundamentals.

## Project Overview

This project implements a RAG system that can answer questions about the book's content using semantic search and large language models. It progresses from basic text extraction to advanced retrieval strategies.

## Setup

### 1. Create and activate virtual environment

```bash
# Run the setup script
./setup_environment.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
outwitting-the-rag/
├── extract_pdf_test.py      # PDF extraction quality testing
├── extraction_results/      # Output from different PDF extractors
├── requirements.txt         # Project dependencies
├── setup_environment.sh     # Environment setup script
├── rag_learning_plan.md    # Detailed learning plan and timeline
└── Hill_Napoleon_-_Outwitting_the_devil.pdf  # Source PDF
```

## Week 1 Learning Plan

### Day 1: PDF Extraction & Text Quality Check ✅
- Test different PDF extraction methods
- Analyze text quality and identify issues
- Choose best extraction approach

### Day 2: Text Chunking & Embeddings
- Implement smart text chunking
- Generate embeddings for chunks
- Store in vector database

### Day 3: RAG Chain Implementation
- Build retrieval system
- Create answer generation pipeline
- Add configuration management

### Day 4: API & Packaging
- Wrap in FastAPI
- Dockerize application
- Add hot-reload configuration

### Day 5: Evaluation & Testing
- Create test questions
- Implement evaluation metrics
- Analyze system performance

## Current Status

**Day 1 - In Progress**: Setting up PDF extraction and quality testing

## Next Steps

1. Run the extraction test:
   ```bash
   python extract_pdf_test.py
   ```

2. Review extraction results in `extraction_results/`

3. Choose the best extraction method based on quality analysis

## Notes

- The project uses a progressive learning approach, starting simple and adding complexity
- Each day builds on the previous day's work
- Focus is on understanding fundamentals rather than just using pre-built solutions