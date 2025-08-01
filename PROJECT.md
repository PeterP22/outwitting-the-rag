# Outwitting the Devil - RAG Application Project

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for Napoleon Hill's "Outwitting the Devil" book. The goal is to build a complete RAG pipeline from scratch as a learning exercise, progressing from basic text extraction to advanced semantic search capabilities.

## Project Goals

1. **Learn AI Engineering Fundamentals** - Understanding RAG systems from the ground up
2. **Build Production-Ready Code** - Clean, well-documented, testable code
3. **Master Key Concepts** - Embeddings, vector search, semantic retrieval, and LLM integration
4. **Progressive Learning** - Start simple, add complexity incrementally

## Week-Long Learning Plan

### ✅ Day 1: PDF Extraction & Text Quality (COMPLETED)
- Tested multiple PDF extraction methods (PyPDF2, pdfplumber, PyMuPDF)
- Analyzed extraction quality and identified issues
- Implemented comprehensive text cleanup pipeline
- Extracted and cleaned the entire book (72,643 words)

### 📋 Day 2: Text Chunking & Metadata
- Implement intelligent text chunking (500 words with 50-word overlap)
- Add metadata tracking (chapter, page, chunk_id)
- Save chunks in structured format (JSONL)
- Consider semantic boundaries

### 📋 Day 3: Embeddings & Vector Storage
- Generate embeddings using OpenAI or local models
- Set up vector database (ChromaDB/FAISS)
- Store embeddings with metadata
- Implement similarity search

### 📋 Day 4: RAG Chain & API
- Build retrieval pipeline
- Create answer generation with LLM
- Wrap in FastAPI
- Add configuration management

### 📋 Day 5: Evaluation & Optimization
- Create test questions and golden answers
- Implement evaluation metrics (BLEU, semantic similarity)
- Analyze and improve retrieval quality
- Document findings

## Technical Stack

### Core Dependencies
- **PDF Processing**: pdfplumber (chosen for best quality)
- **Text Processing**: NLTK, spaCy
- **Vector Database**: ChromaDB (planned)
- **Embeddings**: OpenAI API / Sentence Transformers
- **API Framework**: FastAPI
- **Development**: Jupyter, IPython

### Project Structure
```
outwitting-the-rag/
├── extract_pdf_test.py          # PDF extraction testing script
├── analyze_extraction_quality.py # Quality analysis tool
├── extract_full_pdf.py          # Full extraction with cleanup
├── extraction_results/          # Test extraction outputs
├── processed_text/             
│   ├── outwitting_the_devil_cleaned.txt  # Full cleaned text
│   └── sample_cleaned_text.txt           # Sample for review
├── requirements.txt             # Project dependencies
├── setup_environment.sh         # Environment setup script
├── .gitignore                  # Git ignore rules
├── README.md                   # Basic project info
├── PROJECT.md                  # This detailed documentation
├── rag_learning_plan.md        # Original learning plan from ChatGPT
└── step1_summary.md            # Day 1 accomplishments
```

## Key Accomplishments (Day 1)

### 1. Environment Setup
- Created isolated Python virtual environment
- Organized dependencies with clear sections
- Set up proper .gitignore (including .env protection)

### 2. PDF Extraction Analysis
- Tested 3 extraction methods with quality scoring:
  - **pdfplumber**: 94.6/100 ✅ (selected)
  - **PyPDF2**: 94.6/100
  - **PyMuPDF**: 87.4/100

### 3. Text Cleanup Pipeline
Implemented comprehensive cleanup:
- Removed standalone page numbers
- Cleaned headers/footers
- Fixed hyphenated line breaks
- Normalized whitespace
- Preserved page markers for reference

### 4. Results
- **Total pages**: 302 (296 with content)
- **Total characters**: 408,918
- **Total words**: ~72,643
- **Clean, structured text** ready for chunking

## How to Run

### Initial Setup
```bash
# Clone the repository
git clone <repository-url>
cd outwitting-the-rag

# Set up environment
./setup_environment.sh
# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Extract Text from PDF
```bash
# Test extraction methods (first 10 pages)
python extract_pdf_test.py

# Analyze extraction quality
python analyze_extraction_quality.py

# Extract and clean full PDF
python extract_full_pdf.py
```

## Next Steps

1. **Review cleaned text** - Check `processed_text/sample_cleaned_text.txt`
2. **Implement chunking** - Create intelligent text chunks with metadata
3. **Generate embeddings** - Convert chunks to vector representations
4. **Build retrieval** - Implement semantic search functionality

## Learning Resources

- [Original ChatGPT Conversation](rag_learning_plan.md) - Initial planning discussion
- [Napoleon Hill's Outwitting the Devil](Hill_Napoleon_-_Outwitting_the_devil.pdf) - Source material
- [Step 1 Summary](step1_summary.md) - Detailed Day 1 accomplishments

## Notes

- This is a learning project focused on understanding RAG fundamentals
- Code includes extensive comments for educational purposes
- Each step builds on the previous, allowing for iterative learning
- The project uses "Outwitting the Devil" as it's a substantial text (300+ pages) that provides good complexity for RAG implementation

## Environment Variables

Create a `.env` file for sensitive configuration (this file is gitignored):
```
# OpenAI API Key (for embeddings, if using OpenAI)
OPENAI_API_KEY=your_key_here

# Other API keys as needed
```

## Contributing

This is a personal learning project, but feedback and suggestions are welcome!