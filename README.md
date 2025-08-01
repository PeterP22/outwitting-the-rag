# Outwitting the Devil - RAG Application 📚

A complete Retrieval-Augmented Generation (RAG) system for Napoleon Hill's "Outwitting the Devil", built from scratch as a learning project to understand AI engineering fundamentals.

## 🎯 Project Overview

This project implements a fully functional Q&A system that can answer questions about "Outwitting the Devil" using:
- **Local embeddings** (e5-large-v2)
- **Vector database** (Qdrant)
- **Local LLM** (Ollama with Gemma3)
- **Zero API costs** - runs completely offline

### Key Features
- 📖 Semantic search through 300+ pages of content
- 💬 Natural language answers with source citations
- 🚀 Fast retrieval (<2ms search latency)
- 🔒 Completely local and private
- 📍 Page-level source attribution

## 🏗️ Architecture

```
User Query → Embedding → Vector Search → Retrieval → LLM → Answer
     ↓           ↓              ↓            ↓         ↓        ↓
"What is    e5-large-v2     Qdrant      Top 8      Gemma3   Natural
 fear?"      (1024-dim)    (Cosine)     chunks    (Ollama)  response
```

## 📊 Performance Metrics

- **PDF Processing**: 72,643 words extracted and cleaned
- **Chunks**: 160 chunks of 500 words with 50-word overlap
- **Embeddings**: 1024-dimensional vectors
- **Search Speed**: <2ms per query
- **Answer Generation**: 15-20s (using local Gemma3)
- **Total Storage**: ~3MB (embeddings + database)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) with Gemma3 model installed
- 4GB RAM minimum

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PeterP22/outwitting-the-rag.git
cd outwitting-the-rag
```

2. Set up environment:
```bash
./setup_environment.sh
# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Generate embeddings (one-time setup):
```bash
python generate_embeddings.py
```

### Usage

#### Interactive Mode
```bash
python query_rag.py
```

#### Single Query
```bash
python query_rag.py "What does Hill mean by drifting?"
```

#### Verbose Mode (see retrieval details)
```bash
python query_rag.py "What are the six basic fears?" --verbose
```

#### JSON Output
```bash
python query_rag.py "What is hypnotic rhythm?" --json
```

## 📁 Project Structure

```
outwitting-the-rag/
├── Core Scripts
│   ├── extract_full_pdf.py        # PDF text extraction and cleaning
│   ├── chunk_text_simple.py       # Text chunking with metadata
│   ├── generate_embeddings.py     # Embedding generation and storage
│   └── query_rag.py              # Main RAG interface
│
├── Data Files
│   ├── processed_text/
│   │   ├── outwitting_the_devil_cleaned.txt  # Cleaned book text
│   │   └── book_chunks.jsonl                 # Chunked text with metadata
│   └── qdrant_db/                             # Vector database storage
│
├── Test Scripts
│   ├── test_book_questions.py     # Test with key book questions
│   ├── test_ollama_models.py      # Model comparison
│   └── debug_query_process.py     # Detailed query debugging
│
└── Documentation
    ├── step1_summary.md           # PDF extraction details
    ├── step2_summary.md           # Chunking strategy
    ├── step3_summary.md           # Embedding implementation
    └── step4_summary.md           # RAG interface details
```

## 🔧 Technical Implementation

### 1. PDF Extraction (Step 1)
- Tested 3 extraction methods (PyPDF2, pdfplumber, PyMuPDF)
- Chose pdfplumber for best quality (94.6/100 score)
- Implemented comprehensive text cleaning:
  - Removed headers/footers
  - Fixed hyphenation
  - Normalized whitespace
  - Preserved page markers

### 2. Text Chunking (Step 2)
- Fixed-size chunks: 500 words
- Overlap: 50 words (10%)
- Metadata per chunk:
  ```json
  {
    "chunk_id": "chunk_0001",
    "text": "actual content...",
    "word_count": 500,
    "start_page": 12,
    "end_page": 14,
    "preview": "first 150 chars..."
  }
  ```

### 3. Embeddings & Vector Storage (Step 3)
- Model: `intfloat/e5-large-v2` (1024-dim)
- Vector DB: Qdrant (local mode)
- Indexing: Cosine similarity
- Special e5 prefixes:
  ```python
  # For indexing
  text = f"passage: {chunk_text}"
  # For queries
  query = f"query: {user_question}"
  ```

### 4. RAG Pipeline (Step 4)
- Retrieval: Top 8 chunks with score filtering
- Score threshold: 98% of top score
- Context limit: 3000 characters to LLM
- Answer generation: Ollama Gemma3
- Source citation: Page numbers included

## 📈 Example Results

```bash
❓ Query: "What are the six basic fears?"

📖 Answer: According to pages 60-61 of "Outwitting the Devil," 
the six basic fears the Devil uses to control people are:
- Fear of poverty
- Fear of criticism  
- Fear of ill health
- Fear of loss of love
- Fear of old age
- Fear of death

📚 Sources: Pages 75-77
⏱️ Response time: 2.15s
```

## 🧪 Testing & Evaluation

Test the system with key book concepts:
```bash
# Run comprehensive test suite
python test_book_questions.py

# Quick test of main concepts
./quick_test_questions.sh
```

Key questions that work well:
- "What does Hill mean by drifting?"
- "How does hypnotic rhythm influence habits?"
- "What is definiteness of purpose?"
- "What role does fear play according to the Devil?"

## 🚧 Known Limitations & Future Improvements

### Current Limitations
1. **Retrieval Quality**: Sometimes misses most relevant chunks for direct questions
2. **Speed**: 15-20s per query due to local LLM
3. **Context Window**: Limited to 3000 chars (~4-5 chunks)

### Planned Enhancements
- [ ] Paragraph-based semantic chunking
- [ ] Query expansion for better retrieval
- [ ] Hybrid search (BM25 + semantic)
- [ ] Conversation memory
- [ ] Web UI with Gradio/Streamlit
- [ ] GPU acceleration

## 💰 Cost Analysis

**Completely Free!**
- Embeddings: $0 (local e5-large-v2)
- Vector DB: $0 (local Qdrant)
- LLM: $0 (local Ollama)
- **Total: $0** per query

Compare to cloud-based approach:
- OpenAI embeddings: ~$0.0012
- OpenAI GPT-3.5: ~$0.002 per query
- Cloud vector DB: $0.05+/month

## 🎓 Learning Outcomes

This project demonstrates:
1. **PDF Processing**: Extraction, cleaning, quality analysis
2. **Text Chunking**: Strategies, overlap, metadata design
3. **Embeddings**: Model selection, generation, storage
4. **Vector Search**: Similarity metrics, optimization
5. **RAG Pipeline**: Retrieval, context assembly, answer generation
6. **System Design**: Modular architecture, error handling

## 🤝 Contributing

This is a learning project, but contributions are welcome! Areas for improvement:
- Better chunking strategies
- Additional embedding models
- Performance optimizations
- UI/UX enhancements

## 📄 License

This project is for educational purposes. The book content remains the property of the Napoleon Hill Foundation.

## 🙏 Acknowledgments

- Napoleon Hill Foundation for "Outwitting the Devil"
- HuggingFace for e5-large-v2 model
- Qdrant team for the vector database
- Ollama team for local LLM support

---

Built with 🧠 to understand RAG systems from first principles.