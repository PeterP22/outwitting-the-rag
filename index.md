---
layout: default
---

# Outwitting the Devil - RAG Application 📚

A complete Retrieval-Augmented Generation (RAG) system for Napoleon Hill's "Outwitting the Devil", built from scratch as a learning project to understand AI engineering fundamentals.

[View on GitHub](https://github.com/PeterP22/outwitting-the-rag){: .btn}
[Download](https://github.com/PeterP22/outwitting-the-rag/archive/refs/heads/main.zip){: .btn}

## 🎯 Project Overview

This project implements a fully functional Q&A system that can answer questions about "Outwitting the Devil" using:
- **Local embeddings** (e5-large-v2)
- **Vector database** (Qdrant)
- **Local LLM** (Ollama with Gemma3)
- **Zero API costs** - runs completely offline

### Live Demo

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

## 🏗️ Architecture

<div class="mermaid">
graph LR
    A[User Query] --> B[Embedding Model<br/>e5-large-v2]
    B --> C[Vector Search<br/>Qdrant]
    C --> D[Retrieve Top 8<br/>Chunks]
    D --> E[LLM<br/>Gemma3/OpenAI]
    E --> F[Answer with<br/>Citations]
</div>

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| PDF Processing | 72,643 words extracted |
| Chunks | 160 chunks of 500 words |
| Embedding Dimensions | 1024 |
| Search Speed | <2ms per query |
| Answer Generation | 15-20s (local), 2-6s (OpenAI) |
| Total Storage | ~3MB |
| Cost per Query | $0 |

## 🚀 Key Features

### 1. Intelligent Text Processing
- Extracted text from 300+ page PDF with 94.6% quality score
- Smart chunking with 50-word overlap for context preservation
- Page-level metadata tracking for accurate citations

### 2. Advanced Retrieval
- Semantic search using state-of-the-art embeddings
- Dynamic score thresholding (98% of top score)
- Retrieved chunks include surrounding context

### 3. Flexible LLM Integration
- Local inference with Ollama (completely free)
- OpenAI API support for faster responses
- Consistent prompt engineering for quality answers

## 💻 Installation & Usage

### Quick Start

```bash
# Clone and setup
git clone https://github.com/PeterP22/outwitting-the-rag.git
cd outwitting-the-rag
./setup_environment.sh

# Generate embeddings (one-time)
python generate_embeddings.py

# Start querying!
python query_rag.py --interactive
```

### Example Queries

Try these questions to see the system in action:

- "What does Hill mean by drifting?"
- "How does hypnotic rhythm influence habits?"
- "What is definiteness of purpose?"
- "What role does fear play according to the Devil?"

## 🎓 Learning Outcomes

This project demonstrates end-to-end RAG implementation:

1. **PDF Processing**: Extraction, cleaning, quality analysis
2. **Text Chunking**: Strategies, overlap, metadata design  
3. **Embeddings**: Model selection, generation, storage
4. **Vector Search**: Similarity metrics, optimization
5. **RAG Pipeline**: Retrieval, context assembly, answer generation
6. **System Design**: Modular architecture, error handling

## 📈 Results & Insights

### What Worked Well
- **pdfplumber** outperformed other PDF extractors by 15%
- **500-word chunks** balanced context and retrieval precision
- **Score thresholding** improved answer relevance significantly
- **Local models** achieved good quality at zero cost

### Challenges & Solutions
- **Retrieval Quality**: Sometimes missed most relevant chunks
  - *Solution*: Increased retrieval from 5 to 8 chunks
- **Semantic Gaps**: Direct questions sometimes failed
  - *Solution*: Dynamic score threshold filtering
- **Speed vs Cost**: Local models were slow
  - *Solution*: Added OpenAI integration as option

## 🚧 Future Enhancements

- [ ] Paragraph-based semantic chunking
- [ ] Query expansion and rephrasing
- [ ] Hybrid search (keyword + semantic)
- [ ] Web UI with Gradio/Streamlit
- [ ] Conversation memory
- [ ] Evaluation metrics dashboard

## 📄 License & Acknowledgments

Built with 🧠 to understand RAG systems from first principles.

Special thanks to:
- Napoleon Hill Foundation for "Outwitting the Devil"
- HuggingFace for e5-large-v2 model
- Qdrant team for the vector database
- Ollama team for local LLM support

---

[Back to top](#outwitting-the-devil---rag-application-) | [View Code](https://github.com/PeterP22/outwitting-the-rag)

<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true});</script>