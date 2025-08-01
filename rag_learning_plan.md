# RAG Learning Plan - Outwitting the Devil

## Context

Goal: Learn AI engineering fundamentals through building a RAG application for Napoleon Hill's "Outwitting the Devil" book, progressing from basic to advanced concepts including semantic search and vectorization.

## Initial Learning Goals Discussion

### Question: Learning More About AI Engineering

**User Goal**: 
- Become a more complete coder/developer
- Currently uses LLMs for code generation ("vibe coding")
- Wants to ship fast to production
- Master/ace future interviews
- Be an expert who can have intellectual conversations about AI
- Learn daily

### ChatGPT's Framework

**Five Key Tracks for AI Engineering Mastery:**

| Track | Why it matters | First micro-habit (≤ 30 min) |
|-------|----------------|------------------------------|
| 1. Code Craft | Clean Architecture, testing, patterns, refactoring - AI projects still crumble on brittle software | Review one file you shipped this week → find one smell and refactor it with a unit test |
| 2. ML/AI Fundamentals | Math intuition + core algorithms - Lets you debug & explain models (beats vibe-coding) | Skim 2-page "backprop derivation" cheatsheet → re-derive ∂L/∂W for a single neuron on paper |
| 3. LLM/Generative Systems | Prompt/chain design, RAG, evals - Your daily consulting bread-and-butter | Build a toy RAG: markdown docs + FAISS + OpenAI → answer one query end-to-end |
| 4. MLOps & Infra | CI/CD, orchestration, monitoring, cost - Shipping ≈ automating the boring parts | Containerise the toy RAG with Docker → push to a free cloud instance |
| 5. Systems & Interviews | Distributed design, big-O, coding drills - Pass "bar raiser" rounds confidently | Solve one LeetCode medium with no LLM help; afterwards compare with ChatGPT and refactor |

**Selected Track**: 3 - LLM/Generative Systems

## Week-Long Sprint Plan: RAG for "Outwitting the Devil"

### Goal
Ship a tiny Retrieval-Augmented Generation (RAG) service that can answer questions over "Outwitting the Devil" (300 pages), plus evaluation script.

### Daily Plan (45 min/day)

| Day | Focus | Concrete Outcome | Micro-habit (≤10 min) |
|-----|-------|------------------|----------------------|
| **Mon** | **Ingest & Chunk** | `book_chunks.jsonl` | Note one trade-off while experimenting with chunk sizes |
| | • Convert PDF to clean plain-text/Markdown<br>• Chunk size/overlap (start: 500 words, 50 word stride)<br>• Tag chunks: {chapter, page, chunk_id} | | |
| **Tue** | **Embed & Store** | `faiss_index.bin` | Log average cosine distance of 3 nearest neighbors for 5 random chunks |
| | • Generate embeddings (OpenAI text-embedding-3-small or local bge-small)<br>• Load into FAISS/Chroma with ID + metadata | | |
| **Wed** | **RAG Chain + Semantic Config** | Working Python function | Write down one "gotcha" (e.g., prompt too long, irrelevant chunks) |
| | • `retrieve(question)` → top_k chunks<br>• Config file (rag.yaml) with knobs:<br>  - top_k, similarity_threshold<br>  - prompt_template<br>• Build `answer(question, config)` | | |
| **Thu** | **Packaging** | Dockerfile + local container | Capture one CLI command used today |
| | • Wrap answer() in FastAPI endpoint<br>• Add /config route for hot-swap rag.yaml<br>• Dockerize & run locally | | |
| **Fri** | **Evaluate & Reflect** | `eval_report.md` | End-of-week "3 wins, 1 pain-point, next tweak" journal |
| | • Create 10 gold Q&A pairs (varied difficulty)<br>• Automated BLEU/embedding-sim evaluation<br>• Manual spot-check: "Does answer cite right chunk?"<br>• Summarize precision vs. recall issues | | |

### Next Steps (Beyond Week 1)

1. **Semantic re-ranking**: After initial retrieval, run lightweight cross-encoder to re-rank chunks
2. **Context window optimizer**: Dynamic chunk merging to stay under model token limit
3. **Feedback-driven chunking**: Use evaluation errors to adjust chunk size/overlap

## Monday's Deep Dive: PDF Text Extraction

### PDF to Text Options

| Option | Quick Command | When It's Handy |
|--------|--------------|-----------------|
| **PyPDF2** (pure-Python) | ```python\nfrom PyPDF2 import PdfReader\ntext = "".join(page.extract_text() for page in PdfReader("book.pdf").pages)``` | PDF is mostly text, minimal formatting |
| **pdfminer.six** (better at weird layouts) | `pdf2txt.py -o out.txt book.pdf` | Complex fonts/columns |
| **pandoc** (CLI, good Markdown) | `pandoc book.pdf -t markdown -o book.md` | You'd like headings/italics preserved |

### Chunking Strategy

**Recommended starter**: 500 words per chunk with 50-word overlap
- Keeps single ideas together
- Gives retriever extra context

### Pro Tips

- Skim first 2-3 pages of output to ensure page numbers/headers aren't polluting text
- Clean text = better retrieval

## Key Learning Principles

1. **Understanding → Deliberate Practice → Ship & Review**
2. **Daily micro-habits** to cement learning
3. **Configuration-driven design** for experimentation
4. **Evaluation-first mindset** to measure improvements