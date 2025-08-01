# Building My First RAG System: Lessons from "Outwitting the Devil"

*A journey from PDF to production-ready Q&A system in one week*

## The Beginning: Why RAG?

Last week, I decided to dive deep into AI engineering by building something real. Not another tutorial project, but a complete Retrieval-Augmented Generation (RAG) system from scratch. My goal? Create a Q&A system for Napoleon Hill's "Outwitting the Devil" that could answer complex questions about the book with accurate citations.

Why this book? It's dense with philosophical concepts that would test whether my system could handle nuanced retrieval and generation. Plus, I had the PDF ready to go.

## Week-Long Learning Plan

I structured my learning into four main phases:

### Day 1-2: PDF Extraction & Text Processing
The first challenge hit immediately. How do you extract clean text from a 300-page PDF? I tested three libraries:
- PyPDF2: Fast but messy (79% quality)
- pdfplumber: Slower but cleaner (94.6% quality)
- PyMuPDF: Good balance (87% quality)

I went with pdfplumber. The extra processing time was worth the quality improvement.

**Key Learning**: Always measure quality objectively. I created a scoring system that checked for proper sentence boundaries, word spacing, and special character handling.

### Day 3: The Chunking Dilemma
This was harder than expected. How do you split a book into searchable pieces without losing context? I experimented with:
- Chapter-based chunks (failed - chapters were too variable)
- Sentence-based chunks (too granular)
- Fixed-size word chunks with overlap (winner!)

Final approach: 500-word chunks with 50-word overlap. This preserved context across boundaries while keeping chunks focused.

**Key Learning**: Simple solutions often beat complex ones. My attempt at "smart" chapter detection wasted hours compared to the straightforward word-count approach.

### Day 4-5: Embeddings & Vector Search
This is where the magic happens. I chose:
- **Model**: e5-large-v2 (1024 dimensions)
- **Database**: Qdrant in local mode
- **Search**: Cosine similarity with dynamic thresholding

The e5 model requires special prefixes (`passage:` for indexing, `query:` for searching). Missing this initially gave terrible results!

**Key Learning**: Read the model documentation carefully. Small details like prefix requirements can make or break your system.

### Day 6-7: Building the RAG Pipeline
Connecting everything was satisfying. The pipeline:
1. User asks: "What is hypnotic rhythm?"
2. Generate query embedding
3. Search Qdrant for similar chunks
4. Retrieve top 8 chunks (increased from 5 for better coverage)
5. Feed to LLM with context
6. Return answer with page citations

I integrated both Ollama (local, free) and OpenAI (faster, costs money) for flexibility.

## Challenges & Solutions

### 1. The Semantic Gap Problem
**Issue**: Direct questions like "What is hypnotic rhythm?" weren't finding chunks containing the exact definition.

**Solution**: Implemented dynamic score thresholding - only keep chunks within 2% of the top score. This filtered out marginally relevant content.

### 2. Empty Responses with OpenAI
**Issue**: Complex questions returned empty responses despite using tokens.

**Discovery**: The model was using "reasoning tokens" internally! Increasing `max_completion_tokens` from 150 to 500 solved it.

### 3. Speed vs Cost Trade-off
- Local (Ollama): 15-20s per query, $0 cost
- Cloud (OpenAI): 2-6s per query, ~$0.001 per query

**Solution**: Built support for both. Develop locally, deploy with API.

## Surprising Discoveries

1. **PDF extraction quality matters more than embedding model sophistication**. Garbage in, garbage out.

2. **More chunks isn't always better**. Increasing from 5 to 8 chunks helped, but 15 chunks made answers unfocused.

3. **Simple chunking beat complex approaches**. My fixed-size chunks outperformed attempts at "semantic" chunking.

4. **Metadata is crucial**. Tracking page numbers through the entire pipeline enabled accurate citations.

## Real Results

The system works remarkably well:

```
❓ "What are the six basic fears?"
📖 According to pages 60-61, the six basic fears the Devil uses are:
   fear of poverty, criticism, ill health, loss of love, old age, and death.
📚 Sources: Pages 75-77
```

Response time: 2.15s (with OpenAI)
Accuracy: Correct answer with accurate page citations

## Technical Stack

- **Language**: Python 3.12
- **PDF Processing**: pdfplumber
- **Embeddings**: sentence-transformers with e5-large-v2
- **Vector DB**: Qdrant (local mode)
- **LLMs**: Ollama (Gemma3) / OpenAI API
- **Total cost**: $0 (everything runs locally)

## Code Architecture

I kept it modular:
```
extract_full_pdf.py    → Clean PDF text
chunk_text_simple.py   → Create searchable chunks  
generate_embeddings.py → Create vector embeddings
query_rag.py          → Main RAG interface
```

Each script can run independently, making debugging much easier.

## Lessons for Your First RAG

1. **Start simple**: Get a basic pipeline working before optimizing
2. **Measure everything**: Quality scores, retrieval accuracy, response times
3. **Test with real questions**: Not just "what is X?" but complex queries
4. **Plan for iteration**: Your first approach won't be perfect
5. **Document as you go**: You'll forget why you made certain decisions

## What's Next?

The basic system is complete, but I'm excited about:
- Hybrid search (combining keyword + semantic)
- Streaming responses for better UX
- Web interface with Gradio
- Evaluation metrics dashboard

## Try It Yourself

The entire project is open source: [GitHub - outwitting-the-rag](https://github.com/PeterP22/outwitting-the-rag)

Clone it, run `setup_environment.sh`, and you'll have a working RAG system in minutes. No API keys required!

## Final Thoughts

Building a RAG system from scratch taught me more in a week than months of reading about it. The challenges were real - from PDF extraction nightmares to mysterious empty responses - but solving each one deepened my understanding.

If you're learning AI engineering, stop reading tutorials and build something. Pick a PDF you care about and make it queryable. You'll learn about embeddings, vector databases, prompt engineering, and system design all at once.

The future of AI isn't just about using ChatGPT - it's about building systems that augment human knowledge in specific domains. This project proved that anyone can build these systems with open-source tools and determination.

*What PDF will you make queryable?*

---

**Technical details**: 72,643 words processed, 160 chunks created, 1024-dimensional embeddings, <2ms search latency, 94.6% extraction quality score.

**Contact**: Follow the project on [GitHub](https://github.com/PeterP22/outwitting-the-rag) for updates and improvements.