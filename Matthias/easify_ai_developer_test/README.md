# Easify RAG Assistant

A Retrieval-Augmented Generation system that answers questions about the Easify construction software platform using internal documentation.

## Tech Stack

| Component        | Technology                          |
|-----------------|--------------------------------------|
| LLM             | Claude (via Anthropic API)           |
| Embeddings      | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Database | ChromaDB (persistent storage)        |
| Framework       | LangChain                            |
| API             | FastAPI                              |
| Runtime         | Python 3.11.9                        |

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the system

**Option A — Web interface (recommended)**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Documents are automatically ingested on first startup.

**Option B — CLI**

```bash
# Ingest documents
python cli.py ingest

# Ask a single question
python cli.py ask "Can invoices exist on blocks?"

# Interactive chat
python cli.py chat
```

**Option C — API directly**

```bash
# Ask a question
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between a block and a unit?"}'

# Re-ingest documents
curl -X POST http://localhost:8000/api/ingest

# Health check
curl http://localhost:8000/api/health
```

## Project Structure

```
rag-system/
├── app/
│   ├── main.py             # FastAPI app with API + web UI
│   ├── config.py           # Settings from environment
│   ├── schemas.py          # Pydantic request/response models
│   └── rag/
│       ├── ingestion.py    # Document loading, classification, chunking
│       ├── retriever.py    # ChromaDB vector store + search
│       └── chain.py        # RAG chain with Claude
├── docs/                   # Clean, structured documentation
├── docs_brutal/            # Messy dataset (duplicates, outdated, informal)
├── templates/
│   └── index.html          # Minimal web interface
├── cli.py                  # Command-line interface
├── requirements.txt
├── .env.example
└── README.md
```

## Architecture

### Pipeline Overview

```
Question → Embedding → ChromaDB Search → Top-K Chunks → Claude → Grounded Answer
```

### Key Design Decisions

#### 1. Document Classification

During ingestion, each document is classified by reliability:

| Level      | Meaning                                              |
|-----------|------------------------------------------------------|
| `current`  | Authoritative, up-to-date documentation              |
| `outdated` | Archived or experimental specs (e.g., `archived_specs_2019.md`) |
| `informal` | Engineering notes, Slack dumps — useful but not canonical |

This metadata is stored alongside embeddings and included in the LLM context so Claude can prioritize current information when sources conflict.

#### 2. Chunking Strategy

Uses `RecursiveCharacterTextSplitter` with:
- **chunk_size=500**: Small enough for precise retrieval, large enough for context
- **chunk_overlap=100**: Prevents information loss at chunk boundaries
- Splits on paragraph → sentence → word boundaries (in that order)

Each chunk retains metadata: source file, dataset, reliability, chunk index.

#### 3. Embedding Model

`all-MiniLM-L6-v2` was chosen because:
- No API key required (runs locally)
- Fast inference on CPU
- Good performance on semantic similarity benchmarks
- 384-dimensional vectors (efficient for ChromaDB)

#### 4. Retrieval

ChromaDB similarity search with optional dataset filtering. Relevance scores are attached to results and passed to the LLM context so Claude can gauge how relevant each source is.

#### 5. Answer Generation

Claude receives structured context with source labels, reliability ratings, and relevance scores. The system prompt instructs it to:
- Only use provided context
- Prefer `current` over `outdated` sources
- Explicitly state when information is missing
- Reference source documents

### Handling the "Brutal" Dataset

The `docs_brutal` directory contains deliberately messy data. The system handles this through:

1. **Metadata tagging**: Archived/historical docs are tagged as `outdated`, Slack dumps as `informal`
2. **Context enrichment**: Reliability labels in the LLM prompt help Claude weigh sources
3. **Overlap deduplication**: Identical chunks in the response are deduplicated

## API Endpoints

| Method | Path            | Description                     |
|--------|----------------|---------------------------------|
| POST   | `/api/ask`     | Ask a question                  |
| POST   | `/api/ingest`  | Re-ingest all documents         |
| GET    | `/api/health`  | Health check with chunk count   |
| GET    | `/`            | Web interface                   |

### POST `/api/ask`

Request:
```json
{
  "question": "Can invoices exist on blocks?",
  "top_k": 5,
  "dataset": "all"
}
```

Response:
```json
{
  "question": "Can invoices exist on blocks?",
  "answer": "No, invoices cannot exist on blocks...",
  "sources": [
    {
      "source": "finance_notes.md",
      "content": "Blocks cannot contain invoices...",
      "score": 0.8421,
      "metadata": { "reliability": "current", "dataset": "docs" }
    }
  ],
  "num_chunks_retrieved": 5
}
```

## Improvements With More Time

1. **Hybrid search**: Combine vector similarity with BM25 keyword search for better retrieval of specific terms (e.g., "INVOICING_COMPLETED" status)
2. **Re-ranking**: Add a cross-encoder re-ranker after initial retrieval to improve precision
3. **Query expansion**: Rephrase the user question into multiple search queries to improve recall
4. **Evaluation harness**: Automated tests with expected Q&A pairs to measure retrieval quality and answer accuracy
5. **Document versioning**: Track document timestamps and prefer the most recent version when duplicates exist
6. **Streaming responses**: Stream Claude's response to the web UI for better UX
7. **Caching**: Cache frequent queries to reduce API calls and latency
8. **Metadata filtering in prompts**: More sophisticated weighting of source reliability (e.g., numeric confidence scores)
