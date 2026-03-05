# LegacyLens

**RAG-powered natural language search for legacy enterprise codebases.**

LegacyLens makes large COBOL and Fortran codebases queryable in plain English.  
Ask questions like *"What does the CALCULATE-INTEREST paragraph do?"* and get cited answers  
with exact file paths, line numbers, and clickable deep links back to the source.

**Live demo:** [legacylens-ui-production.up.railway.app](https://legacylens-ui-production.up.railway.app)

---

## What you can ask

- *"Where is the main entry point of this program?"*
- *"What functions modify the CUSTOMER-RECORD?"*
- *"Explain what the SQL-ERROR paragraph does"*
- *"Find all file I/O operations"*
- *"What are the dependencies of PGMOD1?"*
- *"Show me error handling patterns in this codebase"*
- *"Generate documentation for the MAIN-PGTEST1-EX paragraph"*
- *"What business rules are in the CALCULATE-INTEREST section?"*

---

## Architecture Overview

```
Natural language query
        │
        ▼
┌──────────────────────┐
│  Safety & Routing     │  out-of-scope check, prompt injection strip,
│                       │  query expansion, detect_feature_type()
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Query Processor      │  normalize + embed with Voyage Code 2 (1536-dim)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  ChromaDB             │  top-k=5 semantic search (program-aware filter)
│  Vector Store         │  + BM25 fallback when max score < 0.70
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Reranker             │  paragraph boost, DATA deprioritize,
│                       │  dead-code/comment penalties
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Context Assembler    │  copybook injection + DATA DIVISION xref
│                       │  + parent section context + dependencies
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Feature Router       │  general / explain / dependency /
│                       │  business_logic / doc_generate
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  GPT-4o-mini          │  grounded answer generation with streaming;
│                       │  citation validation enforced
└──────────┬───────────┘
           │
           ▼
   Cited answer + file:line GitHub deep links
```

| Layer | Technology | Notes |
|---|---|---|
| Embeddings | Voyage Code 2 (1536-dim) | Code-optimised; same model for ingest and query |
| Vector DB | ChromaDB (persistent volume on Railway) | Cosine similarity; metadata-filtered |
| Retrieval | Hybrid BM25 + Vector | BM25 fallback when vector score < 0.70 |
| Reranker | Custom score-adjustment | Paragraph boost +0.15; DATA deprioritize −0.20 |
| LLM | GPT-4o-mini | temperature=0, streaming, citation enforced |
| Backend API | FastAPI | `/query`, `/query/stream`, `/file/content` |
| UI | Streamlit | Streaming, drill-down, GitHub links, eval sidebar |
| Deployment | Railway (2 services + volume) | Auto-deploy on push to `main` |

---

## Setup

### Prerequisites

- Python 3.9+
- A [Voyage AI](https://www.voyageai.com) API key
- An [OpenAI](https://platform.openai.com) API key
- A fork of the target codebase (default: GnuCOBOL Contrib) pinned to a specific commit

### Install

```bash
git clone https://github.com/<your-username>/LegacyLens.git
cd LegacyLens
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
# Edit .env and fill in all required values
```

Required variables:

| Variable | Description |
|---|---|
| `VOYAGE_API_KEY` | Voyage AI API key for embedding generation |
| `OPENAI_API_KEY` | OpenAI API key for answer generation |
| `REPO_OWNER` | GitHub username owning the target codebase repo |
| `REPO_NAME` | Repository name of the target codebase |
| `REPO_COMMIT` | Pinned commit SHA used during ingestion (for stable deep links and file content) |
| `CHROMA_PERSIST_DIR` | Where ChromaDB stores its data (default: `./chroma_db`) |
| `REPO_PATH` | Absolute path to the cloned target repo on disk (optional; used for "Show full file") |
| `LEGACYLENS_API_URL` | API base URL called by the UI (default: `http://localhost:8000`) |

### Validate environment

```bash
python -c "from legacylens.config.constants import validate_required_env_vars; r = validate_required_env_vars(); print(r)"
```

All values must show `"success": True` before running the application.

---

## Running the application

Activate the venv first (`source .venv/bin/activate`), then start both services in separate terminals:

```bash
# Terminal 1 — FastAPI backend
uvicorn legacylens.api.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run legacylens/ui/app.py
```

Open [http://localhost:8501](http://localhost:8501) to use the UI.

---

## Ingestion

To index a codebase locally:

```bash
source .venv/bin/activate
PYTHONPATH=. python legacylens/ingestion/runner.py
```

To run the production-style ingestion (downloads repo from GitHub, idempotent batches of 80 files):

```bash
PYTHONPATH=. python scripts/run_ingestion.py
```

`REPO_PATH` must be set (or it defaults to `/data/<REPO_NAME>` for Railway). The script skips already-indexed files, so re-runs are safe.

---

## Running tests

With the venv activated:

```bash
python -m pytest tests/ -v
```

Test results are saved to `tests/results/`.

**20 test modules** cover every pipeline stage: chunker, preprocessor, embedder, reference scraper, file discovery, vector store, searcher, reranker, context assembler, answer generator, all 4 features (code explainer, dependency mapper, business logic extractor, doc generator), API, UI, constants, query router, and program-aware search.

**Retrieval fallback behavior:**
- Vector search runs first (`top_k=5`; `top_k=10` for compound queries).
- If max score < `MIN_RELEVANCE_THRESHOLD` (0.70) or fewer than 3 results, BM25 fallback runs.
- If all scores < `NOT_FOUND_SCORE_THRESHOLD` (0.55), LLM is skipped and a structured "not found" response is returned immediately.

---

## Running the evaluation benchmark

**Always run eval inside the virtual environment.**

```bash
# Full 20-case benchmark
source .venv/bin/activate && python3 -m eval.run_eval

# Fast mode — 6 mandatory rubric cases only (ll-001 through ll-006)
source .venv/bin/activate && python3 -m eval.run_eval --fast
```

Results are written to `tests/results/eval_<timestamp>.txt`. Run eval after every new feature and fix any regressions before merging.

**Eval categories (20 total):**

| Category | Count | Description |
|----------|-------|-------------|
| `happy_path` | 6 | Core rubric scenarios |
| `structural` | 4 | COBOL structure / data division queries |
| `dependency` | 4 | Cross-file CALL/COPY dependency chains |
| `edge_case` | 4 | Dead code, hallucination probe, not-found programs |
| `business_logic` | 2 | Business rule extraction |

**Metrics evaluated:**
- **Retrieval Precision:** Were expected chunks (by `file_path` + `paragraph_name`) in top-k results?
- **Answer Faithfulness:** Did the answer contain all `must_contain` phrases and none of `must_not_contain`?

---

## Code Understanding Features

LegacyLens automatically routes queries to one of four specialized feature modules based on keyword detection:

| Feature | Routing keywords (examples) | What it does |
|---------|----------------------------|--------------|
| **Code Explanation** | `"explain what"`, `"what does ... do"`, `"purpose of"` | Explains a COBOL paragraph in plain English |
| **Dependency Mapping** | `"dependencies of"`, `"what does ... call"`, `"copy statements"` | Parses CALL/COPY/USING — direct ChromaDB lookup, no semantic search |
| **Business Logic Extraction** | `"business rule"`, `"validation logic"`, `"control flow"` | Extracts IF/EVALUATE conditions and data transformation rules |
| **Documentation Generation** | `"generate documentation"`, `"document the"`, `"auto-document"` | Generates structured markdown docs with Summary/Parameters/Dependencies/Side Effects |

Queries that don't match any feature keyword go through the general RAG pipeline.

---

## Project structure

```
LegacyLens/
├── legacylens/
│   ├── config/
│   │   └── constants.py              # All project constants (no magic numbers anywhere else)
│   ├── ingestion/
│   │   ├── file_discovery.py         # Recursive COBOL/Fortran file discovery with path safety
│   │   ├── preprocessor.py           # Column strip, comment extract, PII redact, dead code detect
│   │   ├── chunker.py                # Paragraph-level (PROCEDURE) + fixed-size (DATA/COPYBOOK)
│   │   ├── reference_scraper.py      # Regex CALL/COPY/USING dependency extraction
│   │   ├── embedder.py               # Voyage Code 2 batch embedding (128/batch, 3 workers)
│   │   └── runner.py                 # Full ingestion pipeline entry point
│   ├── retrieval/
│   │   ├── vector_store.py           # ChromaDB wrapper (insert, query, metadata sanitization)
│   │   ├── query_processor.py        # Query normalization and COBOL term expansion
│   │   ├── searcher.py               # Hybrid BM25+vector, program-aware filter, compound queries
│   │   ├── reranker.py               # Score-adjustment reranker (paragraph boost, DATA penalty)
│   │   └── context_assembler.py      # Copybook injection, DATA xref, section context, truncation
│   ├── features/
│   │   ├── __init__.py               # detect_feature_type() — query routing
│   │   ├── code_explainer.py         # Plain-English paragraph explanation
│   │   ├── dependency_mapper.py      # Direct ChromaDB CALL/COPY/USING mapper
│   │   ├── business_logic_extractor.py  # Business rule / validation extraction
│   │   └── doc_generator.py          # Structured markdown documentation generator
│   ├── generation/
│   │   └── answer_generator.py       # GPT-4o-mini, streaming, citation validation, backoff
│   ├── api/
│   │   └── main.py                   # FastAPI: /query, /query/stream, /file/content
│   └── ui/
│       └── app.py                    # Streamlit UI: streaming, drill-down, eval sidebar
├── eval/
│   ├── golden_data.yaml              # 20-query benchmark (5 categories)
│   └── run_eval.py                   # Evaluation runner with --fast flag
├── scripts/
│   ├── run_ingestion.py              # Production ingestion (Railway): GitHub ZIP download, batches
│   ├── run_ui_railway.sh             # Railway UI start script (PORT expansion fix)
│   └── clear_chromadb.py            # Wipe ChromaDB collection for fresh ingestion
├── tests/
│   ├── test_chunker.py
│   ├── test_preprocessor.py
│   ├── test_embedder.py
│   ├── test_reference_scraper.py
│   ├── test_file_discovery.py
│   ├── test_vector_store.py
│   ├── test_searcher.py
│   ├── test_reranker.py
│   ├── test_context_assembler.py
│   ├── test_query_processor.py
│   ├── test_answer_generator.py
│   ├── test_code_explainer.py
│   ├── test_dependency_mapper.py
│   ├── test_business_logic_extractor.py
│   ├── test_doc_generator.py
│   ├── test_api.py
│   ├── test_ui_app.py
│   ├── test_constants.py
│   ├── test_query_router.py
│   ├── test_program_aware_search.py
│   └── results/                      # TDD + eval result logs (timestamped)
├── docs/
│   ├── LegacyLens_PRD.md
│   ├── LegacyLens_PreSearch.md
│   ├── RAG_Architecture_Doc.md
│   ├── AI_Cost_Analysis.md
│   └── G4-Week-3-LegacyLens.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Deployment (Railway)

1. Push to GitHub.
2. Create a new Railway project → connect the repo. Create **two services** from the same repo: one for the API (`legacylens-api`), one for the UI (`legacylens-ui`).
3. Set all environment variables from `.env.example` in the Railway dashboard for **both services** (or as shared variables).
   - For the UI service, set `LEGACYLENS_API_URL` to the **API service's public URL** (e.g. `https://legacylens-api-production-….up.railway.app`).
4. Add a Railway **volume** to the **API** service. Set:
   - `CHROMA_PERSIST_DIR` = volume mount path (e.g. `/data/chroma_db`)
   - `REPO_PATH` = path where ingestion writes the repo (e.g. `/data/gnucobol-contrib`)
5. **Run ingestion** on Railway to populate ChromaDB and download the repo to `REPO_PATH`:
   ```
   Start Command (one-time): PYTHONPATH=. python scripts/run_ingestion.py
   ```
   After ingestion completes, switch the API service start command back to `uvicorn`.
6. **API service start command:** `uvicorn legacylens.api.main:app --host 0.0.0.0 --port $PORT`
7. **UI service start command:** `sh scripts/run_ui_railway.sh`  
   *(Do not use the raw `streamlit run ... --server.port=$PORT` command — use the script so `PORT` is always expanded.)*
8. Railway auto-deploys on each push to `main`.

**"Show full file" in production:**  
If `REPO_PATH` is set but the file isn't on disk, the API automatically fetches it from `raw.githubusercontent.com` using `REPO_OWNER`, `REPO_NAME`, and `REPO_COMMIT`. No re-ingestion needed.

---

## Key constants

All tunable values live in `legacylens/config/constants.py` — no magic numbers anywhere else.

| Constant | Value | What it controls |
|----------|-------|-----------------|
| `TOP_K` | 5 | Default number of retrieved chunks per query |
| `TOP_K_COMPOUND` | 10 | Top-k for queries with multiple COBOL entities |
| `MIN_RELEVANCE_THRESHOLD` | 0.70 | Below this → BM25 fallback triggers |
| `NOT_FOUND_SCORE_THRESHOLD` | 0.55 | Below this → skip LLM, return "not found" |
| `EMBEDDING_MODEL` | `voyage-code-2` | Voyage AI model for embeddings |
| `EMBEDDING_DIMENSIONS` | 1536 | Vector dimension |
| `INGESTION_BATCH_SIZE` | 128 | Chunks per Voyage API call |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model for answer generation |
| `LLM_MAX_TOKENS` | 1500 | Max tokens in LLM answer |
| `MAX_CHUNK_TOKENS` | 500 | Max tokens per chunk before sub-splitting |
| `MAX_ASSEMBLED_CONTEXT_CHARS` | 12,000 | Char budget for assembled context sent to LLM |
| `MAX_FILE_VIEW_LINES` | 2000 | Lines returned by `GET /file/content` |
| `MAX_RETRIES` | 3 | Retries on Voyage/OpenAI transient errors |

---

## License

This project is part of the Gauntlet AI G4 program.
