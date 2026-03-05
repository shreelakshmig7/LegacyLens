# LegacyLens

**RAG-powered natural language search for legacy enterprise codebases.**

LegacyLens makes large COBOL and Fortran codebases queryable in plain English.  
Ask questions like *"What does the CALCULATE-INTEREST paragraph do?"* and get cited answers  
with exact file paths, line numbers, and clickable deep links back to the source.

---

## Architecture Overview

```
Natural language query
        │
        ▼
┌─────────────────┐
│  Query Processor │  normalize, expand, embed with voyage-code-2
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB       │  top-k semantic search + BM25 fallback on low max relevance
│  Vector Store    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Context Assembler│  copybook injection + DATA DIVISION xref
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPT-4o-mini     │  grounded answer generation with streaming
└────────┬────────┘
         │
         ▼
   Cited answer + file:line deep links
```

| Layer | Technology |
|---|---|
| Embeddings | Voyage Code 2 (1536-dim, code-optimised) |
| Vector DB | ChromaDB (persistent volume on Railway) |
| LLM | GPT-4o-mini |
| Backend API | FastAPI |
| UI | Streamlit (React in a later phase) |
| Deployment | Railway |

---

## Setup

### Prerequisites

- Python 3.11+
- A [Voyage AI](https://www.voyageai.com) API key
- An [OpenAI](https://platform.openai.com) API key
- A fork of the target codebase (default: OpenCOBOL Contrib) pinned to a specific commit

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
| `REPO_OWNER` | GitHub username of your fork of the target codebase |
| `REPO_NAME` | Repository name of the target codebase |
| `REPO_COMMIT` | Pinned commit SHA used during ingestion (for stable deep links) |
| `CHROMA_PERSIST_DIR` | Directory where ChromaDB stores its data (default: `./chroma_db`) |

### Validate environment

```bash
python -c "from legacylens.config.constants import validate_required_env_vars; r = validate_required_env_vars(); print(r)"
```

All values must show `"success": True` before running the application.

---

## Running the application

Activate the venv first (`source .venv/bin/activate`), then:

```bash
# Start the FastAPI backend
uvicorn legacylens.api.main:app --reload

# Start the Streamlit UI (separate terminal)
streamlit run legacylens/ui/app.py
```

---

## Running tests

With the venv activated:

```bash
python -m pytest tests/ -v
```

Test results are saved to `tests/results/`.

Retrieval fallback behavior:
- Vector search runs first.
- If vector max score is below `MIN_RELEVANCE_THRESHOLD` (0.70), BM25 fallback runs.
- If BM25 also returns nothing useful, the API returns a structured `not found` response with query reformulation suggestions.

---

## Running the evaluation benchmark

**Always run eval inside the virtual environment.** Running with system Python will fail (e.g. `No module named 'chromadb'`). Never run any LegacyLens command outside the venv.

```bash
source .venv/bin/activate && python3 -m eval.run_eval --fast
```

Use `--fast` to run only the 6 mandatory cases (ll-001 through ll-006). For the full 20-case benchmark:

```bash
source .venv/bin/activate && python3 -m eval.run_eval
```

Results are written to `tests/results/`. Run after every new feature and fix any regressions before moving on.

---

## Project structure

```
LegacyLens/
├── legacylens/
│   ├── config/
│   │   └── constants.py          # All project constants
│   ├── ingestion/
│   │   ├── file_discovery.py
│   │   ├── preprocessor.py
│   │   ├── chunker.py
│   │   ├── reference_scraper.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   ├── query_processor.py
│   │   ├── searcher.py
│   │   ├── reranker.py
│   │   └── context_assembler.py
│   ├── features/
│   │   ├── code_explainer.py
│   │   ├── dependency_mapper.py
│   │   ├── business_logic.py
│   │   └── doc_generator.py
│   ├── generation/
│   │   └── answer_generator.py
│   ├── api/
│   │   └── main.py
│   └── ui/
│       └── app.py
├── eval/
│   ├── golden_data.yaml          # 20-query benchmark
│   └── run_eval.py               # Evaluation runner
├── tests/
│   ├── test_constants.py
│   └── results/                  # TDD + eval result logs
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Deployment (Railway)

1. Push to GitHub.
2. Create a new Railway project → connect the repo. Create **two services** from the same repo: one for the API, one for the UI.
3. Set all environment variables from `.env.example` in the Railway dashboard (for both services, or as shared variables). For the UI, set `LEGACYLENS_API_URL` to your **API service’s public URL** (e.g. `https://legacylens-api-production-….up.railway.app`).
4. Add a Railway volume to the **API** service and set `CHROMA_PERSIST_DIR` to the volume mount path so ChromaDB data survives redeploys.
5. **API service (legacylens-api):** Use the repo’s Procfile (default). It runs `uvicorn … --host 0.0.0.0 --port $PORT`. If you set a custom start command, use: `uvicorn legacylens.api.main:app --host 0.0.0.0 --port $PORT`.
6. **UI service (legacylens-ui):** Use the **script** so `PORT` is always expanded (Railway may not expand `$PORT` in a raw start command, so Streamlit can end up on the wrong port → 502):
   - **Start Command:** `sh scripts/run_ui_railway.sh`
   - Do **not** use the raw `streamlit run ... --server.port=$PORT` command; use the script instead.
7. Railway auto-deploys on each push to `main`.

---

## License

This project is part of the Gauntlet AI G4 program.
