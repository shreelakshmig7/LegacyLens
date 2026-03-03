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
│   ChromaDB       │  top-k semantic search + BM25 fallback
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

```bash
# Start the FastAPI backend
uvicorn legacylens.api.main:app --reload

# Start the Streamlit UI (separate terminal)
streamlit run legacylens/ui/app.py
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

Test results are saved to `tests/results/`.

---

## Running the evaluation benchmark

```bash
python eval/run_eval.py
```

This runs all 20 golden test cases and writes a timestamped results file to `tests/results/`.  
Run after every new feature. Fix any regressions before moving on.

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
2. Create a new Railway project → connect the repo.
3. Set all environment variables from `.env.example` in the Railway dashboard.
4. Add a Railway volume and set `CHROMA_PERSIST_DIR` to the volume mount path so ChromaDB data survives redeploys.
5. Railway auto-deploys on each push to `main`.

---

## License

This project is part of the Gauntlet AI G4 program.
