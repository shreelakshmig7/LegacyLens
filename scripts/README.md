# Scripts

## Run Ingestion (Railway)

Use `scripts/run_ingestion.py` as a one-shot ingestion job to populate ChromaDB without using FastAPI admin endpoints.

### Required environment variables

- `VOYAGE_API_KEY`
- `OPENAI_API_KEY`
- `REPO_OWNER`
- `REPO_NAME`
- `REPO_COMMIT` (preferred pinned SHA)  
  If you only provide `REPO_BRANCH`, the script will map it to `REPO_COMMIT` at runtime.

### Recommended environment variables

- `CHROMA_PERSIST_DIR=/data/chroma_db` (Railway volume path)
- `REPO_PATH=/data/<repo-name>` (extraction target on Railway volume)
- `REPO_BRANCH=<branch-name>` (optional fallback when `REPO_COMMIT` is not set)

### Command

```bash
python3 scripts/run_ingestion.py
```

### Railway-safe run pattern

- Run this script as a manual one-off job after deploy or when re-indexing.
- Keep the same `CHROMA_PERSIST_DIR` volume path across runs for persistence.
- Prefer pinning `REPO_COMMIT` to avoid ingesting moving targets.
- Re-running is idempotent at file level: already-ingested file paths are skipped.
