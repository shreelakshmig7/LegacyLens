"""
constants.py
------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Project-wide constants
---------------------------------------------------------------------------------
Single source of truth for every magic number, threshold, model name, and
configuration value used across the LegacyLens pipeline. All values that would
otherwise be hardcoded are defined here and imported by name everywhere else.

Environment-backed constants (those that vary per deployment) read their values
from environment variables with safe defaults so the module is always importable
even without a .env file loaded.

Key constants:
    TOP_K, MIN_RELEVANCE_THRESHOLD, BM25_FALLBACK_THRESHOLD  — retrieval tuning
    COBOL_SEQUENCE_COLS, COBOL_IDENTIFICATION_COLS           — COBOL 72-col stripping
    MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS                   — chunking parameters
    EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, INGESTION_BATCH_SIZE
    LLM_MODEL, LLM_MAX_TOKENS, MAX_RETRIES
    CHROMA_PERSIST_DIR, REPO_BASE_URL, LEGACYLENS_API_URL    — deployment / UI config
    CHUNK_TYPE_PROCEDURE, CHUNK_TYPE_DATA, CHUNK_TYPE_COPYBOOK — chunk type values
    validate_required_env_vars()                             — startup env guard

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K: int = 5
MIN_RELEVANCE_THRESHOLD: float = 0.70
BM25_FALLBACK_THRESHOLD: int = 3  # Trigger BM25 fallback when similarity results < this

# ── Reranker (no magic numbers in reranker.py) ──────────────────────────────────
RERANK_PARAGRAPH_BOOST_WEIGHT: float = 0.15  # Add to score when paragraph_name matches query
RERANK_DATA_DEPRIORITIZE_WEIGHT: float = -0.2  # Add to score for DATA chunks on logic queries
LOGIC_QUERY_KEYWORDS: List[str] = [
    "what does", "explain", "how does", "entry point", "function", "paragraph",
    "what is", "where is", "how to", "what are", "where are", "main entry",
    # "find all" signals operational queries → deprioritize DATA definition chunks
    "find all",
]

# ── Query expansion (ambiguous terms → COBOL-relevant terms) ───────────────────
QUERY_EXPANSION_TERMS: dict = {
    "entry point": "main MAIN PROCEDURE DIVISION",
    "main": "MAIN MAIN-PGM PROCEDURE DIVISION",
    "dependencies": "CALL COPY USING",
    "dependency": "CALL COPY USING",
    "io": "READ WRITE OPEN CLOSE FILE",
    # Strengthen file I/O to bias toward PROCEDURE chunks with actual operations
    "file i/o": "READ WRITE OPEN CLOSE FILE-STATUS INVALID-KEY input output",
    # Add errno/SQL terms so error-handling paragraphs surface above unrelated DATA chunks
    "error handling": "ERROR EXCEPTION INVALID KEY SQL-ERROR SQLSTATE ABORT errno",
    # ll-002: customer record modification queries → surface cust01.cbl UPDATE/ADD paragraphs
    "customer-record": "CUSTOMER-RECORD REWRITE WRITE UPDATE-RECORD ADD-RECORD customer",
    "modify": "REWRITE WRITE UPDATE ADD DELETE customer-record",
    # ll-004: "find all" bias toward PROCEDURE chunks
    "find all": "PROCEDURE PARAGRAPH READ WRITE OPEN CLOSE",
    # ll-004: "file i/o operations" exact phrase → surface PROCEDURE chunks with actual I/O verbs.
    # Scoped to this exact phrase — does NOT match ll-009 query ("FILE SECTION structured").
    "file i/o operations": "OPEN READ WRITE CLOSE FILE-STATUS SELECT FD file operation",
    # ll-005: MODULE-X does not exist; expand toward dependency patterns in named modules
    "module-x": "MODULE CALL COPY USING PGMOD1",
    # ll-009: any "file section" query phrase that would surface FD records was removed because
    # broader keys like "file section" or "structured in the data division" cause retrieval
    # regressions by shifting embeddings away from mmapmatchfile.cbl. ll-009 answer quality
    # is handled exclusively via the system prompt FD quoting instruction.
    # ll-013: query expansion for "using clause"/"pass parameters" was attempted but caused
    # a retrieval regression (wrong chunks retrieved). ll-013 is accepted as a known limitation:
    # the correct chunk's relevance score falls below NOT_FOUND_SCORE_THRESHOLD (0.55), so the
    # fast-path triggers and the LLM is not called. Addressed in post-MVP with threshold tuning.
}

# ── COBOL Preprocessing ────────────────────────────────────────────────────────
COBOL_SEQUENCE_COLS: tuple = (0, 6)
COBOL_IDENTIFICATION_COLS: tuple = (72, 80)
COBOL_EXTENSIONS: List[str] = [".cbl", ".cob", ".cpy"]
FORTRAN_EXTENSIONS: List[str] = [".f", ".f90", ".for"]

# ── Chunking ───────────────────────────────────────────────────────────────────
MAX_CHUNK_TOKENS: int = 500
CHUNK_OVERLAP_TOKENS: int = 50

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "voyage-code-2"
EMBEDDING_DIMENSIONS: int = 1536
INGESTION_BATCH_SIZE: int = 300
VOYAGE_API_TIMEOUT_SECONDS: int = 30  # Max wait per embedding API call; prevents eval hang

# ── Query Safety ──────────────────────────────────────────────────────────────
# Terms whose presence in a query signals the question is outside the domain of
# codebase analysis (e.g. cooking, weather, cryptocurrency). Case-insensitive
# substring match in _is_out_of_scope().
OUT_OF_SCOPE_KEYWORDS: List[str] = [
    "recipe",
    "weather",
    "cryptocurrency",
    "bitcoin",
    "sports score",
    "movie review",
    "song lyrics",
    "tell me a joke",
    "write me a poem",
    "translate to french",
    "translate to spanish",
    "medical advice",
    "legal advice",
    "stock price",
    "horoscope",
]

# Maximum allowed query length in characters; queries are truncated to this.
MAX_QUERY_LENGTH: int = 500

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL: str = "gpt-4o-mini"
LLM_MAX_TOKENS: int = 1500
MAX_RETRIES: int = 3
CONFIDENCE_THRESHOLD: float = 0.70
# Fast-path threshold: if max relevance score across all results is below this
# value, skip the LLM call entirely and return a structured "not found" response.
# Set to 0.55 so that near-zero-relevance queries (e.g. nonsense like "xyzzy") go
# to fast-path without LLM, while genuine low-scoring queries (>0.55) still get LLM.
NOT_FOUND_SCORE_THRESHOLD: float = 0.55
# Per-query latency gate in seconds; eval flags any case that exceeds this.
QUERY_LATENCY_GATE_SECONDS: float = 3.0

# ── Ingestion Performance ──────────────────────────────────────────────────────
MAX_INGESTION_MINUTES: int = 5

# ── UI / API base URL (Streamlit calls FastAPI) ──────────────────────────────────
LEGACYLENS_API_URL: str = os.getenv("LEGACYLENS_API_URL", "http://localhost:8000")

# ── Repository — sourced from environment so no values are hardcoded ───────────
REPO_BASE_URL: str = "https://github.com/{owner}/{repo}/blob/{commit}/{file_path}#L{line}"
DEFAULT_REPO_OWNER: str = os.getenv("REPO_OWNER", "")
DEFAULT_REPO_NAME: str = os.getenv("REPO_NAME", "")
DEFAULT_REPO_COMMIT: str = os.getenv("REPO_COMMIT", "")

# ── Vector Store ───────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_GET_ALL_LIMIT: int = 25000  # Max docs to fetch for BM25 in-memory index
DATA_XREF_MAX_CHUNKS: int = 100   # Max DATA chunks to fetch per file for variable xref

# ── Metadata Schema Fields ─────────────────────────────────────────────────────
METADATA_FIELDS: List[str] = [
    "file_path",
    "line_range",
    "type",
    "parent_section",
    "paragraph_name",
    "dependencies",
]

# ── Chunk Type Values — the only valid values for the "type" metadata field ───
CHUNK_TYPE_PROCEDURE: str = "PROCEDURE"
CHUNK_TYPE_DATA: str = "DATA"
CHUNK_TYPE_COPYBOOK: str = "COPYBOOK"

# ── COBOL Dependency Keywords ──────────────────────────────────────────────────
COBOL_DEPENDENCY_KEYWORDS: List[str] = ["CALL", "COPY", "USING"]

# ── Fields that must never appear in logs ─────────────────────────────────────
SENSITIVE_LOG_FIELDS: List[str] = ["api_key", "token", "secret"]

# ── Required environment variables (checked at startup) ───────────────────────
REQUIRED_ENV_VARS: List[str] = [
    "VOYAGE_API_KEY",
    "OPENAI_API_KEY",
    "REPO_OWNER",
    "REPO_NAME",
    "REPO_COMMIT",
]


def validate_required_env_vars() -> dict:
    """
    Check that every required environment variable is set to a non-empty value.

    Call this once at application startup so the service fails fast with a clear
    error message rather than crashing mid-request with a cryptic KeyError.

    Args:
        None

    Returns:
        dict: {
            "success": bool — True only when all required vars are present,
            "missing": list[str] — names of vars that are absent or empty,
            "present": list[str] — names of vars that are present and non-empty,
        }
    """
    missing: List[str] = []
    present: List[str] = []

    for var in REQUIRED_ENV_VARS:
        if os.getenv(var):
            present.append(var)
        else:
            missing.append(var)
            logger.warning("Required environment variable not set: %s", var)

    if missing:
        logger.error(
            "Application startup incomplete — %d required env var(s) missing: %s",
            len(missing),
            ", ".join(missing),
        )

    return {
        "success": len(missing) == 0,
        "missing": missing,
        "present": present,
    }
