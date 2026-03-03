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
    CHROMA_PERSIST_DIR, REPO_BASE_URL                        — deployment config
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
BM25_FALLBACK_THRESHOLD: int = 3

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
INGESTION_BATCH_SIZE: int = 200

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL: str = "gpt-4o-mini"
LLM_MAX_TOKENS: int = 1000
MAX_RETRIES: int = 3
CONFIDENCE_THRESHOLD: float = 0.70

# ── Ingestion Performance ──────────────────────────────────────────────────────
MAX_INGESTION_MINUTES: int = 5

# ── Repository — sourced from environment so no values are hardcoded ───────────
REPO_BASE_URL: str = "https://github.com/{owner}/{repo}/blob/{commit}/{file_path}#L{line}"
DEFAULT_REPO_OWNER: str = os.getenv("REPO_OWNER", "")
DEFAULT_REPO_NAME: str = os.getenv("REPO_NAME", "")
DEFAULT_REPO_COMMIT: str = os.getenv("REPO_COMMIT", "")

# ── Vector Store ───────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

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
