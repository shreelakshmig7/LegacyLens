"""
test_constants.py
-----------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for config/constants.py
-----------------------------------------------------------------------------------------
Validates that all project constants exist with correct types and that environment-backed
constants read correctly from the environment with proper fallback defaults.

Functions:
    test_retrieval_constants_exist
    test_cobol_constants_exist
    test_chunking_constants_exist
    test_embedding_constants_exist
    test_llm_constants_exist
    test_repository_constants_exist
    test_metadata_fields_exist
    test_chroma_persist_dir_default
    test_chroma_persist_dir_from_env
    test_validate_required_env_vars_all_missing
    test_validate_required_env_vars_all_present
    test_validate_required_env_vars_partial

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import importlib
import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_constants():
    """Reload constants module so os.getenv picks up patched env vars."""
    import legacylens.config.constants as mod
    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# Static constant existence & type tests
# ---------------------------------------------------------------------------

def test_retrieval_constants_exist() -> None:
    """Verify retrieval constants exist and have correct types."""
    from legacylens.config.constants import (
        BM25_FALLBACK_THRESHOLD,
        MIN_RELEVANCE_THRESHOLD,
        TOP_K,
    )

    assert isinstance(TOP_K, int)
    assert isinstance(MIN_RELEVANCE_THRESHOLD, float)
    assert isinstance(BM25_FALLBACK_THRESHOLD, int)
    assert TOP_K > 0
    assert 0.0 < MIN_RELEVANCE_THRESHOLD < 1.0
    assert BM25_FALLBACK_THRESHOLD > 0


def test_cobol_constants_exist() -> None:
    """Verify COBOL preprocessing constants exist and have correct types."""
    from legacylens.config.constants import (
        COBOL_DEPENDENCY_KEYWORDS,
        COBOL_EXTENSIONS,
        COBOL_IDENTIFICATION_COLS,
        COBOL_SEQUENCE_COLS,
        FORTRAN_EXTENSIONS,
    )

    assert isinstance(COBOL_SEQUENCE_COLS, tuple)
    assert isinstance(COBOL_IDENTIFICATION_COLS, tuple)
    assert isinstance(COBOL_EXTENSIONS, list)
    assert isinstance(FORTRAN_EXTENSIONS, list)
    assert isinstance(COBOL_DEPENDENCY_KEYWORDS, list)
    assert len(COBOL_SEQUENCE_COLS) == 2
    assert len(COBOL_IDENTIFICATION_COLS) == 2
    assert ".cbl" in COBOL_EXTENSIONS
    assert "CALL" in COBOL_DEPENDENCY_KEYWORDS


def test_chunking_constants_exist() -> None:
    """Verify chunking constants exist with correct types and sensible values."""
    from legacylens.config.constants import CHUNK_OVERLAP_TOKENS, MAX_CHUNK_TOKENS

    assert isinstance(MAX_CHUNK_TOKENS, int)
    assert isinstance(CHUNK_OVERLAP_TOKENS, int)
    assert MAX_CHUNK_TOKENS > 0
    assert CHUNK_OVERLAP_TOKENS >= 0
    assert CHUNK_OVERLAP_TOKENS < MAX_CHUNK_TOKENS


def test_embedding_constants_exist() -> None:
    """Verify embedding constants exist with correct types."""
    from legacylens.config.constants import (
        EMBEDDING_DIMENSIONS,
        EMBEDDING_MODEL,
        INGESTION_BATCH_SIZE,
    )

    assert isinstance(EMBEDDING_MODEL, str)
    assert isinstance(EMBEDDING_DIMENSIONS, int)
    assert isinstance(INGESTION_BATCH_SIZE, int)
    assert len(EMBEDDING_MODEL) > 0
    assert EMBEDDING_DIMENSIONS > 0
    assert INGESTION_BATCH_SIZE > 0


def test_llm_constants_exist() -> None:
    """Verify LLM constants exist with correct types and sensible values."""
    from legacylens.config.constants import (
        CONFIDENCE_THRESHOLD,
        LLM_MAX_TOKENS,
        LLM_MODEL,
        MAX_INGESTION_MINUTES,
        MAX_RETRIES,
    )

    assert isinstance(LLM_MODEL, str)
    assert isinstance(LLM_MAX_TOKENS, int)
    assert isinstance(MAX_RETRIES, int)
    assert isinstance(CONFIDENCE_THRESHOLD, float)
    assert isinstance(MAX_INGESTION_MINUTES, int)
    assert len(LLM_MODEL) > 0
    assert LLM_MAX_TOKENS > 0
    assert MAX_RETRIES >= 1
    assert 0.0 < CONFIDENCE_THRESHOLD < 1.0


def test_repository_constants_exist() -> None:
    """Verify repository URL template and sensitive log fields exist."""
    from legacylens.config.constants import REPO_BASE_URL, SENSITIVE_LOG_FIELDS

    assert isinstance(REPO_BASE_URL, str)
    assert "{file_path}" in REPO_BASE_URL
    assert "{line}" in REPO_BASE_URL
    assert isinstance(SENSITIVE_LOG_FIELDS, list)
    assert "api_key" in SENSITIVE_LOG_FIELDS


def test_metadata_fields_exist() -> None:
    """Verify METADATA_FIELDS contains all required schema fields."""
    from legacylens.config.constants import METADATA_FIELDS

    assert isinstance(METADATA_FIELDS, list)
    required = {"file_path", "line_range", "type", "parent_section", "paragraph_name", "dependencies"}
    assert required.issubset(set(METADATA_FIELDS))


# ---------------------------------------------------------------------------
# Environment-backed constant tests
# ---------------------------------------------------------------------------

def test_chroma_persist_dir_default() -> None:
    """CHROMA_PERSIST_DIR must fall back to './chroma_db' when env var is absent."""
    env_without_chroma = {k: v for k, v in os.environ.items() if k != "CHROMA_PERSIST_DIR"}
    with patch.dict(os.environ, env_without_chroma, clear=True):
        mod = _reload_constants()
        assert mod.CHROMA_PERSIST_DIR == "./chroma_db"


def test_chroma_persist_dir_from_env() -> None:
    """CHROMA_PERSIST_DIR must read the value from the environment variable when set."""
    with patch.dict(os.environ, {"CHROMA_PERSIST_DIR": "/mnt/volumes/chroma"}):
        mod = _reload_constants()
        assert mod.CHROMA_PERSIST_DIR == "/mnt/volumes/chroma"


# ---------------------------------------------------------------------------
# validate_required_env_vars tests (genuinely useful startup guard)
# ---------------------------------------------------------------------------

_REQUIRED_VARS = ["VOYAGE_API_KEY", "OPENAI_API_KEY", "REPO_OWNER", "REPO_NAME", "REPO_COMMIT"]


def test_validate_required_env_vars_all_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation must fail and list every missing var when none are set."""
    for var in _REQUIRED_VARS:
        monkeypatch.delenv(var, raising=False)

    from legacylens.config.constants import validate_required_env_vars

    result = validate_required_env_vars()

    assert result["success"] is False
    assert set(_REQUIRED_VARS).issubset(set(result["missing"]))
    assert result["present"] == []


def test_validate_required_env_vars_all_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation must succeed when all required vars are set to non-empty values."""
    monkeypatch.setenv("VOYAGE_API_KEY", "voy-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("REPO_OWNER", "test-owner")
    monkeypatch.setenv("REPO_NAME", "opencobol-contrib")
    monkeypatch.setenv("REPO_COMMIT", "abc123def456")

    from legacylens.config.constants import validate_required_env_vars

    result = validate_required_env_vars()

    assert result["success"] is True
    assert result["missing"] == []
    assert set(_REQUIRED_VARS).issubset(set(result["present"]))


def test_validate_required_env_vars_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation must report only the actually missing vars, not all of them."""
    monkeypatch.setenv("VOYAGE_API_KEY", "voy-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("REPO_OWNER", raising=False)
    monkeypatch.delenv("REPO_NAME", raising=False)
    monkeypatch.delenv("REPO_COMMIT", raising=False)

    from legacylens.config.constants import validate_required_env_vars

    result = validate_required_env_vars()

    assert result["success"] is False
    assert "REPO_OWNER" in result["missing"]
    assert "REPO_NAME" in result["missing"]
    assert "REPO_COMMIT" in result["missing"]
    assert "VOYAGE_API_KEY" not in result["missing"]
    assert "OPENAI_API_KEY" not in result["missing"]
    assert "VOYAGE_API_KEY" in result["present"]
    assert "OPENAI_API_KEY" in result["present"]


# ---------------------------------------------------------------------------
# Chunk type constants (PR 2 additions)
# ---------------------------------------------------------------------------

def test_chunk_type_constants_exist() -> None:
    """CHUNK_TYPE_* constants must exist as non-empty strings with expected values."""
    from legacylens.config.constants import (
        CHUNK_TYPE_COPYBOOK,
        CHUNK_TYPE_DATA,
        CHUNK_TYPE_PROCEDURE,
    )

    assert isinstance(CHUNK_TYPE_PROCEDURE, str) and CHUNK_TYPE_PROCEDURE == "PROCEDURE"
    assert isinstance(CHUNK_TYPE_DATA, str) and CHUNK_TYPE_DATA == "DATA"
    assert isinstance(CHUNK_TYPE_COPYBOOK, str) and CHUNK_TYPE_COPYBOOK == "COPYBOOK"


def test_chunk_types_are_distinct() -> None:
    """All three chunk type values must be distinct strings."""
    from legacylens.config.constants import (
        CHUNK_TYPE_COPYBOOK,
        CHUNK_TYPE_DATA,
        CHUNK_TYPE_PROCEDURE,
    )

    types = {CHUNK_TYPE_PROCEDURE, CHUNK_TYPE_DATA, CHUNK_TYPE_COPYBOOK}
    assert len(types) == 3


def test_ingestion_batch_size_updated() -> None:
    """INGESTION_BATCH_SIZE must be 200 to halve API round-trips for ~6,700 chunks."""
    from legacylens.config.constants import INGESTION_BATCH_SIZE

    assert INGESTION_BATCH_SIZE == 200
