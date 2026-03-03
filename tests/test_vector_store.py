"""
test_vector_store.py
--------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for retrieval/vector_store.py
----------------------------------------------------------------------------------------------
Validates the ChromaDB wrapper without connecting to a real database. ChromaDB is mocked
so tests run offline and deterministically.

Tests cover: chunk insertion, zero-tolerance verification (expected vs actual count),
metadata sanitisation (special chars, SENSITIVE_LOG_FIELDS, path containment),
query input sanitisation (whitelist filter fields, reject bad operators),
similarity query return structure, and structured return format.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import pathlib
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = str(pathlib.Path(__file__).parent.parent / "data" / "gnucobol-contrib")


def _make_embedded_chunks(n: int) -> List[Dict[str, Any]]:
    """Produce n minimal embedded chunk dicts for testing."""
    return [
        {
            "text": f"MOVE WS-A-{i} TO WS-B.",
            "file_path": f"{REPO_ROOT}/test_{i}.cbl",
            "line_range": [i * 10 + 1, i * 10 + 5],
            "type": "PROCEDURE",
            "parent_section": "PROCEDURE DIVISION",
            "paragraph_name": f"PARA-{i:04d}",
            "dependencies": ["PROG-A"],
            "embedding": [0.1] * 1536,
        }
        for i in range(n)
    ]


def _make_mock_collection(count: int = 0) -> MagicMock:
    """Build a mock ChromaDB collection that simulates before/after insertion counts.

    count=n means: before insertion returns 0, after insertion returns n.
    This matches how insert_chunks calls collection.count() twice.
    """
    col = MagicMock()
    col.count.side_effect = [0, count]   # [before_insert, after_insert]
    col.query.return_value = {
        "ids": [["id-0", "id-1"]],
        "documents": [["code line A", "code line B"]],
        "metadatas": [[{"file_path": "a.cbl", "paragraph_name": "MAIN"}, {"file_path": "b.cbl", "paragraph_name": "CALC"}]],
        "distances": [[0.12, 0.25]],
    }
    return col


# ---------------------------------------------------------------------------
# insert_chunks — return structure
# ---------------------------------------------------------------------------

def test_insert_returns_success_dict() -> None:
    """insert_chunks must return a structured dict on success."""
    from legacylens.retrieval.vector_store import insert_chunks

    chunks = _make_embedded_chunks(3)
    mock_col = _make_mock_collection(count=3)

    with patch("legacylens.retrieval.vector_store._get_collection", return_value=mock_col):
        result = insert_chunks(chunks)

    assert isinstance(result, dict)
    assert result["success"] is True
    assert "data" in result
    assert result.get("error") is None


def test_insert_empty_chunks_returns_success() -> None:
    """insert_chunks must succeed immediately for an empty input list."""
    from legacylens.retrieval.vector_store import insert_chunks

    result = insert_chunks([])
    assert result["success"] is True
    assert result["data"]["inserted_count"] == 0


# ---------------------------------------------------------------------------
# Zero-tolerance verification
# ---------------------------------------------------------------------------

def test_insert_verification_passes_when_counts_match() -> None:
    """Verification must pass when ChromaDB count equals the number of chunks inserted."""
    from legacylens.retrieval.vector_store import insert_chunks

    n = 5
    chunks = _make_embedded_chunks(n)
    mock_col = _make_mock_collection(count=n)

    with patch("legacylens.retrieval.vector_store._get_collection", return_value=mock_col):
        result = insert_chunks(chunks)

    assert result["success"] is True
    assert result["data"]["verified"] is True


def test_insert_verification_fails_when_counts_differ() -> None:
    """Zero-tolerance: if actual count != expected, insert_chunks must return failure."""
    from legacylens.retrieval.vector_store import insert_chunks

    n = 5
    chunks = _make_embedded_chunks(n)
    # Simulate: before=0, after=n-1 (one chunk missing — should fail)
    mock_col = MagicMock()
    mock_col.count.side_effect = [0, n - 1]
    mock_col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    with patch("legacylens.retrieval.vector_store._get_collection", return_value=mock_col):
        result = insert_chunks(chunks)

    assert result["success"] is False
    assert "error" in result
    assert result.get("error") is not None


# ---------------------------------------------------------------------------
# Metadata sanitisation
# ---------------------------------------------------------------------------

def test_special_chars_sanitised_in_metadata() -> None:
    """sanitize_metadata must strip or escape special characters in string fields."""
    from legacylens.retrieval.vector_store import sanitize_metadata

    dirty = {
        "file_path": "/repo/file\x00name.cbl",
        "paragraph_name": "PARA<INJECTION>",
        "type": "PROCEDURE",
        "parent_section": "PROC\nDIV",
        "dependencies": ["DEP-A"],
        "line_range": [1, 10],
    }
    clean = sanitize_metadata(dirty)
    assert "\x00" not in clean["file_path"]
    assert "<" not in clean.get("paragraph_name", "")
    assert "\n" not in clean.get("parent_section", "")


def test_sensitive_fields_removed_from_log_metadata() -> None:
    """sanitize_metadata must remove any key listed in SENSITIVE_LOG_FIELDS."""
    from legacylens.config.constants import SENSITIVE_LOG_FIELDS
    from legacylens.retrieval.vector_store import sanitize_metadata

    dirty = {
        "file_path": "/repo/file.cbl",
        "paragraph_name": "MAIN",
        "type": "PROCEDURE",
        "parent_section": "PROC DIV",
        "dependencies": [],
        "line_range": [1, 5],
        "api_key": "secret-key-should-not-be-here",
        "token": "bearer-token",
    }
    clean = sanitize_metadata(dirty)
    for field in SENSITIVE_LOG_FIELDS:
        assert field not in clean, f"Sensitive field {field!r} not removed from metadata"


def test_path_outside_repo_sanitised() -> None:
    """sanitize_metadata must flag or sanitise file_path values outside repo root."""
    from legacylens.retrieval.vector_store import sanitize_metadata

    outside = {
        "file_path": "/etc/passwd",
        "paragraph_name": "MAIN",
        "type": "PROCEDURE",
        "parent_section": "",
        "dependencies": [],
        "line_range": [1, 1],
    }
    clean = sanitize_metadata(outside, repo_root=REPO_ROOT)
    # Either the path is flagged/blanked or the function marks it unsafe
    assert clean.get("file_path") != "/etc/passwd" or clean.get("_path_unsafe") is True


# ---------------------------------------------------------------------------
# Query input sanitisation
# ---------------------------------------------------------------------------

def test_query_sanitisation_rejects_disallowed_filter_fields() -> None:
    """sanitize_query_filters must reject filter dicts containing non-whitelisted fields."""
    from legacylens.retrieval.vector_store import sanitize_query_filters

    bad_filter = {"__collection__": "drop_table", "file_path": "test.cbl"}
    result = sanitize_query_filters(bad_filter)
    assert result["success"] is False
    assert "error" in result


def test_query_sanitisation_accepts_valid_filter_fields() -> None:
    """sanitize_query_filters must accept filters using only whitelisted fields."""
    from legacylens.retrieval.vector_store import sanitize_query_filters

    good_filter = {"file_path": "test.cbl", "type": "PROCEDURE"}
    result = sanitize_query_filters(good_filter)
    assert result["success"] is True


def test_query_sanitisation_rejects_operator_injection() -> None:
    """sanitize_query_filters must reject filters that include unexpected operator keys."""
    from legacylens.retrieval.vector_store import sanitize_query_filters

    injected = {"$where": "1=1", "file_path": "test.cbl"}
    result = sanitize_query_filters(injected)
    assert result["success"] is False


# ---------------------------------------------------------------------------
# query_similar — similarity search return structure
# ---------------------------------------------------------------------------

def test_query_similar_returns_success_dict() -> None:
    """query_similar must return a structured dict with results list."""
    from legacylens.retrieval.vector_store import query_similar

    mock_col = _make_mock_collection()
    query_vector = [0.1] * 1536

    with patch("legacylens.retrieval.vector_store._get_collection", return_value=mock_col):
        result = query_similar(query_vector, top_k=2)

    assert isinstance(result, dict)
    assert result["success"] is True
    assert "data" in result
    assert "results" in result["data"]
    assert isinstance(result["data"]["results"], list)


def test_query_similar_results_have_expected_fields() -> None:
    """Each result from query_similar must have text, metadata, and score fields."""
    from legacylens.retrieval.vector_store import query_similar

    mock_col = _make_mock_collection()

    with patch("legacylens.retrieval.vector_store._get_collection", return_value=mock_col):
        result = query_similar([0.1] * 1536, top_k=2)

    for r in result["data"]["results"]:
        assert "text" in r
        assert "metadata" in r
        assert "score" in r


def test_query_similar_respects_top_k() -> None:
    """query_similar must pass top_k to ChromaDB and return at most top_k results."""
    from legacylens.retrieval.vector_store import query_similar

    mock_col = _make_mock_collection()

    with patch("legacylens.retrieval.vector_store._get_collection", return_value=mock_col):
        query_similar([0.1] * 1536, top_k=3)

    call_kwargs = mock_col.query.call_args
    assert call_kwargs is not None
    called_n = call_kwargs[1].get("n_results") or call_kwargs[0][1] if call_kwargs[0] else None
    if called_n is None:
        # Check keyword args
        called_n = mock_col.query.call_args.kwargs.get("n_results")
    assert called_n == 3
