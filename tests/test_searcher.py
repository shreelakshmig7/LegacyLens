"""
test_searcher.py
----------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for retrieval/searcher.py
--------------------------------------------------------------------------------------------
Runs against real ChromaDB (CHROMA_PERSIST_DIR) and real Voyage Code 2 embeddings. No mocks.
Requires VOYAGE_API_KEY and populated ChromaDB (e.g. from ingestion).

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import os

import pytest

# Load .env so VOYAGE_API_KEY and CHROMA_PERSIST_DIR are set when running from project root
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from legacylens.config.constants import CHROMA_PERSIST_DIR
from legacylens.retrieval.searcher import search


def _env_ready() -> bool:
    """True if Voyage key, ChromaDB package, and ChromaDB persist dir are available for real tests."""
    if not os.getenv("VOYAGE_API_KEY"):
        return False
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return os.path.isdir(CHROMA_PERSIST_DIR)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _env_ready(), reason="VOYAGE_API_KEY and/or ChromaDB persist dir not available")
def test_search_returns_success_dict() -> None:
    """search must return dict with success, data, error."""
    result = search("Where is the main entry point of this program?")
    assert isinstance(result, dict)
    assert "success" in result
    assert "data" in result
    assert "error" in result
    assert result["success"] is True
    assert result["error"] is None


@pytest.mark.skipif(not _env_ready(), reason="VOYAGE_API_KEY and/or ChromaDB persist dir not available")
def test_search_data_has_results_list() -> None:
    """data must contain results list."""
    result = search("Where is MAIN-PGM or main entry?")
    assert result["success"] is True
    assert "results" in result["data"]
    assert isinstance(result["data"]["results"], list)


@pytest.mark.skipif(not _env_ready(), reason="VOYAGE_API_KEY and/or ChromaDB persist dir not available")
def test_search_results_have_text_metadata_score() -> None:
    """Each result must have text, metadata, score."""
    result = search("Explain PROCEDURE DIVISION")
    assert result["success"] is True
    results = result["data"]["results"]
    for item in results:
        assert "text" in item
        assert "metadata" in item
        assert "score" in item
        assert isinstance(item["metadata"], dict)
        assert isinstance(item["score"], (int, float))


@pytest.mark.skipif(not _env_ready(), reason="VOYAGE_API_KEY and/or ChromaDB persist dir not available")
def test_search_respects_top_k() -> None:
    """At most TOP_K (5) results must be returned."""
    result = search("What does the program do?")
    assert result["success"] is True
    assert len(result["data"]["results"]) <= 5


@pytest.mark.skipif(not _env_ready(), reason="VOYAGE_API_KEY and/or ChromaDB persist dir not available")
def test_search_metadata_has_file_path_and_paragraph_name() -> None:
    """Metadata should contain file_path and paragraph_name when present in Chroma."""
    result = search("Where is the main entry point?")
    assert result["success"] is True
    results = result["data"]["results"]
    assert len(results) > 0
    first = results[0]
    meta = first["metadata"]
    assert "file_path" in meta or "paragraph_name" in meta or "type" in meta


@pytest.mark.skipif(not _env_ready(), reason="VOYAGE_API_KEY and/or ChromaDB persist dir not available")
def test_search_returns_at_least_one_result_for_known_topic() -> None:
    """A query about the codebase should return at least one result (real index)."""
    result = search("main entry point PROCEDURE DIVISION")
    assert result["success"] is True
    assert len(result["data"]["results"]) >= 1


# ---------------------------------------------------------------------------
# Error / edge
# ---------------------------------------------------------------------------

def test_search_empty_query_returns_structured_result() -> None:
    """Empty query must not crash; return success with empty results or failure."""
    result = search("")
    assert isinstance(result, dict)
    assert "success" in result
    if result["success"]:
        assert isinstance(result["data"]["results"], list)
    else:
        assert "error" in result and result["error"]
