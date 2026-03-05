"""
test_embedder.py
----------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for ingestion/embedder.py
-------------------------------------------------------------------------------------------
Validates the embedding pipeline without making real API calls. The Voyage AI client
is mocked so tests run offline and deterministically.

Tests cover: batch sizing, token pre-validation (reject oversized chunks), structured
return format, exponential backoff retry logic on API errors, empty input handling,
and correct attachment of embedding vectors to chunk dicts.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunks(n: int, text: str = "MOVE WS-A TO WS-B.") -> list:
    """Produce n minimal chunk dicts for testing."""
    return [
        {
            "text": text,
            "file_path": f"test_{i}.cbl",
            "line_range": [i, i + 1],
            "type": "PROCEDURE",
            "parent_section": "PROCEDURE DIVISION",
            "paragraph_name": f"PARA-{i:04d}",
            "dependencies": [],
        }
        for i in range(n)
    ]


def _fake_embed_response(texts: List[str]) -> MagicMock:
    """Build a mock Voyage embed response that returns 1536-dim zero vectors."""
    mock_response = MagicMock()
    mock_response.embeddings = [[0.0] * 1536 for _ in texts]
    return mock_response


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

def test_embed_returns_success_dict() -> None:
    """embed_chunks must return a structured dict on success."""
    from legacylens.ingestion.embedder import embed_chunks

    chunks = _make_chunks(3)
    mock_client = MagicMock()
    mock_client.embed.return_value = _fake_embed_response([c["text"] for c in chunks])

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        result = embed_chunks(chunks)

    assert isinstance(result, dict)
    assert result["success"] is True
    assert "data" in result
    assert result.get("error") is None


def test_embed_attaches_vectors_to_chunks() -> None:
    """Each returned chunk must have an 'embedding' key with a list of floats."""
    from legacylens.ingestion.embedder import embed_chunks

    chunks = _make_chunks(3)
    mock_client = MagicMock()
    mock_client.embed.return_value = _fake_embed_response([c["text"] for c in chunks])

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        result = embed_chunks(chunks)

    for chunk in result["data"]["chunks"]:
        assert "embedding" in chunk
        assert isinstance(chunk["embedding"], list)
        assert len(chunk["embedding"]) == 1536


def test_embed_empty_input_returns_success() -> None:
    """embed_chunks must return success with empty list for empty input."""
    from legacylens.ingestion.embedder import embed_chunks

    result = embed_chunks([])
    assert result["success"] is True
    assert result["data"]["chunks"] == []


# ---------------------------------------------------------------------------
# Batch sizing
# ---------------------------------------------------------------------------

def test_embed_uses_correct_batch_size() -> None:
    """embed_chunks must call the API in batches of INGESTION_BATCH_SIZE."""
    from legacylens.config.constants import INGESTION_BATCH_SIZE
    from legacylens.ingestion.embedder import embed_chunks

    n = INGESTION_BATCH_SIZE + 50   # forces at least 2 batches
    chunks = _make_chunks(n)

    mock_client = MagicMock()
    def side_effect(texts, model, **kwargs):
        return _fake_embed_response(texts)
    mock_client.embed.side_effect = side_effect

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        result = embed_chunks(chunks)

    assert result["success"] is True
    # Should have been called at least 2 times (ceil(n / INGESTION_BATCH_SIZE))
    import math
    expected_calls = math.ceil(n / INGESTION_BATCH_SIZE)
    assert mock_client.embed.call_count == expected_calls


def test_embed_all_chunks_returned() -> None:
    """Total chunks returned must equal total chunks input."""
    from legacylens.ingestion.embedder import embed_chunks

    n = 5
    chunks = _make_chunks(n)
    mock_client = MagicMock()
    mock_client.embed.side_effect = lambda texts, **kwargs: _fake_embed_response(texts)

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        result = embed_chunks(chunks)

    assert len(result["data"]["chunks"]) == n


# ---------------------------------------------------------------------------
# Token pre-validation — oversized chunks must be sub-split, never dropped
# ---------------------------------------------------------------------------

def test_oversized_chunk_is_subsplit_not_skipped() -> None:
    """An oversized chunk must be sub-split into valid sub-chunks and embedded — not discarded."""
    from legacylens.config.constants import MAX_CHUNK_TOKENS
    from legacylens.ingestion.embedder import embed_chunks

    oversized_text = " ".join([f"WORD-{i:05d}" for i in range(MAX_CHUNK_TOKENS * 2)])
    chunks = [
        {
            "text": oversized_text,
            "file_path": "big.cbl",
            "line_range": [1, 100],
            "type": "PROCEDURE",
            "parent_section": "PROCEDURE DIVISION",
            "paragraph_name": "HUGE-PARA",
            "dependencies": [],
        }
    ]

    mock_client = MagicMock()
    mock_client.embed.side_effect = lambda texts, **kwargs: _fake_embed_response(texts)

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        result = embed_chunks(chunks)

    assert result["success"] is True
    # Must produce at least 2 sub-chunks (one oversized chunk → multiple valid ones)
    assert len(result["data"]["chunks"]) >= 2
    # Nothing discarded — skipped_count must be 0
    assert result["data"]["skipped_count"] == 0
    # Every sub-chunk must have an embedding and be within the token limit
    for sub in result["data"]["chunks"]:
        assert "embedding" in sub
        est = max(1, int(len(sub["text"].split()) / 0.75))
        assert est <= MAX_CHUNK_TOKENS, f"Sub-chunk still oversized: {est} tokens"


def test_subsplit_preserves_chunk_metadata() -> None:
    """Every sub-chunk produced from an oversized split must inherit the original metadata."""
    from legacylens.config.constants import MAX_CHUNK_TOKENS
    from legacylens.ingestion.embedder import embed_chunks

    oversized_text = " ".join([f"WORD-{i:05d}" for i in range(MAX_CHUNK_TOKENS * 2)])
    chunks = [
        {
            "text": oversized_text,
            "file_path": "meta_test.cbl",
            "line_range": [5, 50],
            "type": "PROCEDURE",
            "parent_section": "PROCEDURE DIVISION",
            "paragraph_name": "META-PARA",
            "dependencies": ["PROG-X"],
        }
    ]

    mock_client = MagicMock()
    mock_client.embed.side_effect = lambda texts, **kwargs: _fake_embed_response(texts)

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        result = embed_chunks(chunks)

    assert result["success"] is True
    assert len(result["data"]["chunks"]) >= 2, "Oversized chunk must produce multiple sub-chunks"
    for sub in result["data"]["chunks"]:
        assert sub["file_path"] == "meta_test.cbl"
        assert sub["type"] == "PROCEDURE"
        assert sub["paragraph_name"] == "META-PARA"
        assert sub["dependencies"] == ["PROG-X"]


def test_subsplit_count_reported_in_result() -> None:
    """embed_chunks result must include subsplit_count tracking how many chunks were split."""
    from legacylens.config.constants import MAX_CHUNK_TOKENS
    from legacylens.ingestion.embedder import embed_chunks

    oversized_text = " ".join([f"WORD-{i:05d}" for i in range(MAX_CHUNK_TOKENS * 2)])
    normal_text = "MOVE WS-A TO WS-B."
    chunks = [
        {"text": oversized_text, "file_path": "big.cbl", "line_range": [1, 100],
         "type": "PROCEDURE", "parent_section": "PROC DIV", "paragraph_name": "HUGE", "dependencies": []},
        {"text": normal_text, "file_path": "small.cbl", "line_range": [1, 1],
         "type": "PROCEDURE", "parent_section": "PROC DIV", "paragraph_name": "SMALL", "dependencies": []},
    ]

    mock_client = MagicMock()
    mock_client.embed.side_effect = lambda texts, **kwargs: _fake_embed_response(texts)

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        result = embed_chunks(chunks)

    assert result["success"] is True
    assert "subsplit_count" in result["data"]
    # Exactly one of the two input chunks was oversized → subsplit_count == 1
    assert result["data"]["subsplit_count"] == 1


# ---------------------------------------------------------------------------
# Retry logic on API errors
# ---------------------------------------------------------------------------

def test_api_error_triggers_retry() -> None:
    """Transient API errors must trigger exponential backoff retries."""
    from legacylens.ingestion.embedder import embed_chunks

    chunks = _make_chunks(1)
    mock_client = MagicMock()

    call_count = {"n": 0}
    def flaky_embed(texts, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise Exception("Simulated API rate limit error")
        return _fake_embed_response(texts)

    mock_client.embed.side_effect = flaky_embed

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        with patch("legacylens.ingestion.embedder.time.sleep"):  # skip real sleep
            result = embed_chunks(chunks)

    # After 2 failures + 1 success, should succeed
    assert result["success"] is True
    assert mock_client.embed.call_count == 3


def test_api_error_exceeds_max_retries_returns_failure() -> None:
    """If all retries are exhausted, embed_chunks must return success=False."""
    from legacylens.config.constants import MAX_RETRIES
    from legacylens.ingestion.embedder import embed_chunks

    chunks = _make_chunks(1)
    mock_client = MagicMock()
    mock_client.embed.side_effect = Exception("Persistent API failure")

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        with patch("legacylens.ingestion.embedder.time.sleep"):
            result = embed_chunks(chunks)

    assert result["success"] is False
    assert result.get("error") is not None
    # Must have retried MAX_RETRIES times (not more, not fewer)
    assert mock_client.embed.call_count == MAX_RETRIES


# ---------------------------------------------------------------------------
# Parallel embedding worker path (EMBEDDING_BATCH_WORKERS > 1)
# ---------------------------------------------------------------------------

def test_parallel_workers_embed_all_chunks() -> None:
    """When EMBEDDING_BATCH_WORKERS > 1, all chunks must still be embedded correctly."""
    from legacylens.config.constants import INGESTION_BATCH_SIZE
    from legacylens.ingestion.embedder import embed_chunks

    # Force parallel path by patching EMBEDDING_BATCH_WORKERS to 2
    n = INGESTION_BATCH_SIZE + 10   # forces at least 2 batches
    chunks = _make_chunks(n)

    mock_client = MagicMock()
    mock_client.embed.side_effect = lambda texts, **kwargs: _fake_embed_response(texts)

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        with patch("legacylens.ingestion.embedder.EMBEDDING_BATCH_WORKERS", 2):
            result = embed_chunks(chunks)

    assert result["success"] is True
    # All chunks must have embeddings
    assert len(result["data"]["chunks"]) == n
    for chunk in result["data"]["chunks"]:
        assert "embedding" in chunk
        assert len(chunk["embedding"]) == 1536


def test_parallel_workers_preserve_chunk_order() -> None:
    """Parallel embedding must return chunks in the same order as the input."""
    from legacylens.config.constants import INGESTION_BATCH_SIZE
    from legacylens.ingestion.embedder import embed_chunks

    n = INGESTION_BATCH_SIZE + 10
    # Give each chunk a unique paragraph_name to verify order preservation
    chunks = [
        {
            "text": f"MOVE WS-{i:04d} TO OUT.",
            "file_path": "order_test.cbl",
            "line_range": [i, i + 1],
            "type": "PROCEDURE",
            "parent_section": "PROCEDURE DIVISION",
            "paragraph_name": f"PARA-{i:04d}",
            "dependencies": [],
        }
        for i in range(n)
    ]

    mock_client = MagicMock()
    mock_client.embed.side_effect = lambda texts, **kwargs: _fake_embed_response(texts)

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        with patch("legacylens.ingestion.embedder.EMBEDDING_BATCH_WORKERS", 2):
            result = embed_chunks(chunks)

    assert result["success"] is True
    returned_names = [c["paragraph_name"] for c in result["data"]["chunks"]]
    expected_names = [f"PARA-{i:04d}" for i in range(n)]
    assert returned_names == expected_names, "Parallel embedding must preserve chunk order"


# ---------------------------------------------------------------------------
# Original chunks not mutated
# ---------------------------------------------------------------------------

def test_embed_does_not_mutate_input_chunks() -> None:
    """embed_chunks must not mutate the original input chunk dicts."""
    from legacylens.ingestion.embedder import embed_chunks

    chunks = _make_chunks(2)
    original_keys = set(chunks[0].keys())

    mock_client = MagicMock()
    mock_client.embed.side_effect = lambda texts, **kwargs: _fake_embed_response(texts)

    with patch("legacylens.ingestion.embedder._get_voyage_client", return_value=mock_client):
        embed_chunks(chunks)

    # Original dicts must not have gained an 'embedding' key
    assert "embedding" not in chunks[0]
    assert set(chunks[0].keys()) == original_keys
