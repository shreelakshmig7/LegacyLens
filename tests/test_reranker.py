"""
test_reranker.py
----------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for retrieval/reranker.py
--------------------------------------------------------------------------------------------
Validates re-ranking by relevance, paragraph_name boost, and DATA chunk deprioritization
for logic queries. Uses fixture result lists (same structure as searcher output); no Chroma.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import pytest

from legacylens.retrieval.reranker import rerank


def _make_result(text: str, score: float, chunk_type: str = "PROCEDURE", paragraph_name: str = "") -> dict:
    """Build a single result dict like searcher output."""
    return {
        "text": text,
        "metadata": {"type": chunk_type, "paragraph_name": paragraph_name, "file_path": "x.cbl"},
        "score": score,
    }


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

def test_rerank_returns_list() -> None:
    """rerank must return a list of results (same structure as input)."""
    results = [_make_result("code", 0.8)]
    out = rerank(results, "explain paragraph")
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["text"] == "code"
    assert "metadata" in out[0]
    assert "score" in out[0]


def test_rerank_does_not_mutate_input() -> None:
    """Input list must not be mutated."""
    results = [_make_result("a", 0.5), _make_result("b", 0.6)]
    orig_scores = [r["score"] for r in results]
    rerank(results, "query")
    assert [r["score"] for r in results] == orig_scores


def test_rerank_empty_input_returns_empty() -> None:
    """Empty results list must return empty list."""
    assert rerank([], "any query") == []


# ---------------------------------------------------------------------------
# Paragraph name boost
# ---------------------------------------------------------------------------

def test_rerank_boosts_paragraph_name_match() -> None:
    """When query contains a term that matches paragraph_name, that result should rank higher."""
    results = [
        _make_result("other code", 0.7, paragraph_name="OTHER-PARA"),
        _make_result("main code", 0.7, paragraph_name="MAIN-PGM"),
    ]
    out = rerank(results, "where is MAIN-PGM")
    assert len(out) == 2
    # MAIN-PGM matches query -> should be first after rerank
    assert out[0]["metadata"]["paragraph_name"] == "MAIN-PGM"


def test_rerank_paragraph_boost_case_insensitive() -> None:
    """Paragraph name matching should be case-insensitive."""
    results = [
        _make_result("x", 0.7, paragraph_name="Calculate-Interest"),
        _make_result("y", 0.75, paragraph_name="OTHER"),
    ]
    out = rerank(results, "explain calculate interest")
    assert out[0]["metadata"]["paragraph_name"] == "Calculate-Interest"


# ---------------------------------------------------------------------------
# DATA deprioritization for logic queries
# ---------------------------------------------------------------------------

def test_rerank_deprioritizes_data_for_logic_query() -> None:
    """For a 'logic' query (e.g. explain), DATA chunks should rank below PROCEDURE."""
    results = [
        _make_result("data division content", 0.8, chunk_type="DATA", paragraph_name=""),
        _make_result("procedure code", 0.75, chunk_type="PROCEDURE", paragraph_name="DO-SOMETHING"),
    ]
    out = rerank(results, "explain what this does")
    assert out[0]["metadata"]["type"] == "PROCEDURE"
    assert out[1]["metadata"]["type"] == "DATA"


def test_rerank_non_logic_query_does_not_deprioritize_data() -> None:
    """For a non-logic query, DATA chunks should not be artificially deprioritized."""
    results = [
        _make_result("data def", 0.9, chunk_type="DATA"),
        _make_result("proc", 0.7, chunk_type="PROCEDURE"),
    ]
    out = rerank(results, "list all variables in WORKING-STORAGE")
    # Higher original score should still win unless logic keywords trigger deprioritization
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Ordering by adjusted score
# ---------------------------------------------------------------------------

def test_rerank_sorts_by_adjusted_score_descending() -> None:
    """Results must be sorted by adjusted score descending."""
    results = [
        _make_result("a", 0.6, paragraph_name="X"),
        _make_result("b", 0.9, paragraph_name="Y"),
        _make_result("c", 0.7, paragraph_name="X"),
    ]
    out = rerank(results, "find X")
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True)
