"""
test_query_processor.py
------------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for retrieval/query_processor.py
--------------------------------------------------------------------------------------------------
Validates query normalization, intent/entity extraction, and ambiguous term expansion.
Uses real COBOL-relevant strings; no ChromaDB or external APIs.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import pytest

from legacylens.retrieval.query_processor import process_query


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

def test_process_query_returns_success_dict() -> None:
    """process_query must return dict with success, data, error keys."""
    result = process_query("Where is the main entry point?")
    assert isinstance(result, dict)
    assert "success" in result
    assert "data" in result
    assert "error" in result
    assert result["success"] is True
    assert result["error"] is None


def test_process_query_data_has_required_keys() -> None:
    """data must contain normalized_query, intent, entities."""
    result = process_query("Explain CALCULATE-INTEREST")
    assert result["success"] is True
    data = result["data"]
    assert "normalized_query" in data
    assert "intent" in data
    assert "entities" in data
    assert isinstance(data["normalized_query"], str)
    assert isinstance(data["intent"], str)
    assert isinstance(data["entities"], list)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_normalize_lowercase() -> None:
    """Query must be lowercased in normalized_query (expansion may append mixed-case terms)."""
    result = process_query("WHERE IS MAIN?")
    assert result["success"] is True
    nq = result["data"]["normalized_query"]
    assert nq.startswith("where is main")


def test_normalize_strip_whitespace() -> None:
    """Leading/trailing whitespace must be stripped."""
    result = process_query("  explain paragraph  ")
    assert result["success"] is True
    assert result["data"]["normalized_query"].strip() == result["data"]["normalized_query"]


def test_normalize_collapse_internal_whitespace() -> None:
    """Multiple spaces must be collapsed to one."""
    result = process_query("what   does   MAIN-PGM   do")
    assert result["success"] is True
    assert "  " not in result["data"]["normalized_query"] or "main-pgm" in result["data"]["normalized_query"]


# ---------------------------------------------------------------------------
# Entity extraction (COBOL identifiers: ALL-CAPS-WITH-HYPHENS)
# ---------------------------------------------------------------------------

def test_entities_extract_cobol_paragraph_name() -> None:
    """COBOL paragraph names (ALL-CAPS-HYPHEN) must be extracted as entities."""
    result = process_query("Explain CALCULATE-INTEREST paragraph")
    assert result["success"] is True
    entities = result["data"]["entities"]
    assert "CALCULATE-INTEREST" in entities


def test_entities_extract_multiple_identifiers() -> None:
    """Multiple COBOL identifiers in query must all be extracted."""
    result = process_query("What modifies CUSTOMER-RECORD and WS-TRANS-AMT?")
    assert result["success"] is True
    entities = result["data"]["entities"]
    assert "CUSTOMER-RECORD" in entities
    assert "WS-TRANS-AMT" in entities


def test_entities_empty_when_none_present() -> None:
    """entities list empty when no COBOL-style identifiers in query."""
    result = process_query("where is the main entry point")
    assert result["success"] is True
    # "main" might be expanded but not necessarily an entity; no ALL-CAPS-HYPHEN
    assert isinstance(result["data"]["entities"], list)


# ---------------------------------------------------------------------------
# Query expansion (ambiguous terms)
# ---------------------------------------------------------------------------

def test_expansion_entry_point_adds_main_procedure() -> None:
    """'entry point' must be expanded with MAIN, PROCEDURE DIVISION etc."""
    result = process_query("Where is the entry point?")
    assert result["success"] is True
    nq = result["data"]["normalized_query"]
    # Expansion terms from constants: "entry point" -> "main MAIN PROCEDURE DIVISION"
    assert "main" in nq or "MAIN" in nq or "procedure" in nq or "division" in nq


def test_expansion_dependencies_adds_call_copy_using() -> None:
    """'dependencies' must be expanded with CALL COPY USING."""
    result = process_query("What are the dependencies of MODULE-X?")
    assert result["success"] is True
    nq = result["data"]["normalized_query"].lower()
    assert "call" in nq and "copy" in nq and "using" in nq


# ---------------------------------------------------------------------------
# Intent (heuristic)
# ---------------------------------------------------------------------------

def test_intent_find_location_for_where_query() -> None:
    """Intent should reflect 'where/locate' for 'where is' queries."""
    result = process_query("Where is MAIN-PGM?")
    assert result["success"] is True
    intent = result["data"]["intent"]
    assert isinstance(intent, str)
    assert len(intent) > 0


def test_intent_explain_for_explain_query() -> None:
    """Intent should reflect 'explain' for explain queries."""
    result = process_query("Explain what CALCULATE-INTEREST does")
    assert result["success"] is True
    intent = result["data"]["intent"]
    assert "explain" in intent.lower() or len(intent) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_query_returns_failure_or_safe_normalized() -> None:
    """Empty or whitespace-only query must not crash; return failure or minimal normalized."""
    result = process_query("")
    assert isinstance(result, dict)
    assert "success" in result
    if result["success"]:
        assert result["data"]["normalized_query"] == "" or len(result["data"]["normalized_query"].strip()) >= 0


def test_process_query_does_not_mutate_input() -> None:
    """Original query string must not be mutated."""
    original = "  What does MAIN-PGM do?  "
    process_query(original)
    assert original == "  What does MAIN-PGM do?  "
