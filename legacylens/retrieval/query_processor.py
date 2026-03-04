"""
query_processor.py
------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Query normalization and expansion
--------------------------------------------------------------------------------------------
Normalizes natural language queries, extracts intent and COBOL entity names (paragraph/section
identifiers), and expands ambiguous terms using QUERY_EXPANSION_TERMS. Also detects whether
the user's query mentions a specific program from PROGRAM_CATEGORIES so the retrieval layer
can apply a file_name metadata filter for program-scoped searches.

Key functions:
    process_query(raw_query)  -> dict
    detect_program(query)     -> dict

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import re
from typing import List

from legacylens.config.constants import (
    LOGIC_QUERY_KEYWORDS,
    PROGRAM_CATEGORIES,
    QUERY_EXPANSION_TERMS,
)

logger = logging.getLogger(__name__)

# COBOL identifier: one or more ALL-CAPS words with optional hyphens (e.g. MAIN-PGM, WS-TRANS-AMT)
_COBOL_IDENTIFIER_RE = re.compile(r"\b([A-Z][A-Z0-9\-]+(?:\s+[A-Z][A-Z0-9\-]+)*)\b")


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    if not text or not isinstance(text, str):
        return ""
    t = text.strip().lower()
    t = " ".join(t.split())
    return t


def _extract_entities(query: str) -> List[str]:
    """Extract COBOL-style identifiers (ALL-CAPS-WITH-HYPHENS) from query."""
    seen: set = set()
    entities: List[str] = []
    for m in _COBOL_IDENTIFIER_RE.finditer(query):
        name = m.group(1).strip()
        if name not in seen:
            seen.add(name)
            entities.append(name)
    return entities


def _expand_ambiguous_terms(normalized: str) -> str:
    """Append expansion phrases for known ambiguous terms (from QUERY_EXPANSION_TERMS)."""
    parts = [normalized]
    for term, expansion in QUERY_EXPANSION_TERMS.items():
        if term in normalized:
            parts.append(expansion)
    return " ".join(parts)


def _infer_intent(normalized: str) -> str:
    """Heuristic intent: find_location, explain, list_dependencies, or general."""
    if not normalized:
        return "general"
    n = normalized.lower()
    if any(kw in n for kw in ["where is", "where are", "locate", "find the", "find all"]):
        return "find_location"
    if any(kw in n for kw in ["explain", "what does", "how does", "what is"]):
        return "explain"
    if any(kw in n for kw in ["dependencies", "dependency", "calls", "copy", "using"]):
        return "list_dependencies"
    if any(kw in n for kw in ["entry point", "main entry"]):
        return "find_location"
    return "general"


def _infer_target_type(normalized: str) -> str:
    """
    Infer chunk type filter target based on query semantics.

    Args:
        normalized: Normalized lowercase query.

    Returns:
        str: "PROCEDURE", "DATA", or "".
    """
    if not normalized:
        return ""
    data_signals = (
        "data division",
        "working-storage",
        "linkage section",
        "file section",
        "fd ",
        "field",
        "record layout",
        "variable",
        "definition",
    )
    procedure_signals = (
        "what does",
        "explain",
        "logic",
        "flow",
        "how does",
        "paragraph",
        "section",
        "entry point",
        "perform",
    )
    if any(s in normalized for s in data_signals):
        return "DATA"
    if any(s in normalized for s in procedure_signals):
        return "PROCEDURE"
    return ""


def process_query(raw_query: str) -> dict:
    """
    Normalize query, extract intent and entities, expand ambiguous terms.

    Args:
        raw_query: User's natural language question (may be empty or whitespace).

    Returns:
        dict: {
            "success": bool,
            "data": {
                "normalized_query": str,
                "intent": str,
                "entities": list[str],
                "target_type": str,
            },
            "error": None | str,
        }
    """
    try:
        if not raw_query or not isinstance(raw_query, str):
            return {
                "success": True,
                "data": {
                    "normalized_query": "",
                    "intent": "general",
                    "entities": [],
                    "target_type": "",
                },
                "error": None,
            }

        normalized = _normalize(raw_query)
        entities = _extract_entities(raw_query)
        expanded = _expand_ambiguous_terms(normalized)
        intent = _infer_intent(normalized)
        target_type = _infer_target_type(normalized)

        return {
            "success": True,
            "data": {
                "normalized_query": expanded,
                "intent": intent,
                "entities": entities,
                "target_type": target_type,
            },
            "error": None,
        }

    except Exception as exc:
        logger.exception("Unexpected error in process_query: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }


def detect_program(query: str) -> dict:
    """
    Detect whether a known program name from PROGRAM_CATEGORIES appears in the query.

    Uses case-insensitive substring matching — no LLM call, zero added latency.
    When two program names appear in the same query the first match (by list order
    in PROGRAM_CATEGORIES) is returned so results are deterministic.

    Args:
        query: Raw user query (may be None, empty, or any string).

    Returns:
        dict: {
            "success": bool,
            "data": {
                "program": str | None  — matched uppercase program name, or None,
            },
            "error": None | str,
        }
    """
    try:
        if not query or not isinstance(query, str):
            return {"success": True, "data": {"program": None}, "error": None}

        q_lower = query.strip().lower()
        for program in PROGRAM_CATEGORIES:
            if program.lower() in q_lower:
                logger.debug("Program-aware filter activated for query — program: %s", program)
                return {"success": True, "data": {"program": program}, "error": None}

        return {"success": True, "data": {"program": None}, "error": None}

    except Exception as exc:
        logger.exception("Unexpected error in detect_program: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }
