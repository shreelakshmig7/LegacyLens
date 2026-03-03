"""
query_processor.py
------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Query normalization and expansion
--------------------------------------------------------------------------------------------
Normalizes natural language queries, extracts intent and COBOL entity names (paragraph/section
identifiers), and expands ambiguous terms using QUERY_EXPANSION_TERMS. All magic values
come from config.constants.

Key functions:
    process_query(raw_query) -> dict

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import re
from typing import List

from legacylens.config.constants import (
    LOGIC_QUERY_KEYWORDS,
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
                },
                "error": None,
            }

        normalized = _normalize(raw_query)
        entities = _extract_entities(raw_query)
        expanded = _expand_ambiguous_terms(normalized)
        intent = _infer_intent(normalized)

        return {
            "success": True,
            "data": {
                "normalized_query": expanded,
                "intent": intent,
                "entities": entities,
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
