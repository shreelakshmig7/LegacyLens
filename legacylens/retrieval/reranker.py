"""
reranker.py
-----------
LegacyLens — RAG System for Legacy Enterprise Codebases — Re-rank results by relevance
---------------------------------------------------------------------------------------
Re-orders searcher results: boosts chunks whose paragraph_name matches query terms,
deprioritizes DATA chunks when the query is a "logic" query (explain, what does, etc.).
All weights and keywords from config.constants. No magic numbers.

Key functions:
    rerank(results, query) -> list

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import copy
import logging
from typing import Any, Dict, List

from legacylens.config.constants import (
    LOGIC_QUERY_KEYWORDS,
    RERANK_COMMENT_HEAVY_WEIGHT,
    RERANK_DATA_DEPRIORITIZE_WEIGHT,
    RERANK_DEAD_CODE_WEIGHT,
    RERANK_PARAGRAPH_BOOST_WEIGHT,
)

logger = logging.getLogger(__name__)


def _is_logic_query(query: str) -> bool:
    """True if query contains any LOGIC_QUERY_KEYWORDS."""
    if not query:
        return False
    q = query.lower()
    return any(kw in q for kw in LOGIC_QUERY_KEYWORDS)


def _tokenize_for_match(text: str) -> set:
    """Lowercase, replace hyphens with space, split into tokens (for paragraph/query match)."""
    if not text:
        return set()
    return set(text.lower().replace("-", " ").split())


def _paragraph_matches_query(paragraph_name: str, query_tokens: set) -> bool:
    """True if paragraph_name (case-insensitive) shares any token with query."""
    if not paragraph_name or not query_tokens:
        return False
    para_tokens = _tokenize_for_match(paragraph_name)
    return bool(para_tokens & query_tokens)


def rerank(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Re-rank results by relevance: boost paragraph_name matches, deprioritize DATA for logic queries.

    Does not mutate input; returns a new list sorted by adjusted score descending.

    Args:
        results: List of dicts with keys text, metadata, score (searcher output).
        query: Original or normalized query string.

    Returns:
        list: New list of same dicts with score updated to adjusted score, sorted descending.
    """
    if not results:
        return []

    query_tokens = _tokenize_for_match(query)
    is_logic = _is_logic_query(query)

    adjusted: List[Dict[str, Any]] = []
    for r in results:
        item = copy.deepcopy(r)
        base = float(item.get("score", 0.0))
        meta = item.get("metadata") or {}
        chunk_type = (meta.get("type") or "").strip()
        paragraph_name = (meta.get("paragraph_name") or "").strip()

        delta = 0.0
        if _paragraph_matches_query(paragraph_name, query_tokens):
            delta += RERANK_PARAGRAPH_BOOST_WEIGHT
        if is_logic and chunk_type == "DATA":
            delta += RERANK_DATA_DEPRIORITIZE_WEIGHT
        comment_weight = float(meta.get("comment_weight", 1.0))
        if comment_weight < 0.6:
            delta += RERANK_COMMENT_HEAVY_WEIGHT
        if bool(meta.get("dead_code_flag", False)):
            delta += RERANK_DEAD_CODE_WEIGHT

        item["score"] = round(base + delta, 4)
        adjusted.append(item)

    adjusted.sort(key=lambda x: x["score"], reverse=True)
    return adjusted
