"""
searcher.py
-----------
LegacyLens — RAG System for Legacy Enterprise Codebases — Top-k search with BM25 fallback
------------------------------------------------------------------------------------------
Runs similarity search via vector_store.query_similar (Voyage Code 2 + ChromaDB). When
results are fewer than BM25_FALLBACK_THRESHOLD, builds an in-memory BM25 index from the
full Chroma document set and returns top-k from keyword search. All constants from
config.constants. Does not modify vector_store structure.

Key functions:
    search(query) -> dict

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from legacylens.config.constants import (
    BM25_FALLBACK_THRESHOLD,
    CHROMA_GET_ALL_LIMIT,
    TOP_K,
)
from legacylens.ingestion.embedder import embed_chunks
from legacylens.retrieval.query_processor import detect_program, process_query
from legacylens.retrieval.vector_store import _get_collection, query_similar

logger = logging.getLogger(__name__)

# Lazy BM25 state: (corpus_tokens, documents, metadatas, bm25_index)
_bm25_state: Optional[Tuple[List[List[str]], List[str], List[Dict[str, Any]], Any]] = None


def _tokenize(text: str) -> List[str]:
    """Simple tokenize: split on non-alphanumeric, lowercase, drop empty."""
    if not text:
        return []
    tokens = re.sub(r"[^a-zA-Z0-9\-]", " ", text).split()
    return [t.lower() for t in tokens if t]


def _build_bm25_index() -> Tuple[List[str], List[Dict[str, Any]], Any]:
    """
    Build BM25 index from full Chroma document set. Returns (documents, metadatas, bm25_index).
    """
    global _bm25_state
    if _bm25_state is not None:
        return _bm25_state[1], _bm25_state[2], _bm25_state[3]

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        logger.error("rank_bm25 not installed: %s", exc)
        raise

    collection = _get_collection()
    raw = collection.get(
        include=["documents", "metadatas"],
        limit=CHROMA_GET_ALL_LIMIT,
    )
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    if len(metadatas) < len(documents):
        metadatas = metadatas + [{}] * (len(documents) - len(metadatas))
    elif len(documents) < len(metadatas):
        documents = documents + [""] * (len(metadatas) - len(documents))

    corpus_tokens = [_tokenize(d) for d in documents]

    if not corpus_tokens:
        logger.warning("BM25 index skipped: ChromaDB collection is empty (no documents ingested yet).")
        _bm25_state = ([], [], [], None)
        return [], [], None

    bm25 = BM25Okapi(corpus_tokens)
    _bm25_state = (corpus_tokens, documents, metadatas, bm25)
    logger.info("BM25 index built from %d Chroma documents", len(documents))
    return documents, metadatas, bm25


def _bm25_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Run BM25 search over in-memory index; return list of {text, metadata, score}."""
    try:
        documents, metadatas, bm25 = _build_bm25_index()
    except Exception as exc:
        logger.exception("BM25 index build failed: %s", exc)
        return []

    if not documents or bm25 is None:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in indices:
        if scores[i] <= 0:
            continue
        results.append({
            "text": documents[i],
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "score": float(scores[i]),
        })
    return results


def _filter_bm25_by_program(
    results: List[Dict[str, Any]],
    program: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Filter a list of BM25 result dicts to only those whose metadata file_name
    matches the requested program. If program is None all results are returned.

    Args:
        results: List of BM25 result dicts, each with a "metadata" sub-dict.
        program: Uppercase program stem to filter on, or None for no filtering.

    Returns:
        list[dict]: Filtered (or unfiltered) results.
    """
    if program is None:
        return results
    return [r for r in results if r.get("metadata", {}).get("file_name") == program]


def search(query: str, top_k: int = TOP_K) -> dict:
    """
    Run retrieval: normalize query, embed with Voyage, similarity search; if results < 3, BM25 fallback.

    Args:
        query: Natural language question (raw or pre-normalized).
        top_k: Max results to return (default from constants).

    Returns:
        dict: {
            "success": bool,
            "data": {
                "results": list[dict]  — each {text, metadata, score},
            } | None,
            "error": None | str,
        }
    """
    try:
        if not query or not isinstance(query, str):
            return {
                "success": True,
                "data": {"results": []},
                "error": None,
            }

        processed = process_query(query.strip())
        if not processed["success"]:
            return {
                "success": False,
                "data": None,
                "error": processed.get("error", "Query processing failed"),
            }

        normalized = processed["data"]["normalized_query"]
        if not normalized:
            return {"success": True, "data": {"results": []}, "error": None}

        # Detect if query targets a specific program; build ChromaDB filter if so.
        program_filter: Optional[str] = None
        detect_result = detect_program(query.strip())
        if detect_result["success"] and detect_result["data"]["program"]:
            program_filter = detect_result["data"]["program"]
            logger.info("Program-aware search: restricting to file_name=%s", program_filter)

        chroma_filters = {"file_name": program_filter} if program_filter else None

        # Embed query (single chunk)
        embed_result = embed_chunks([{"text": normalized}])
        if not embed_result["success"]:
            return {
                "success": False,
                "data": None,
                "error": embed_result.get("error", "Embedding failed"),
            }

        chunks = embed_result["data"]["chunks"]
        if not chunks or "embedding" not in chunks[0]:
            return {"success": False, "data": None, "error": "No query embedding returned"}

        query_vector = chunks[0]["embedding"]
        sim_result = query_similar(query_vector, top_k=top_k, filters=chroma_filters)

        if not sim_result["success"]:
            return {
                "success": False,
                "data": None,
                "error": sim_result.get("error", "Similarity search failed"),
            }

        results = sim_result["data"]["results"]

        if len(results) < BM25_FALLBACK_THRESHOLD:
            logger.info(
                "Similarity returned %d results (< %d), triggering BM25 fallback",
                len(results),
                BM25_FALLBACK_THRESHOLD,
            )
            bm25_results = _bm25_search(normalized, top_k=top_k)
            results = _filter_bm25_by_program(bm25_results, program_filter)

        return {
            "success": True,
            "data": {"results": results},
            "error": None,
        }

    except Exception as exc:
        logger.exception("Unexpected error in search: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }
