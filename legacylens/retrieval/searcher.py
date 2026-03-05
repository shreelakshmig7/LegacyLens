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
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from legacylens.config.constants import (
    BM25_FALLBACK_THRESHOLD,
    CHROMA_GET_ALL_LIMIT,
    EMBEDDING_MODEL,
    MIN_RELEVANCE_THRESHOLD,
    QUERY_EMBED_TIMEOUT_SECONDS,
    TOP_K,
    TOP_K_COMPOUND,
)
from legacylens.retrieval.query_processor import detect_program, process_query
from legacylens.retrieval.vector_store import _get_collection, query_similar

logger = logging.getLogger(__name__)

_PARAGRAPH_QUERY_SIGNALS = frozenset({
    "paragraph", "section", "procedure", "what does", "explain",
})


def _embed_query(text: str) -> List[float]:
    """
    Embed a single query string using Voyage AI directly.

    Bypasses the batch ingestion pipeline (ThreadPoolExecutor, retry sleeps,
    60s timeout) in favour of a direct API call with a tight query-time timeout.
    On any failure the caller falls back to BM25 immediately rather than waiting
    for retries — fast failure is preferable at query time.

    Args:
        text: The normalised query string to embed.

    Returns:
        List[float]: A 1536-dimensional embedding vector.

    Raises:
        Exception: Any Voyage AI client or network error; caller handles fallback.
    """
    import voyageai

    client = voyageai.Client(
        api_key=os.getenv("VOYAGE_API_KEY"),
        timeout=QUERY_EMBED_TIMEOUT_SECONDS,
    )
    response = client.embed([text], model=EMBEDDING_MODEL)
    return response.embeddings[0]


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


def _max_result_score(results: List[Dict[str, Any]]) -> float:
    """
    Return the maximum score from a retrieval result list.

    Args:
        results: Retrieval results where each item may include a numeric "score".

    Returns:
        float: Maximum score found, or 0.0 for empty/invalid lists.
    """
    if not results:
        return 0.0
    numeric_scores = [float(r.get("score", 0.0)) for r in results]
    return max(numeric_scores) if numeric_scores else 0.0


def _should_trigger_bm25(results: List[Dict[str, Any]]) -> bool:
    """
    Decide whether BM25 fallback must run for current vector results.

    Args:
        results: Vector similarity result list.

    Returns:
        bool: True when fallback should run.
    """
    if not results:
        return True
    max_score = _max_result_score(results)
    if max_score < MIN_RELEVANCE_THRESHOLD:
        return True
    return len(results) < BM25_FALLBACK_THRESHOLD


def _paragraph_metadata_lookup(
    entities: List[str],
    query_lower: str,
) -> List[Dict[str, Any]]:
    """
    When the query explicitly references a paragraph/section by name, fetch
    matching chunks directly from ChromaDB metadata so they are never missed
    by vector similarity or BM25.

    Args:
        entities: COBOL identifiers extracted from the query.
        query_lower: Lowercase query text for signal detection.

    Returns:
        list[dict]: Matching chunks (may be empty).
    """
    if not entities:
        return []
    if not any(sig in query_lower for sig in _PARAGRAPH_QUERY_SIGNALS):
        return []

    collection = _get_collection()
    hits: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for entity in entities:
        try:
            raw = collection.get(
                where={"paragraph_name": entity},
                include=["documents", "metadatas"],
                limit=5,
            )
            docs = raw.get("documents") or []
            metas = raw.get("metadatas") or []
            ids = raw.get("ids") or []
            for i, doc in enumerate(docs):
                if ids[i] in seen_ids:
                    continue
                seen_ids.add(ids[i])
                hits.append({
                    "text": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": 1.0,
                })
        except Exception as exc:
            logger.warning("Paragraph metadata lookup failed for %r: %s", entity, exc)

    if hits:
        logger.info(
            "Paragraph metadata lookup found %d chunk(s) for entities %s",
            len(hits), entities,
        )
    return hits


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

        entity_count = processed["data"].get("entity_count", 0)
        if entity_count > 1 and top_k < TOP_K_COMPOUND:
            logger.info(
                "Compound query detected (%d entities) — expanding top_k from %d to %d",
                entity_count, top_k, TOP_K_COMPOUND,
            )
            top_k = TOP_K_COMPOUND

        # Detect if query targets a specific program; build ChromaDB filter if so.
        program_filter: Optional[str] = None
        detect_result = detect_program(query.strip())
        if detect_result["success"] and detect_result["data"]["program"]:
            program_filter = detect_result["data"]["program"]
            logger.info("Program-aware search: restricting to file_name=%s", program_filter)

        target_type = (processed.get("data") or {}).get("target_type") or ""
        filter_clauses: List[Dict[str, Any]] = []
        if program_filter:
            filter_clauses.append({"file_name": program_filter})
        if target_type in ("PROCEDURE", "DATA"):
            filter_clauses.append({"type": target_type})

        chroma_filters: Optional[Dict[str, Any]] = None
        if len(filter_clauses) == 1:
            chroma_filters = filter_clauses[0]
        elif len(filter_clauses) > 1:
            chroma_filters = {"$and": filter_clauses}

        results: List[Dict[str, Any]] = []

        try:
            query_vector = _embed_query(normalized)
        except Exception as exc:
            logger.warning(
                "Query embedding failed, falling back to BM25-only retrieval: %s", exc
            )
            bm25_results = _bm25_search(normalized, top_k=top_k)
            results = _filter_bm25_by_program(bm25_results, program_filter)
            return {"success": True, "data": {"results": results}, "error": None}

        retrieval_mode = os.getenv("LEGACYLENS_RETRIEVAL_MODE", "hybrid").strip().lower()

        if retrieval_mode == "bm25":
            logger.info("Retrieval mode=BM25 (eval baseline)")
            bm25_results = _bm25_search(normalized, top_k=top_k)
            results = _filter_bm25_by_program(bm25_results, program_filter)

        else:
            sim_result = query_similar(query_vector, top_k=top_k, filters=chroma_filters)

            if not sim_result["success"]:
                return {
                    "success": False,
                    "data": None,
                    "error": sim_result.get("error", "Similarity search failed"),
                }

            results = sim_result["data"]["results"]

            if not results and chroma_filters and "type" in chroma_filters:
                relaxed = dict(chroma_filters)
                relaxed.pop("type", None)
                relaxed_result = query_similar(query_vector, top_k=top_k, filters=relaxed or None)
                if relaxed_result.get("success"):
                    results = (relaxed_result.get("data") or {}).get("results") or []

            if retrieval_mode != "vector_only" and _should_trigger_bm25(results):
                max_score = _max_result_score(results)
                logger.info(
                    "Similarity returned %d results (max score=%.3f, threshold=%.2f), triggering BM25 fallback",
                    len(results),
                    max_score,
                    MIN_RELEVANCE_THRESHOLD,
                )
                bm25_results = _bm25_search(normalized, top_k=top_k)
                results = _filter_bm25_by_program(bm25_results, program_filter)

            if program_filter and not results:
                logger.info(
                    "Program filter '%s' returned 0 results — falling back to global search",
                    program_filter,
                )
                global_sim = query_similar(query_vector, top_k=top_k, filters=None)
                if global_sim["success"]:
                    results = global_sim["data"]["results"]
                if _should_trigger_bm25(results):
                    results = _bm25_search(normalized, top_k=top_k)

        # Paragraph-name metadata lookup: when the query explicitly names a
        # paragraph, fetch matching chunks directly so they are never lost to
        # low vector/BM25 scores. Runs after ALL retrieval paths converge.
        entities = processed["data"].get("entities") or []
        para_hits = _paragraph_metadata_lookup(entities, normalized)
        if para_hits:
            existing_texts = {r.get("text", "")[:200] for r in results}
            for ph in para_hits:
                if ph.get("text", "")[:200] not in existing_texts:
                    results.insert(0, ph)

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
