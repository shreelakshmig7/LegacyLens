"""
embedder.py
-----------
LegacyLens — RAG System for Legacy Enterprise Codebases — Voyage Code 2 batch embedder
----------------------------------------------------------------------------------------
Generates 1536-dimensional vector embeddings for COBOL code chunks using the
Voyage AI voyage-code-2 model. Embedding is performed in batches to minimise
API round-trips and respect rate limits.

Safety guarantees:
  - Chunks exceeding MAX_CHUNK_TOKENS are sub-split into smaller pieces before embedding
    so no content is ever silently dropped (zero-drop guarantee).
  - Failed API calls and timeouts are retried up to MAX_RETRIES times with exponential
    backoff (1s → 2s → 4s) before returning a structured error.
  - Input chunk dicts are never mutated — new dicts are returned with 'embedding' added.
  - SENSITIVE_LOG_FIELDS are filtered from any metadata that appears in log output.

Key functions:
    embed_chunks(chunks) -> dict
    _split_chunk_to_subchunks(chunk, max_tokens) -> list[dict]
    _get_voyage_client() -> voyageai.Client  (injectable for testing)

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import concurrent.futures
import copy
import logging
import os
import time
from typing import Any, Dict, List

from legacylens.config.constants import (
    EMBEDDING_BATCH_WORKERS,
    EMBEDDING_MODEL,
    INGESTION_BATCH_SIZE,
    MAX_CHUNK_TOKENS,
    MAX_RETRIES,
    SENSITIVE_LOG_FIELDS,
    VOYAGE_API_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client factory (isolated so tests can patch it cleanly)
# ---------------------------------------------------------------------------

def _get_voyage_client():
    """
    Create and return a Voyage AI client using the VOYAGE_API_KEY environment variable.

    Returns:
        voyageai.Client: Authenticated Voyage AI client.

    Raises:
        ImportError: If the voyageai package is not installed.
        ValueError: If VOYAGE_API_KEY is not set in the environment.
    """
    import voyageai  # type: ignore
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY environment variable is not set")
    return voyageai.Client(api_key=api_key, timeout=VOYAGE_API_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """
    Estimate token count using a word-count heuristic (words / 0.75).

    Args:
        text: Text to estimate.

    Returns:
        int: Estimated token count, minimum 1.
    """
    return max(1, int(len(text.split()) / 0.75))


# ---------------------------------------------------------------------------
# Metadata sanitisation for safe logging
# ---------------------------------------------------------------------------

def _split_chunk_to_subchunks(
    chunk: Dict[str, Any],
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Split an oversized chunk dict into sub-chunks that each fit within max_tokens.

    Uses a word-window approach with the same token formula as embed_chunks pre-validation
    (int floor of words / 0.75). All metadata fields from the original chunk are deep-copied
    to every sub-chunk. This is the final zero-drop safety net — called only when a chunk
    survives the chunker but is still detected as oversized at embedding time.

    Args:
        chunk:      Original chunk dict with at minimum a "text" key.
        max_tokens: Token ceiling per sub-chunk (same units as _estimate_tokens).

    Returns:
        list[dict]: One or more sub-chunk dicts, each estimated within max_tokens.
                    Returns a single-element list if the text is already within limit.
    """
    text = chunk.get("text", "")
    words = text.split()

    if not words:
        return [chunk]

    # Inverse of int(words / 0.75): max word count per part that keeps tokens under limit.
    max_words_per_part = max(1, int(max_tokens * 0.75))

    if len(words) <= max_words_per_part:
        return [chunk]

    sub_chunks: List[Dict[str, Any]] = []
    for start in range(0, len(words), max_words_per_part):
        part_words = words[start : start + max_words_per_part]
        new_chunk = copy.deepcopy(chunk)
        new_chunk["text"] = " ".join(part_words)
        sub_chunks.append(new_chunk)

    return sub_chunks


def _safe_log_chunk(chunk: Dict[str, Any]) -> dict:
    """
    Return a copy of a chunk dict with SENSITIVE_LOG_FIELDS removed.

    Args:
        chunk: A chunk metadata dict.

    Returns:
        dict: Safe copy with sensitive fields removed.
    """
    return {k: v for k, v in chunk.items() if k not in SENSITIVE_LOG_FIELDS}


# ---------------------------------------------------------------------------
# Single-batch embedding with exponential backoff
# ---------------------------------------------------------------------------

def _embed_batch_with_retry(client, texts: List[str]) -> List[List[float]]:
    """
    Call the Voyage AI embed API for a batch of texts, retrying on failure.

    Implements exponential backoff: waits 2^attempt seconds between retries
    (1s after attempt 0, 2s after attempt 1, 4s after attempt 2).
    Each attempt is capped at VOYAGE_API_TIMEOUT_SECONDS via a thread timeout
    so a single API call never blocks the eval indefinitely.

    Args:
        client: Voyage AI client instance.
        texts:  List of text strings to embed (already validated for token count).

    Returns:
        list[list[float]]: Embedding vectors, one per input text.

    Raises:
        Exception: Re-raises the last exception if all MAX_RETRIES attempts fail,
            or concurrent.futures.TimeoutError if the API does not respond in time.
    """
    last_exc: Exception = Exception("Unknown embedding error")
    timeout_sec = VOYAGE_API_TIMEOUT_SECONDS

    for attempt in range(MAX_RETRIES):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    lambda: client.embed(texts, model=EMBEDDING_MODEL),
                )
                response = future.result(timeout=timeout_sec)
            return response.embeddings
        except concurrent.futures.TimeoutError as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Embedding API timeout after %ds on attempt %d/%d — retrying in %ds",
                timeout_sec,
                attempt + 1,
                MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Embedding API error on attempt %d/%d: %s — retrying in %ds",
                attempt + 1,
                MAX_RETRIES,
                str(exc),
                wait,
            )
            time.sleep(wait)

    raise last_exc


def _embed_batch_task(batch_index: int, texts: List[str]) -> Dict[str, Any]:
    """
    Embed one batch in a worker-safe wrapper.

    Args:
        batch_index: Stable batch order index.
        texts: Batch texts.

    Returns:
        dict: {"batch_index": int, "embeddings": list[list[float]]}
    """
    client = _get_voyage_client()
    embeddings = _embed_batch_with_retry(client, texts)
    return {"batch_index": batch_index, "embeddings": embeddings}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[Dict[str, Any]]) -> dict:
    """
    Generate embeddings for a list of chunk dicts and return enriched copies.

    Processing steps:
      1. Filter out chunks that exceed MAX_CHUNK_TOKENS (logged as warnings).
      2. Split remaining chunks into batches of INGESTION_BATCH_SIZE.
      3. Call Voyage Code 2 for each batch with exponential backoff.
      4. Attach the returned vector to a deep copy of each chunk dict.

    Args:
        chunks: List of chunk dicts as produced by the chunker + reference scraper.
                Each dict must have at minimum a "text" key.

    Returns:
        dict: {
            "success": bool,
            "data": {
                "chunks": list[dict]  — copies of input chunks with "embedding" added,
                "embedded_count": int,
                "skipped_count": int  — chunks dropped for exceeding token limit,
            },
            "error": None | str,
        }
    """
    if not chunks:
        return {
            "success": True,
            "data": {"chunks": [], "embedded_count": 0, "skipped_count": 0},
            "error": None,
        }

    try:
        if EMBEDDING_BATCH_WORKERS <= 1:
            _ = _get_voyage_client()
    except Exception as exc:
        logger.error("Failed to initialise Voyage AI client: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Voyage AI client error: {str(exc)}",
        }

    # ── Step 1: token pre-validation + zero-drop sub-splitting ───────────────
    # Oversized chunks are sub-split into valid pieces instead of discarded.
    # This guarantees 100% content coverage regardless of upstream chunker edge cases.
    valid_chunks: List[Dict[str, Any]] = []
    subsplit_count = 0

    for chunk in chunks:
        token_count = _estimate_tokens(chunk.get("text", ""))
        if token_count > MAX_CHUNK_TOKENS:
            logger.warning(
                "Sub-splitting oversized chunk (~%d tokens > %d limit) in %s para=%s",
                token_count,
                MAX_CHUNK_TOKENS,
                chunk.get("file_path", "unknown"),
                chunk.get("paragraph_name", ""),
            )
            sub_chunks = _split_chunk_to_subchunks(chunk, MAX_CHUNK_TOKENS)
            valid_chunks.extend(sub_chunks)
            subsplit_count += 1
        else:
            valid_chunks.append(chunk)

    if not valid_chunks:
        return {
            "success": True,
            "data": {"chunks": [], "embedded_count": 0, "skipped_count": 0, "subsplit_count": 0},
            "error": None,
        }

    # ── Step 2 & 3: batch and embed ───────────────────────────────────────────
    embedded_chunks: List[Dict[str, Any]] = []

    try:
        batches: List[List[Dict[str, Any]]] = []
        for batch_start in range(0, len(valid_chunks), INGESTION_BATCH_SIZE):
            batch = valid_chunks[batch_start : batch_start + INGESTION_BATCH_SIZE]
            batches.append(batch)

        if EMBEDDING_BATCH_WORKERS <= 1:
            client = _get_voyage_client()
            for batch_index, batch in enumerate(batches):
                batch_start = batch_index * INGESTION_BATCH_SIZE
                texts = [c["text"] for c in batch]
                logger.info(
                    "Embedding batch %d-%d of %d chunks",
                    batch_start + 1,
                    batch_start + len(batch),
                    len(valid_chunks),
                )
                embeddings = _embed_batch_with_retry(client, texts)
                for chunk, vector in zip(batch, embeddings):
                    new_chunk = copy.deepcopy(chunk)
                    new_chunk["embedding"] = vector
                    embedded_chunks.append(new_chunk)
        else:
            logger.info("Embedding with %d parallel workers", EMBEDDING_BATCH_WORKERS)
            futures: List[Any] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=EMBEDDING_BATCH_WORKERS) as executor:
                for batch_index, batch in enumerate(batches):
                    batch_start = batch_index * INGESTION_BATCH_SIZE
                    logger.info(
                        "Queueing batch %d-%d of %d chunks",
                        batch_start + 1,
                        batch_start + len(batch),
                        len(valid_chunks),
                    )
                    futures.append(
                        executor.submit(
                            _embed_batch_task,
                            batch_index,
                            [c["text"] for c in batch],
                        )
                    )

                ordered: Dict[int, List[List[float]]] = {}
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    ordered[result["batch_index"]] = result["embeddings"]

            for batch_index, batch in enumerate(batches):
                embeddings = ordered.get(batch_index, [])
                for chunk, vector in zip(batch, embeddings):
                    new_chunk = copy.deepcopy(chunk)
                    new_chunk["embedding"] = vector
                    embedded_chunks.append(new_chunk)

    except Exception as exc:
        logger.error("Embedding failed after %d retries: %s", MAX_RETRIES, exc)
        return {
            "success": False,
            "data": None,
            "error": f"Embedding failed after {MAX_RETRIES} retries: {str(exc)}",
        }

    logger.info(
        "Embedding complete: %d embedded, %d chunks sub-split (zero dropped)",
        len(embedded_chunks),
        subsplit_count,
    )

    return {
        "success": True,
        "data": {
            "chunks": embedded_chunks,
            "embedded_count": len(embedded_chunks),
            "skipped_count": 0,        # maintained for backward compat; sub-splitting replaced skipping
            "subsplit_count": subsplit_count,
        },
        "error": None,
    }
