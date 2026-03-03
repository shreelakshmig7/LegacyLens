"""
embedder.py
-----------
LegacyLens — RAG System for Legacy Enterprise Codebases — Voyage Code 2 batch embedder
----------------------------------------------------------------------------------------
Generates 1536-dimensional vector embeddings for COBOL code chunks using the
Voyage AI voyage-code-2 model. Embedding is performed in batches to minimise
API round-trips and respect rate limits.

Safety guarantees:
  - Chunks exceeding MAX_CHUNK_TOKENS are rejected before any API call is made.
  - Failed API calls are retried up to MAX_RETRIES times with exponential backoff
    (1s → 2s → 4s) before returning a structured error.
  - Input chunk dicts are never mutated — new dicts are returned with 'embedding' added.
  - SENSITIVE_LOG_FIELDS are filtered from any metadata that appears in log output.

Key functions:
    embed_chunks(chunks) -> dict
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
            logger.warning(
                "Embedding API timeout after %ds on attempt %d/%d — retrying",
                timeout_sec,
                attempt + 1,
                MAX_RETRIES,
            )
            # No sleep on timeout; retry immediately (total wall time still bounded per attempt)
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
        client = _get_voyage_client()
    except Exception as exc:
        logger.error("Failed to initialise Voyage AI client: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Voyage AI client error: {str(exc)}",
        }

    # ── Step 1: token pre-validation ─────────────────────────────────────────
    valid_chunks: List[Dict[str, Any]] = []
    skipped = 0

    for chunk in chunks:
        token_count = _estimate_tokens(chunk.get("text", ""))
        if token_count > MAX_CHUNK_TOKENS:
            logger.warning(
                "Skipping oversized chunk (~%d tokens > %d limit) in %s para=%s",
                token_count,
                MAX_CHUNK_TOKENS,
                chunk.get("file_path", "unknown"),
                chunk.get("paragraph_name", ""),
            )
            skipped += 1
        else:
            valid_chunks.append(chunk)

    if not valid_chunks:
        return {
            "success": True,
            "data": {"chunks": [], "embedded_count": 0, "skipped_count": skipped},
            "error": None,
        }

    # ── Step 2 & 3: batch and embed ───────────────────────────────────────────
    embedded_chunks: List[Dict[str, Any]] = []

    try:
        for batch_start in range(0, len(valid_chunks), INGESTION_BATCH_SIZE):
            batch = valid_chunks[batch_start : batch_start + INGESTION_BATCH_SIZE]
            texts = [c["text"] for c in batch]

            logger.info(
                "Embedding batch %d-%d of %d chunks",
                batch_start + 1,
                batch_start + len(batch),
                len(valid_chunks),
            )

            embeddings = _embed_batch_with_retry(client, texts)

            # Step 4: attach vectors to deep copies — never mutate input
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
        "Embedding complete: %d embedded, %d skipped",
        len(embedded_chunks),
        skipped,
    )

    return {
        "success": True,
        "data": {
            "chunks": embedded_chunks,
            "embedded_count": len(embedded_chunks),
            "skipped_count": skipped,
        },
        "error": None,
    }
