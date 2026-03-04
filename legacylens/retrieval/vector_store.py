"""
vector_store.py
---------------
LegacyLens — RAG System for Legacy Enterprise Codebases — ChromaDB vector store wrapper
----------------------------------------------------------------------------------------
Provides the interface between the ingestion pipeline and ChromaDB. Handles chunk
insertion with zero-tolerance verification, metadata sanitisation before storage and
logging, query input validation, and similarity search.

Security controls:
  - sanitize_metadata()       strips special chars, removes SENSITIVE_LOG_FIELDS,
                              and flags file paths outside the repo root
  - sanitize_query_filters()  whitelists allowed filter fields and rejects operator
                              injection before any filter reaches ChromaDB

Zero-tolerance verification:
  After insertion, the actual ChromaDB document count is compared to the expected count.
  Any discrepancy halts the pipeline with a structured error — a partial index is never
  silently accepted.

Key functions:
    insert_chunks(chunks)                    -> dict
    query_similar(vector, top_k, filters)    -> dict
    sanitize_metadata(metadata, repo_root)   -> dict
    sanitize_query_filters(filters)          -> dict
    _get_collection()                        -> chromadb.Collection

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import hashlib
import logging
import pathlib
import re
import uuid
from typing import Any, Dict, List, Optional

from legacylens.config.constants import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_DIMENSIONS,
    MANDATORY_METADATA_FIELDS,
    SENSITIVE_LOG_FIELDS,
    TOP_K,
)

logger = logging.getLogger(__name__)

# Whitelisted metadata filter fields — only these may appear in ChromaDB where-filters
_ALLOWED_FILTER_FIELDS = frozenset(
    {"file_path", "file_name", "type", "parent_section", "paragraph_name"}
)

# Characters that must be stripped from string metadata values before storage
_UNSAFE_CHARS_RE = re.compile(r"[\x00-\x1f<>&\"'\\]")

_COLLECTION_NAME = "legacylens_cobol"


def _is_valid_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate mandatory metadata fields and required non-empty values.

    Args:
        metadata: Sanitized metadata candidate.

    Returns:
        dict: {"valid": bool, "reason": str}
    """
    for field in MANDATORY_METADATA_FIELDS:
        if field not in metadata:
            return {"valid": False, "reason": f"Missing mandatory metadata field: {field}"}
    non_empty_fields = ("file_path", "file_name", "line_range", "type", "parent_section")
    for field in non_empty_fields:
        if str(metadata.get(field, "")).strip() == "":
            return {"valid": False, "reason": f"Mandatory metadata field is empty: {field}"}
    return {"valid": True, "reason": ""}


# ---------------------------------------------------------------------------
# ChromaDB client factory
# ---------------------------------------------------------------------------

def _get_collection():
    """
    Return (or create) the persistent ChromaDB collection used by LegacyLens.

    Uses CHROMA_PERSIST_DIR for on-disk persistence so data survives restarts.

    Returns:
        chromadb.Collection: The LegacyLens chunk collection.
    """
    import chromadb  # type: ignore
    from chromadb.config import Settings

    # Path from config only — never hardcoded (local: ./chroma_db, Railway: /data/chroma_db via env)
    # Disable anonymized telemetry to avoid "capture() takes 1 positional argument but 3 were given" (Chroma/PostHog bug)
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Metadata sanitisation
# ---------------------------------------------------------------------------

def sanitize_metadata(
    metadata: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a sanitised copy of a chunk metadata dict safe for storage and logging.

    Operations performed:
      1. Remove any key in SENSITIVE_LOG_FIELDS.
      2. Strip control characters and dangerous chars from all string values.
      3. If repo_root is provided, flag file_path values that resolve outside it.

    Args:
        metadata:  The raw metadata dict to sanitise.
        repo_root: Optional path to the repo root directory. When supplied,
                   file_path values that resolve outside this directory are
                   flagged with _path_unsafe=True and the path is blanked.

    Returns:
        dict: Sanitised metadata dict — never the same object as input.
    """
    clean: Dict[str, Any] = {}

    for key, value in metadata.items():
        # Drop sensitive fields entirely
        if key in SENSITIVE_LOG_FIELDS:
            continue

        if isinstance(value, str):
            clean[key] = _UNSAFE_CHARS_RE.sub("", value)
        elif isinstance(value, list):
            clean[key] = [
                _UNSAFE_CHARS_RE.sub("", v) if isinstance(v, str) else v
                for v in value
            ]
        else:
            clean[key] = value

    # Path containment check
    if repo_root and "file_path" in clean:
        try:
            file_resolved = pathlib.Path(clean["file_path"]).resolve()
            repo_resolved = pathlib.Path(repo_root).resolve()
            file_resolved.relative_to(repo_resolved)
        except ValueError:
            logger.warning(
                "file_path resolves outside repo root — flagging as unsafe: %s",
                clean["file_path"],
            )
            clean["_path_unsafe"] = True
            clean["file_path"] = ""

    return clean


# ---------------------------------------------------------------------------
# Query filter sanitisation
# ---------------------------------------------------------------------------

def sanitize_query_filters(filters: Dict[str, Any]) -> dict:
    """
    Validate a ChromaDB where-filter dict before it touches the database.

    Rejects filters that contain non-whitelisted field names or keys that begin
    with $ (ChromaDB operator injection).

    Args:
        filters: A dict intended for use as a ChromaDB where= filter.

    Returns:
        dict: {
            "success": bool,
            "data": {"filters": dict} | None,
            "error": None | str,
        }
    """
    for key in filters:
        if key.startswith("$"):
            return {
                "success": False,
                "data": None,
                "error": f"Operator injection rejected in query filter: {key!r}",
            }
        if key not in _ALLOWED_FILTER_FIELDS:
            return {
                "success": False,
                "data": None,
                "error": (
                    f"Disallowed filter field: {key!r}. "
                    f"Allowed: {sorted(_ALLOWED_FILTER_FIELDS)}"
                ),
            }

    return {"success": True, "data": {"filters": filters}, "error": None}


# ---------------------------------------------------------------------------
# Chunk insertion
# ---------------------------------------------------------------------------

def insert_chunks(
    chunks: List[Dict[str, Any]],
    repo_root: Optional[str] = None,
) -> dict:
    """
    Insert a list of embedded chunk dicts into ChromaDB with zero-tolerance verification.

    Each chunk must have an "embedding" key (list of floats) plus the full metadata
    schema. After insertion, the actual document count in the collection is compared to
    the expected count. Any discrepancy returns a failure — a partial index is never
    silently accepted.

    Args:
        chunks:    List of embedded chunk dicts (output of embedder.embed_chunks).
        repo_root: Optional repo root path for file_path containment validation.

    Returns:
        dict: {
            "success": bool,
            "data": {
                "inserted_count": int,
                "verified": bool,
            } | None,
            "error": None | str,
        }
    """
    if not chunks:
        return {
            "success": True,
            "data": {"inserted_count": 0, "verified": True},
            "error": None,
        }

    try:
        collection = _get_collection()

        # Record count before insertion for delta verification
        count_before = collection.count()

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        rejected_count = 0

        for chunk in chunks:
            raw_meta = {
                "file_path": chunk.get("file_path", ""),
                "file_name": chunk.get("file_name", "") or pathlib.Path(chunk.get("file_path", "")).stem.upper(),
                "line_range": str(chunk.get("line_range", [0, 0])),
                "type": chunk.get("type", ""),
                "parent_section": chunk.get("parent_section", ""),
                "paragraph_name": chunk.get("paragraph_name", ""),
                "dependencies": ",".join(chunk.get("dependencies", [])),
                "comment_weight": float(chunk.get("comment_weight", 1.0)),
                "dead_code_flag": bool(chunk.get("dead_code_flag", False)),
            }
            clean_meta = sanitize_metadata(raw_meta, repo_root=repo_root)
            meta_check = _is_valid_metadata(clean_meta)
            if not meta_check["valid"]:
                rejected_count += 1
                logger.warning("Rejecting chunk due to invalid metadata: %s", meta_check["reason"])
                continue

            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            metadatas.append(clean_meta)

        if not ids:
            return {
                "success": True,
                "data": {"inserted_count": 0, "rejected_count": rejected_count, "verified": True},
                "error": None,
            }

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # Zero-tolerance verification
        count_after = collection.count()
        expected = count_before + len(ids)

        if count_after != expected:
            msg = (
                f"Insertion verification failed: expected {expected} documents "
                f"in collection, found {count_after}. "
                f"({len(ids)} chunks were submitted, delta={count_after - count_before})"
            )
            logger.error(msg)
            return {"success": False, "data": None, "error": msg}

        logger.info(
            "Inserted and verified %d chunks into ChromaDB (total: %d)",
            len(ids),
            count_after,
        )

        return {
            "success": True,
            "data": {"inserted_count": len(ids), "rejected_count": rejected_count, "verified": True},
            "error": None,
        }

    except Exception as exc:
        logger.exception("Unexpected error during chunk insertion: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def query_similar(
    query_vector: List[float],
    top_k: int = TOP_K,
    filters: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Perform a cosine similarity search against the ChromaDB collection.

    If filters are provided, they are sanitised via sanitize_query_filters before
    being passed to ChromaDB. Invalid filters return a failure immediately.

    Args:
        query_vector: The embedded query as a list of floats (must be EMBEDDING_DIMENSIONS).
        top_k:        Number of most similar chunks to return.
        filters:      Optional ChromaDB where-filter dict (must use whitelisted fields only).

    Returns:
        dict: {
            "success": bool,
            "data": {
                "results": list[dict]  — each with text, metadata, score keys,
            } | None,
            "error": None | str,
        }
    """
    try:
        # Validate filters before touching the database
        if filters:
            filter_check = sanitize_query_filters(filters)
            if not filter_check["success"]:
                return {
                    "success": False,
                    "data": None,
                    "error": filter_check["error"],
                }

        collection = _get_collection()

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_kwargs["where"] = filters

        raw = collection.query(**query_kwargs)

        results = []
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert cosine distance to similarity score (1 - distance for cosine)
            score = round(1.0 - float(dist), 4)
            results.append({"text": doc, "metadata": meta, "score": score})

        return {
            "success": True,
            "data": {"results": results},
            "error": None,
        }

    except Exception as exc:
        logger.exception("Unexpected error during similarity query: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }
