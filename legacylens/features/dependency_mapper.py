"""
dependency_mapper.py
--------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Dependency Mapping (PRD 7.2)
---------------------------------------------------------------------------------------
Surfaces all CALL, COPY, and USING references for a queried module using static
analysis metadata from the Reference Scraper stored in ChromaDB — not semantic search.

Approach:
  1. Extract the target module name from the user query.
  2. Query ChromaDB via collection.get() with a file_name where-filter.
  3. Parse the "dependencies" metadata field and the chunk document text for
     CALL / COPY / USING statements.
  4. Classify each dependency as internal (target exists in the indexed codebase)
     or external_copybook (target is a COPY member not in the index).
  5. Call the LLM to produce a narrative summary layered on top of the structured
     graph.  If the LLM fails, the structured data is still returned.

PRD 7.2 requirements:
  - Must surface all CALL, COPY, and USING references for a queried module
  - Must use static analysis data from the Reference Scraper, not only semantic search
  - Must indicate whether each dependency is internal module or external copybook
  - Must be answerable for any module present in the indexed codebase

Key functions:
    map_dependencies(query)               -> dict  (public entry point)
    _extract_module_name(query)           -> str | None
    _parse_dependencies_from_chunks(...)  -> dict
    _classify_dependency(target, index)   -> str
    _call_dependency_llm(messages)        -> str

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

import openai

from legacylens.config.constants import (
    DEPENDENCY_MAP_MAX_TOKENS,
    LLM_MODEL,
    MAX_RETRIES,
    PROGRAM_CATEGORIES,
)
from legacylens.generation.answer_generator import (
    _normalize_file_path,
    _parse_line_range,
    _sanitize_query,
)
from legacylens.retrieval.vector_store import _get_collection

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

_DEPENDENCY_SYSTEM_PROMPT = """\
You are an expert COBOL code analyst specializing in program dependencies. \
Given a structured dependency graph for a COBOL module, write a clear, \
concise narrative summary in plain English that a developer unfamiliar \
with COBOL can understand.

MANDATORY OUTPUT RULES:
1. List every CALL target by name and explain it is a sub-program invocation.
2. List every COPY target by name and explain it includes a copybook.
3. List every USING target by name and explain it is a parameter passed.
4. State whether each dependency is internal (another program in this codebase) \
   or external (a copybook or system module not in the index).
5. Keep the summary under 200 words.
6. Do NOT hallucinate — only reference dependencies present in the provided data.\
"""

_NOT_FOUND_SUMMARY = (
    "The requested module was not found in the indexed codebase. "
    "No dependency information is available for this module. "
    "Try a different module name (e.g. PGMOD1, CUST01)."
)

_CALL_RE = re.compile(r"CALL\s+['\"]?([A-Z0-9_-]+)['\"]?", re.IGNORECASE)
_COPY_RE = re.compile(r"COPY\s+([A-Z0-9_-]+)", re.IGNORECASE)
_USING_RE = re.compile(r"USING\s+([A-Z0-9_-]+(?:\s+[A-Z0-9_-]+)*)", re.IGNORECASE)


def _extract_module_name(query: str) -> Optional[str]:
    """
    Extract the target module name from a dependency query.

    Tries two strategies:
      1. Match a known program name from PROGRAM_CATEGORIES in the query.
      2. Regex fallback: look for an uppercase identifier after common patterns.

    Args:
        query: The user's natural language query string.

    Returns:
        str | None: Uppercase module name, or None if extraction fails.
    """
    if not query:
        return None

    upper_query = query.upper()

    for program in PROGRAM_CATEGORIES:
        if program in upper_query:
            return program

    patterns = [
        r"dependencies\s+of\s+([A-Z0-9_-]+)",
        r"depends\s+on\s+([A-Z0-9_-]+)",
        r"what\s+does\s+([A-Z0-9_-]+)\s+call",
        r"module\s+([A-Z0-9_-]+)",
        r"program\s+([A-Z0-9_-]+)",
    ]
    for pat in patterns:
        match = re.search(pat, upper_query)
        if match:
            return match.group(1)

    return None


def _get_all_indexed_file_names() -> List[str]:
    """
    Retrieve all distinct file_name values from the ChromaDB index.

    Used to classify dependencies as internal (target exists in index) vs external.

    Args:
        None

    Returns:
        list[str]: Uppercase file_name values from all indexed chunks.
    """
    try:
        collection = _get_collection()
        raw = collection.get(include=["metadatas"], limit=25000)
        metadatas = raw.get("metadatas") or []
        names: Set[str] = set()
        for m in metadatas:
            fn = (m.get("file_name") or "").strip().upper()
            if fn:
                names.add(fn)
        return list(names)
    except Exception as exc:
        logger.warning("Failed to fetch indexed file names: %s", exc)
        return list(PROGRAM_CATEGORIES)


def _classify_dependency(target: str, indexed_names: List[str]) -> str:
    """
    Classify a dependency target as internal or external.

    Args:
        target: Uppercase dependency target name.
        indexed_names: List of known indexed file names (uppercase).

    Returns:
        str: "internal" if target is in the index, else "external_copybook".
    """
    if target.upper() in [n.upper() for n in indexed_names]:
        return "internal"
    return "external_copybook"


def _parse_dependencies_from_chunks(
    chunks_data: Dict[str, Any],
    indexed_names: List[str],
) -> Dict[str, Any]:
    """
    Parse CALL, COPY, and USING dependencies from ChromaDB chunk data.

    Combines two sources:
      1. The "dependencies" metadata field (comma-separated, from Reference Scraper).
      2. Regex parsing of the document text for explicit CALL/COPY/USING statements.

    Args:
        chunks_data: Raw ChromaDB collection.get() return value with documents/metadatas.
        indexed_names: All indexed file names for internal/external classification.

    Returns:
        dict: {
            "calls": [{"target": str, "dep_type": str, "source_file": str, "paragraph": str}],
            "copies": [{"target": str, "dep_type": str, "source_file": str}],
            "usings": [{"target": str, "dep_type": str, "source_file": str, "paragraph": str}],
        }
    """
    calls: List[Dict[str, str]] = []
    copies: List[Dict[str, str]] = []
    usings: List[Dict[str, str]] = []

    seen_calls: Set[str] = set()
    seen_copies: Set[str] = set()
    seen_usings: Set[str] = set()

    documents = chunks_data.get("documents") or []
    metadatas = chunks_data.get("metadatas") or []

    for doc, meta in zip(documents, metadatas):
        file_path = (meta.get("file_path") or "").strip()
        file_name = (meta.get("file_name") or "").strip()
        paragraph = (meta.get("paragraph_name") or "").strip()
        doc_text = doc or ""

        for match in _CALL_RE.finditer(doc_text):
            target = match.group(1).upper()
            if target not in seen_calls:
                seen_calls.add(target)
                calls.append({
                    "target": target,
                    "dep_type": _classify_dependency(target, indexed_names),
                    "source_file": _normalize_file_path(file_path),
                    "paragraph": paragraph,
                })

        for match in _COPY_RE.finditer(doc_text):
            target = match.group(1).upper()
            if target not in seen_copies:
                seen_copies.add(target)
                copies.append({
                    "target": target,
                    "dep_type": _classify_dependency(target, indexed_names),
                    "source_file": _normalize_file_path(file_path),
                })

        for match in _USING_RE.finditer(doc_text):
            raw_params = match.group(1).upper().split()
            for param in raw_params:
                param = param.strip()
                if param and param not in seen_usings:
                    seen_usings.add(param)
                    usings.append({
                        "target": param,
                        "dep_type": _classify_dependency(param, indexed_names),
                        "source_file": _normalize_file_path(file_path),
                        "paragraph": paragraph,
                    })

        deps_field = (meta.get("dependencies") or "").strip()
        if deps_field:
            for dep in deps_field.split(","):
                dep = dep.strip().upper()
                if not dep:
                    continue
                if dep not in seen_calls and dep not in seen_copies and dep not in seen_usings:
                    dep_type = _classify_dependency(dep, indexed_names)
                    if any(kw in doc_text.upper() for kw in [f"CALL '{dep}'", f'CALL "{dep}"', f"CALL {dep}"]):
                        if dep not in seen_calls:
                            seen_calls.add(dep)
                            calls.append({
                                "target": dep,
                                "dep_type": dep_type,
                                "source_file": _normalize_file_path(file_path),
                                "paragraph": paragraph,
                            })
                    elif f"COPY {dep}" in doc_text.upper():
                        if dep not in seen_copies:
                            seen_copies.add(dep)
                            copies.append({
                                "target": dep,
                                "dep_type": dep_type,
                                "source_file": _normalize_file_path(file_path),
                            })

    return {"calls": calls, "copies": copies, "usings": usings}


def _build_sources(chunks_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build the sources list from chunks data for the structured return value.

    Args:
        chunks_data: Raw ChromaDB collection.get() return value.

    Returns:
        list[dict]: Each with file_path, line_number, paragraph_name.
    """
    sources: List[Dict[str, str]] = []
    seen: Set[str] = set()
    metadatas = chunks_data.get("metadatas") or []
    for meta in metadatas:
        fp = _normalize_file_path((meta.get("file_path") or "").strip())
        lr = meta.get("line_range") or ""
        key = f"{fp}:{lr}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file_path": fp,
                "line_number": str(_parse_line_range(str(lr))),
                "paragraph_name": (meta.get("paragraph_name") or "").strip(),
            })
    return sources


def _call_dependency_llm(messages: List[Dict[str, str]]) -> str:
    """
    Call OpenAI with exponential backoff for the dependency summary prompt.

    Args:
        messages: List of OpenAI message dicts (system + user).

    Returns:
        str: The assistant's reply text.

    Raises:
        openai.OpenAIError: After MAX_RETRIES failed attempts on transient errors.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=DEPENDENCY_MAP_MAX_TOKENS,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            wait = 2 ** (attempt - 1)
            logger.warning(
                "Dependency LLM error (attempt %d/%d): %s — retrying in %ds",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        except openai.OpenAIError:
            raise
    raise last_exc  # type: ignore[misc]


def map_dependencies(query: Optional[str]) -> Dict[str, Any]:
    """
    Map all CALL, COPY, and USING dependencies for a queried COBOL module.

    Pipeline: validate → extract module name → ChromaDB metadata query →
    parse dependencies → classify internal/external → LLM summary.

    Args:
        query: Natural language question about module dependencies.
               May be None or empty.

    Returns:
        dict: {
            "success": bool,
            "dependencies": {
                "calls": list[dict],
                "copies": list[dict],
                "usings": list[dict],
            } | None,
            "summary": str — LLM narrative summary (or not-found message),
            "sources": list[dict] — file_path, line_number, paragraph_name per chunk,
            "error": str | None — only present when success is False,
        }
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "dependencies": None,
            "summary": "",
            "sources": [],
            "error": "Query is required and must be a non-empty string",
        }

    try:
        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return {
                "success": False,
                "dependencies": None,
                "summary": "",
                "sources": [],
                "error": "Query is empty after sanitization",
            }

        module_name = _extract_module_name(sanitized)

        collection = _get_collection()

        if module_name:
            chunks_data = collection.get(
                where={"file_name": module_name},
                include=["documents", "metadatas"],
                limit=500,
            )
        else:
            chunks_data = collection.get(
                include=["documents", "metadatas"],
                limit=500,
            )

        documents = chunks_data.get("documents") or []
        if not documents:
            return {
                "success": True,
                "dependencies": {"calls": [], "copies": [], "usings": []},
                "summary": _NOT_FOUND_SUMMARY,
                "sources": [],
            }

        indexed_names = _get_all_indexed_file_names()
        deps = _parse_dependencies_from_chunks(chunks_data, indexed_names)
        sources = _build_sources(chunks_data)

        deps_text = (
            f"Module: {module_name or 'unknown'}\n"
            f"CALLs: {', '.join(c['target'] for c in deps['calls']) or 'none'}\n"
            f"COPYs: {', '.join(c['target'] for c in deps['copies']) or 'none'}\n"
            f"USINGs: {', '.join(c['target'] for c in deps['usings']) or 'none'}\n"
        )

        messages = [
            {"role": "system", "content": _DEPENDENCY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {sanitized}\n\nDEPENDENCY DATA:\n{deps_text}"},
        ]

        try:
            summary = _call_dependency_llm(messages)
        except Exception as exc:
            logger.error("Dependency LLM failed for query '%.60s': %s", sanitized, exc)
            summary = (
                f"LLM summary unavailable. Structured dependency data: "
                f"{len(deps['calls'])} CALL(s), {len(deps['copies'])} COPY(s), "
                f"{len(deps['usings'])} USING(s)."
            )

        return {
            "success": True,
            "dependencies": deps,
            "summary": summary,
            "sources": sources,
        }

    except Exception as exc:
        logger.exception("dependency_mapper.map_dependencies failed: %s", exc)
        return {
            "success": False,
            "dependencies": None,
            "summary": "",
            "sources": [],
            "error": f"Dependency mapping failed: {exc}",
        }
