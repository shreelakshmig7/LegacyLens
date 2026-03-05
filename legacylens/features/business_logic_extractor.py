"""
business_logic_extractor.py
----------------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Business Logic Extraction (PRD 7.3)
-----------------------------------------------------------------------------------------------
Identifies and extracts business rules embedded in COBOL code: IF/EVALUATE conditions,
thresholds, validation checks, data transformations, and control flow patterns.
Reuses the retrieval pipeline (search → rerank → assemble_context) with a custom
system prompt focused on rule extraction rather than general Q&A.

PRD 7.3 requirements:
  - Must identify and extract business rules embedded in COBOL code
  - Must surface IF/EVALUATE conditions, thresholds, validation checks
  - Must explain business intent behind conditional logic
  - Must reference specific variables and conditions from context

Key functions:
    extract_business_logic(query) -> dict   (public entry point)
    _call_business_logic_llm(msg) -> str    (single LLM call with backoff)

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import openai

from legacylens.config.constants import (
    BUSINESS_LOGIC_MAX_TOKENS,
    LLM_MODEL,
    MAX_RETRIES,
    NOT_FOUND_SCORE_THRESHOLD,
    TOP_K,
)
from legacylens.generation.answer_generator import (
    _build_github_link,
    _normalize_file_path,
    _parse_line_range,
    _sanitize_query,
)
from legacylens.retrieval.context_assembler import assemble_context
from legacylens.retrieval.reranker import rerank
from legacylens.retrieval.searcher import search

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

_BUSINESS_LOGIC_SYSTEM_PROMPT = """\
You are an expert COBOL code analyst specializing in business rule extraction. \
Your task is to identify, extract, and explain every business rule embedded in \
the provided COBOL code so that a developer unfamiliar with COBOL can understand \
the business intent.

MANDATORY OUTPUT RULES:

RULE 1 — IDENTIFY EVERY CONDITIONAL:
Find and describe every IF, EVALUATE, PERFORM UNTIL, and PERFORM VARYING \
statement. For each, state: (a) what condition is checked, (b) what variables \
are involved, and (c) what business decision it represents.

RULE 2 — SURFACE THRESHOLDS AND CONSTANTS:
Identify any numeric thresholds (e.g. "IF WS-BALANCE > 10000"), string \
comparisons (e.g. "IF WS-STATUS = 'ACTIVE'"), and status codes. Explain \
the business meaning of each threshold.

RULE 3 — EXPLAIN VALIDATION RULES:
For any input validation logic (range checks, format checks, required fields), \
describe what is being validated and what happens when validation fails.

RULE 4 — DESCRIBE DATA TRANSFORMATIONS:
Identify COMPUTE, MOVE, ADD, SUBTRACT, MULTIPLY, DIVIDE operations that \
implement business calculations. Explain the formula and its business purpose.

RULE 5 — CITE FILE PATH AND LINE NUMBER:
Your answer MUST include the exact phrases "file path" and "line number" when \
referencing source code.

RULE 6 — FORMAT AS NUMBERED RULES:
Present each business rule as a numbered item: "Business Rule N: [title] — [description]".

RULE 7 — NO HALLUCINATION:
Only reference variables, operations, and logic present in the provided CONTEXT. \
If a variable or paragraph is not in the context, do not mention it.

RULE 8 — NOT-FOUND FORMAT:
If no business rules are found in the context, say "no business rules found \
in this indexed codebase" and describe what the context does contain instead.

RULE 9 — COMPOUND QUERIES:
If the query asks about business rules across multiple programs or sections, \
address each one separately. If context exists for one part but not another, \
answer what you can and state "not found" for the remainder. Never silently \
ignore any portion of the request.\
"""

_NOT_FOUND_RULES = (
    "No business rules or conditional logic were not found in the indexed codebase "
    "for this query. No matching IF/EVALUATE conditions, thresholds, or validation "
    "checks exist in the retrieved chunks. "
    "Try rephrasing your query with a specific paragraph name or module name."
)


def _format_business_logic_context(
    assembled_results: List[Dict[str, Any]],
) -> str:
    """
    Format assembled retrieval results into context blocks for the business logic prompt.

    Args:
        assembled_results: List of dicts from assemble_context().

    Returns:
        str: Formatted context string for the LLM prompt.
    """
    if not assembled_results:
        return "(no context retrieved)"

    blocks = []
    for i, result in enumerate(assembled_results, start=1):
        meta = result.get("metadata") or {}
        fp = (meta.get("file_path") or "").strip()
        lr = meta.get("line_range") or ""
        para = (meta.get("paragraph_name") or "").strip()
        section = (meta.get("parent_section") or "").strip()
        assembled = (result.get("assembled_context") or result.get("text") or "").strip()
        score = result.get("score", 0.0)

        relative_path = _normalize_file_path(fp)
        line = _parse_line_range(str(lr))
        github = _build_github_link(fp, str(lr))

        lines = [f"--- Source {i} (relevance: {score:.3f}) ---"]
        lines.append(f"File path: {relative_path}")
        lines.append(f"GitHub: {github}")
        lines.append(f"Line number: {line}")
        if para:
            lines.append(f"Paragraph: {para}")
        if section:
            lines.append(f"Section: {section}")
        lines.append("")
        lines.append(assembled)
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _build_sources(
    assembled_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build the sources list from assembled results for the structured return value.

    Args:
        assembled_results: List of dicts from assemble_context().

    Returns:
        list[dict]: Each with file_path, line_number, paragraph_name, github_link, score.
    """
    sources = []
    for r in assembled_results:
        meta = r.get("metadata") or {}
        fp = (meta.get("file_path") or "").strip()
        lr = meta.get("line_range") or ""
        sources.append({
            "file_path": _normalize_file_path(fp),
            "line_number": _parse_line_range(str(lr)),
            "paragraph_name": (meta.get("paragraph_name") or "").strip(),
            "github_link": _build_github_link(fp, str(lr)),
            "score": float(r.get("score", 0.0)),
        })
    return sources


def _call_business_logic_llm(messages: List[Dict[str, str]]) -> str:
    """
    Call OpenAI with exponential backoff for the business logic extraction prompt.

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
                max_tokens=BUSINESS_LOGIC_MAX_TOKENS,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            wait = 2 ** (attempt - 1)
            logger.warning(
                "Business logic LLM error (attempt %d/%d): %s — retrying in %ds",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        except openai.OpenAIError:
            raise
    raise last_exc  # type: ignore[misc]


def extract_business_logic(query: Optional[str]) -> Dict[str, Any]:
    """
    Extract business rules and conditional logic from COBOL code using retrieved context.

    Pipeline: validate → search → rerank → assemble_context → LLM extraction.

    Args:
        query: Natural language question about business rules, validation logic,
               or conditional behavior. May be None or empty.

    Returns:
        dict: {
            "success": bool,
            "business_rules": str — extracted rules in numbered format (or not-found),
            "sources": list[dict] — file_path, line_number, paragraph_name per chunk,
            "error": str | None — only present when success is False,
        }
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "business_rules": "",
            "sources": [],
            "error": "Query is required and must be a non-empty string",
        }

    try:
        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return {
                "success": False,
                "business_rules": "",
                "sources": [],
                "error": "Query is empty after sanitization",
            }

        search_result = search(sanitized, top_k=TOP_K)
        if not search_result.get("success"):
            return {
                "success": False,
                "business_rules": "",
                "sources": [],
                "error": search_result.get("error", "Search failed"),
            }

        results = (search_result.get("data") or {}).get("results") or []
        if not results:
            return {
                "success": True,
                "business_rules": _NOT_FOUND_RULES,
                "sources": [],
            }

        reranked = rerank(results, sanitized)
        repo_root = os.getenv("REPO_PATH", "data/gnucobol-contrib")
        assembled = assemble_context(reranked, repo_root=repo_root)

        if not assembled:
            return {
                "success": True,
                "business_rules": _NOT_FOUND_RULES,
                "sources": [],
            }

        max_score = max(r.get("score", 0.0) for r in assembled)
        if max_score < NOT_FOUND_SCORE_THRESHOLD:
            return {
                "success": True,
                "business_rules": _NOT_FOUND_RULES,
                "sources": _build_sources(assembled),
            }

        context_text = _format_business_logic_context(assembled)
        messages = [
            {"role": "system", "content": _BUSINESS_LOGIC_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {sanitized}\n\nCONTEXT:\n{context_text}"},
        ]

        try:
            business_rules = _call_business_logic_llm(messages)
        except Exception as exc:
            logger.error(
                "Business logic LLM failed for query '%.60s': %s", sanitized, exc,
            )
            business_rules = (
                "Business logic extraction is currently unavailable. "
                "Returning retrieved code evidence with file path and line number references."
            )

        return {
            "success": True,
            "business_rules": business_rules,
            "sources": _build_sources(assembled),
        }

    except Exception as exc:
        logger.exception("business_logic_extractor.extract_business_logic failed: %s", exc)
        return {
            "success": False,
            "business_rules": "",
            "sources": [],
            "error": f"Business logic extraction failed: {exc}",
        }
