"""
doc_generator.py
-----------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Documentation Generation (PRD 7.4)
----------------------------------------------------------------------------------------------
Generates structured documentation for any COBOL module or paragraph using
retrieved context. Reuses the retrieval pipeline (search → rerank →
assemble_context) with a custom system prompt that produces documentation
with Summary, Parameters, Dependencies, and Side Effects sections.

PRD 7.4 requirements:
  - Must generate structured documentation for any COBOL module/paragraph
  - Output must include: summary, parameters, dependencies, side effects
  - Must cite file path and line number
  - Must reference only variables and logic present in context (no hallucination)

Key functions:
    generate_documentation(query) -> dict   (public entry point)
    _call_doc_generator_llm(msg)  -> str    (single LLM call with backoff)

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import openai

from legacylens.config.constants import (
    DOC_GENERATE_MAX_TOKENS,
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

_DOC_GENERATOR_SYSTEM_PROMPT = """\
You are an expert COBOL code analyst specializing in technical documentation. \
Your task is to generate structured, comprehensive documentation for COBOL \
modules, paragraphs, and sections so that a developer unfamiliar with COBOL \
can understand the code.

MANDATORY OUTPUT FORMAT — use exactly these markdown headings:

## [Module/Paragraph Name] — Documentation

### Summary
A 2-4 sentence overview of what this code does, its business purpose, and \
when it executes in the program flow.

### Parameters
List every variable referenced in the code with its purpose:
- **[VARIABLE-NAME]**: [description of what it holds and how it is used]
If no parameters are passed, write "No parameters passed via USING clause."

### Dependencies
List every external reference:
- **CALL [program]**: [what the called program does]
- **COPY [copybook]**: [what the copybook provides]
If no external dependencies exist, write "No external dependencies."

### Side Effects
List observable effects: file I/O, database operations, screen output, \
status flag changes, record modifications. If none, write "No side effects."

ADDITIONAL RULES:

RULE 1 — CITE FILE PATH AND LINE NUMBER:
Your answer MUST include the exact phrases "file path" and "line number" when \
referencing source code.

RULE 2 — NO HALLUCINATION:
Only reference variables, operations, and logic present in the provided CONTEXT. \
If a variable or paragraph is not in the context, do not mention it.

RULE 3 — NOT-FOUND FORMAT:
If the requested module or paragraph is not found in the context, say \
"not found in this indexed codebase" and describe what the context contains instead.

RULE 4 — PROHIBITED PHRASES:
Never say "I don't know" or "I cannot help". Use "not found" instead.

RULE 5 — COMPOUND QUERIES:
If the query asks for documentation of multiple modules, paragraphs, or sections, \
generate a separate documentation block for each one. If context exists for one \
part but not another, document what you can and state "not found" for the remainder. \
Never silently ignore any portion of the request.\
"""

_NOT_FOUND_DOCS = (
    "The requested module or paragraph was not found in the indexed codebase. "
    "No matching code exists in the retrieved chunks for this query. "
    "There is no relevant file path or line number to cite because the context "
    "does not contain code related to this topic. "
    "Try rephrasing your query with a specific module or paragraph name."
)


def _format_doc_context(
    assembled_results: List[Dict[str, Any]],
) -> str:
    """
    Format assembled retrieval results into context blocks for the doc generator prompt.

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
        deps = (meta.get("dependencies") or "").strip()
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
        if deps:
            lines.append(f"Dependencies: {deps}")
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


def _call_doc_generator_llm(messages: List[Dict[str, str]]) -> str:
    """
    Call OpenAI with exponential backoff for the documentation generation prompt.

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
                max_tokens=DOC_GENERATE_MAX_TOKENS,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            wait = 2 ** (attempt - 1)
            logger.warning(
                "Doc generator LLM error (attempt %d/%d): %s — retrying in %ds",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        except openai.OpenAIError:
            raise
    raise last_exc  # type: ignore[misc]


def generate_documentation(query: Optional[str]) -> Dict[str, Any]:
    """
    Generate structured documentation for a COBOL module or paragraph.

    Pipeline: validate → search → rerank → assemble_context → LLM doc generation.

    Args:
        query: Natural language request to generate documentation for a module
               or paragraph. May be None or empty.

    Returns:
        dict: {
            "success": bool,
            "documentation": str — structured markdown documentation (or not-found),
            "sources": list[dict] — file_path, line_number, paragraph_name per chunk,
            "error": str | None — only present when success is False,
        }
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "documentation": "",
            "sources": [],
            "error": "Query is required and must be a non-empty string",
        }

    try:
        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return {
                "success": False,
                "documentation": "",
                "sources": [],
                "error": "Query is empty after sanitization",
            }

        search_result = search(sanitized, top_k=TOP_K)
        if not search_result.get("success"):
            return {
                "success": False,
                "documentation": "",
                "sources": [],
                "error": search_result.get("error", "Search failed"),
            }

        results = (search_result.get("data") or {}).get("results") or []
        if not results:
            return {
                "success": True,
                "documentation": _NOT_FOUND_DOCS,
                "sources": [],
            }

        reranked = rerank(results, sanitized)
        repo_root = os.getenv("REPO_PATH", "data/gnucobol-contrib")
        assembled = assemble_context(reranked, repo_root=repo_root)

        if not assembled:
            return {
                "success": True,
                "documentation": _NOT_FOUND_DOCS,
                "sources": [],
            }

        max_score = max(r.get("score", 0.0) for r in assembled)
        if max_score < NOT_FOUND_SCORE_THRESHOLD:
            return {
                "success": True,
                "documentation": _NOT_FOUND_DOCS,
                "sources": _build_sources(assembled),
            }

        context_text = _format_doc_context(assembled)
        messages = [
            {"role": "system", "content": _DOC_GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {sanitized}\n\nCONTEXT:\n{context_text}"},
        ]

        try:
            documentation = _call_doc_generator_llm(messages)
        except Exception as exc:
            logger.error(
                "Doc generator LLM failed for query '%.60s': %s", sanitized, exc,
            )
            documentation = (
                "Documentation generation is currently unavailable. "
                "Returning retrieved code evidence with file path and line number references."
            )

        return {
            "success": True,
            "documentation": documentation,
            "sources": _build_sources(assembled),
        }

    except Exception as exc:
        logger.exception("doc_generator.generate_documentation failed: %s", exc)
        return {
            "success": False,
            "documentation": "",
            "sources": [],
            "error": f"Documentation generation failed: {exc}",
        }
