"""
code_explainer.py
-----------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Code Explanation (PRD 7.1)
--------------------------------------------------------------------------------------
Explains the purpose and behavior of any COBOL paragraph or section in plain English.
Reuses the retrieval pipeline (search → rerank → assemble_context) with a custom
system prompt focused on explanation rather than general Q&A.

PRD 7.1 requirements:
  - Explain purpose and behavior of any COBOL PARAGRAPH in plain English
  - Reference specific variables and conditions present in the retrieved code
  - No hallucination — only reference what is in context
  - Include source paragraph name and file:line reference in the response

Key functions:
    explain(query)           -> dict   (blocking, structured result)
    _call_explainer_llm(msg) -> str    (single LLM call with backoff)

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import openai

from legacylens.config.constants import (
    CODE_EXPLAIN_MAX_TOKENS,
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

_EXPLAINER_SYSTEM_PROMPT = """\
You are an expert COBOL code analyst. Your task is to explain COBOL paragraphs \
and sections in clear, plain English so that a developer unfamiliar with COBOL \
can understand the code's purpose, behavior, and business intent.

MANDATORY OUTPUT RULES:

RULE 1 — EXPLAIN PURPOSE AND BEHAVIOR:
For each paragraph or section, describe: (a) what it does, (b) why it exists \
(business purpose), and (c) what happens step by step.

RULE 2 — REFERENCE SPECIFIC VARIABLES AND CONDITIONS:
Name every variable (e.g. WS-CUST-NAME, CUSTOMER-RECORD) and every condition \
(IF, EVALUATE, PERFORM UNTIL) that appears in the code. Do not generalize — \
be specific about what each variable holds and what each condition checks.

RULE 3 — CITE FILE PATH AND LINE NUMBER:
Your answer MUST include the exact phrases "file path" and "line number" when \
referencing source code. Example: "The file path is data/.../cust01.cbl and \
the line number is 42."

RULE 4 — USE THE WORD "paragraph":
Always use the word "paragraph" when referring to a named COBOL code block.

RULE 5 — NO HALLUCINATION:
Only reference variables, operations, and logic present in the provided CONTEXT \
blocks. If a variable or paragraph name is not in the context, do not mention it.

RULE 6 — NOT-FOUND FORMAT:
If the requested paragraph or section is not found in the context, say \
"not found in this indexed codebase" and describe what the retrieved context \
does contain instead. Write at least 3 sentences.

RULE 7 — PROHIBITED PHRASES:
Never say "I don't know" or "I cannot help". Use "not found" instead.

RULE 8 — COMPOUND QUERIES:
If the query asks about multiple paragraphs, sections, or topics, address each one \
separately. If context exists for one part but not another, answer what you can and \
state "not found" for the remainder. Never silently ignore any portion of the request.\
"""

_NOT_FOUND_EXPLANATION = (
    "The requested paragraph or code section was not found in the indexed codebase. "
    "No matching paragraph exists in the retrieved chunks for this query. "
    "There is no relevant file path or line number to cite because the context "
    "does not contain code related to this topic. "
    "Try rephrasing your query with a specific paragraph name or file name."
)


def _format_explainer_context(
    assembled_results: List[Dict[str, Any]],
) -> str:
    """
    Format assembled retrieval results into context blocks for the explainer prompt.

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


def _call_explainer_llm(messages: List[Dict[str, str]]) -> str:
    """
    Call OpenAI with exponential backoff for the code explanation prompt.

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
                max_tokens=CODE_EXPLAIN_MAX_TOKENS,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            wait = 2 ** (attempt - 1)
            logger.warning(
                "Explainer LLM error (attempt %d/%d): %s — retrying in %ds",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        except openai.OpenAIError:
            raise
    raise last_exc  # type: ignore[misc]


def explain(query: Optional[str]) -> Dict[str, Any]:
    """
    Explain a COBOL paragraph or section in plain English using retrieved context.

    Pipeline: validate → search → rerank → assemble_context → LLM explanation.

    Args:
        query: Natural language question asking to explain a paragraph/section.
               May be None or empty.

    Returns:
        dict: {
            "success": bool,
            "explanation": str — plain English explanation (or not-found message),
            "sources": list[dict] — file_path, line_number, paragraph_name per chunk,
            "error": str | None — only present when success is False,
        }
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "explanation": "",
            "sources": [],
            "error": "Query is required and must be a non-empty string",
        }

    try:
        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return {
                "success": False,
                "explanation": "",
                "sources": [],
                "error": "Query is empty after sanitization",
            }

        search_result = search(sanitized, top_k=TOP_K)
        if not search_result.get("success"):
            return {
                "success": False,
                "explanation": "",
                "sources": [],
                "error": search_result.get("error", "Search failed"),
            }

        results = (search_result.get("data") or {}).get("results") or []
        if not results:
            return {
                "success": True,
                "explanation": _NOT_FOUND_EXPLANATION,
                "sources": [],
            }

        reranked = rerank(results, sanitized)
        repo_root = os.getenv("REPO_PATH", "data/gnucobol-contrib")
        assembled = assemble_context(reranked, repo_root=repo_root)

        if not assembled:
            return {
                "success": True,
                "explanation": _NOT_FOUND_EXPLANATION,
                "sources": [],
            }

        max_score = max(r.get("score", 0.0) for r in assembled)
        if max_score < NOT_FOUND_SCORE_THRESHOLD:
            return {
                "success": True,
                "explanation": _NOT_FOUND_EXPLANATION,
                "sources": _build_sources(assembled),
            }

        context_text = _format_explainer_context(assembled)
        messages = [
            {"role": "system", "content": _EXPLAINER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {sanitized}\n\nCONTEXT:\n{context_text}"},
        ]

        try:
            explanation = _call_explainer_llm(messages)
        except Exception as exc:
            logger.error("Explainer LLM failed for query '%.60s': %s", sanitized, exc)
            explanation = (
                "Code explanation is currently unavailable. "
                "Returning retrieved code evidence with file path and line number references."
            )

        return {
            "success": True,
            "explanation": explanation,
            "sources": _build_sources(assembled),
        }

    except Exception as exc:
        logger.exception("code_explainer.explain failed: %s", exc)
        return {
            "success": False,
            "explanation": "",
            "sources": [],
            "error": f"Code explanation failed: {exc}",
        }
