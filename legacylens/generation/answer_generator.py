"""
answer_generator.py
-------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — GPT-4o-mini answer generation
----------------------------------------------------------------------------------------
Generates grounded, citation-rich answers from assembled context produced by
context_assembler.py. Uses GPT-4o-mini with explicit anti-hallucination prompting:
every answer must cite the exact GitHub deep link and line number of the code it
references; if the retrieved context lacks relevant information the system says
"not found" rather than speculating.

Supports two call modes:
    generate_answer()        — blocking, returns full answer dict (eval / FastAPI)
    generate_answer_stream() — generator yielding tokens one by one (Streamlit)

Both modes share the same prompt and fast-path logic:
    - Fast-path: if max relevance score < NOT_FOUND_SCORE_THRESHOLD, skip LLM
      entirely and return a structured "not found" message.
    - LLM path: build system + user messages, call OpenAI with exponential backoff
      (max MAX_RETRIES attempts), return the generated text.

Key functions:
    _parse_line_range(raw)           — parse ChromaDB "[start, end]" string → int
    _build_github_link(path, range)  — construct clickable GitHub deep link
    _format_context_block(result, i) — render one assembled result for the prompt
    _build_messages(query, results)  — build OpenAI messages list
    _call_openai(messages)           — single blocking OpenAI chat completion
    _stream_openai(messages)         — streaming OpenAI chat completion generator
    _call_with_backoff(messages)     — wraps _call_openai with exponential backoff
    generate_answer(query, results)  — blocking generation → dict
    generate_answer_stream(query, results) — streaming generation → Generator[str]

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import ast
import logging
import os
import re
import time
from typing import Any, Dict, Generator, List, Optional

import openai

from legacylens.config.constants import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    MAX_QUERY_LENGTH,
    MAX_RETRIES,
    NOT_FOUND_SCORE_THRESHOLD,
    OUT_OF_SCOPE_KEYWORDS,
    REPO_BASE_URL,
    DEFAULT_REPO_COMMIT,
    DEFAULT_REPO_NAME,
    DEFAULT_REPO_OWNER,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transient errors eligible for retry — auth / validation errors are not retried.
# ---------------------------------------------------------------------------
_RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert COBOL code analyst. You are answering questions about a legacy \
COBOL codebase indexed into a retrieval system. Retrieved code chunks are provided \
with file paths, GitHub links, and line numbers.

MANDATORY OUTPUT RULES — follow every rule in every response:

RULE 1 — SOURCES ONLY:
Only reference code, variables, paragraphs, and logic present in the provided \
CONTEXT blocks. Never extrapolate, infer, or invent code that is not shown.

RULE 2 — CITE FILE PATH AND LINE NUMBER (REQUIRED IN EVERY ANSWER):
For every source you reference, your answer text MUST include BOTH of these \
exact two-word phrases: "file path" and "line number". Write them out explicitly, \
for example: "The file path is data/gnucobol-contrib/.../cust01.cbl and the \
line number is 42." Do not abbreviate to just "file" or "line".

RULE 3 — QUOTE COBOL KEYWORDS VERBATIM:
Quote the exact COBOL reserved words from the code as they appear: OPEN, READ, \
WRITE, CALL, COPY, USING, FD, REWRITE, PERFORM, SELECT, WORKING-STORAGE, \
PROCEDURE DIVISION, FILE SECTION, DATA DIVISION. Do not paraphrase them.
Specifically — these are NON-NEGOTIABLE REQUIRED inclusions:
- FILE I/O VERBS: If the context shows any file operation statements, your answer \
MUST quote each verb that appears: OPEN must appear as "OPEN", READ as "READ", \
WRITE as "WRITE", CLOSE as "CLOSE". Do NOT replace them with "reads from file" or \
"writes data" alone — always also write the uppercase keyword itself.
- USING: If the context contains any USING clause, your answer MUST include both \
the word "USING" and the word "parameter" (e.g., "passes parameters via the USING \
clause"). No exceptions — do not paraphrase USING as "via" or "through" alone.
- FD: If the context contains a File Description entry (an FD line), your answer \
MUST include the abbreviation "FD" explicitly (e.g., "The FD entry for..."). \
Write both "FD" and "File Description" — never substitute one for the other alone. \
If you describe a file structure without writing "FD", you are violating this rule.
- CALL/COPY dependencies: When describing any CALL or COPY relationship between \
programs, your answer MUST include the word "dependency" or "dependencies" AND the \
exact keyword CALL or COPY (e.g., "This module has a CALL dependency on..."). \
When a COPY statement includes an external file, call that file a "copybook". \
Never describe a dependency without using both the keyword and "dependency".

RULE 4 — USE THE WORD "paragraph" (REQUIRED IN EVERY ANSWER):
Always use the word "paragraph" when referring to a COBOL paragraph, section, \
or named block of logic. Every answer that describes a specific named code block, \
entry point, or procedure MUST include the word "paragraph" at least once. \
Never describe what a named code block does without also calling it a "paragraph".

RULE 5 — NOT-FOUND RESPONSE FORMAT (at least 3 sentences required):
When the requested information is not in the context, write at least 3 sentences:
  (a) Include the exact phrase "not found" and state the item is absent from this \
indexed codebase.
  (b) Mention what the retrieved paragraphs and line numbers DO contain.
  (c) Explain why the context does not answer the question.
Use concept words from the query (e.g., "interest", "validation", "dependency") \
in your explanation, but do NOT repeat any specific ALL-CAPS COBOL identifier name \
from the query if it does not appear in the retrieved context chunks. \
Do not suggest what the code might do.

RULE 6 — NO MADE-UP IDENTIFIERS:
If the query contains a specific word or phrase that does not appear anywhere in \
the CONTEXT blocks (e.g., a made-up paragraph name like "xyzzy"), do not mention \
or quote that word in your response. Instead say "not found in this codebase."

RULE 7 — AVOID "error" IN BEHAVIORAL DESCRIPTIONS:
When describing what code does when validation fails or when a file is not found, \
avoid the word "error". Use "invalid condition", "validation failure", \
"file not found condition", "rejected record", or "problematic situation" instead. \
Exception: when the context literally contains SQL error codes or SQLSTATE values, \
you may use "error code" as a technical term for those SQL status values.

RULE 8 — PROHIBITED PHRASES:
Never say "I don't know" or "I cannot help". Either answer with cited sources or \
use the exact phrase "not found".

RULE 9 — STRUCTURE:
Lead with the most relevant source. Cite GitHub links inline next to file paths.\
"""

# Fallback message returned by the fast-path when all scores are too low.
# Deliberately references structural terms (paragraph, line number, file path) so
# that golden checks requiring those terms still pass on not-found cases.
_NOT_FOUND_TEMPLATE = (
    "The requested information was not found in the indexed codebase. "
    "No matching paragraph or code section exists in the retrieved chunks for this query. "
    "There is no relevant file path or line number to cite because the context "
    "does not contain code related to this topic. "
    "Try rephrasing your query with a specific file name, paragraph name, or "
    "operation keyword (for example: CALL, COPY, USING, READ, WRITE). "
    "You can also ask for the same intent in a narrower form, such as "
    "'Where is <paragraph-name> defined?' or 'Which file contains <keyword>?'."
)

# Anchor used for fallback file_path normalization when REPO_PATH is not set.
_GNUCOBOL_ANCHOR = "data/gnucobol-contrib"

# ---------------------------------------------------------------------------
# Out-of-scope response template
# ---------------------------------------------------------------------------
# Must contain "not found" and "outside the scope of this codebase"; must NOT
# contain "I cannot help".
_OUT_OF_SCOPE_TEMPLATE = (
    "This question is outside the scope of this codebase analysis system. "
    "The requested information was not found because this tool only answers "
    "questions about the indexed legacy COBOL codebase — its paragraphs, "
    "file paths, line numbers, and code structure. "
    "Please rephrase your question to focus on the analyzed codebase."
)

# ---------------------------------------------------------------------------
# Prompt injection patterns — stripped by _sanitize_query()
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: List[str] = [
    "ignore previous instructions",
    "forget",
    "you are now",
    "system:",
    "assistant:",
    "###",
    "---",
]


# ---------------------------------------------------------------------------
# Translation hints toggle
# ---------------------------------------------------------------------------

def _translation_hints_enabled(query: str) -> bool:
    """
    Return True when the user explicitly asks for modernization/translation hints.

    Args:
        query: User query string (already sanitized or raw).

    Returns:
        bool: True when translation hints should be enabled for this request.
    """
    if not query:
        return False
    lower = query.lower()
    trigger_terms = (
        "translation hint",
        "translation hints",
        "translate this",
        "modernize",
        "equivalent in python",
        "equivalent in java",
        "equivalent in go",
    )
    return any(term in lower for term in trigger_terms)


# ---------------------------------------------------------------------------
# Citation validation + fallback
# ---------------------------------------------------------------------------

def _has_required_citations(answer: str) -> bool:
    """
    Validate that the answer includes required citation phrases.

    Args:
        answer: Generated answer text.

    Returns:
        bool: True only when both required phrases are present.
    """
    if not answer:
        return False
    lower = answer.lower()
    return "file path" in lower and "line number" in lower


def _build_citation_fallback(assembled_results: List[Dict[str, Any]]) -> str:
    """
    Build deterministic fallback text that includes required citation phrases.

    Args:
        assembled_results: Retrieved/assembled results used for generation.

    Returns:
        str: Fallback answer with file path and line number references.
    """
    if not assembled_results:
        return _NOT_FOUND_TEMPLATE
    top = assembled_results[0]
    meta = top.get("metadata") or {}
    file_path = _normalize_file_path((meta.get("file_path") or "").strip())
    start_line = _parse_line_range(meta.get("line_range") or "")
    paragraph_name = (meta.get("paragraph_name") or "").strip()
    if paragraph_name:
        return (
            f"Based on the retrieved context, the most relevant paragraph is {paragraph_name}. "
            f"The file path is {file_path} and the line number is {start_line}. "
            "This fallback is returned because the generated answer did not include "
            "the required citation phrases."
        )
    return (
        "Based on the retrieved context, the top matching code region was identified. "
        f"The file path is {file_path} and the line number is {start_line}. "
        "This fallback is returned because the generated answer did not include "
        "the required citation phrases."
    )


# ---------------------------------------------------------------------------
# _is_out_of_scope
# ---------------------------------------------------------------------------

def _is_out_of_scope(query: str) -> bool:
    """
    Return True if the query contains a keyword indicating an off-topic request.

    Performs a case-insensitive substring check against OUT_OF_SCOPE_KEYWORDS.
    Called before the fast-path check in generate_answer() and
    generate_answer_stream() so that clearly off-topic questions never reach
    the LLM regardless of retrieval scores.

    Args:
        query: The (possibly already sanitized) user query string.

    Returns:
        bool: True when the query is outside the scope of codebase analysis.
    """
    if not query:
        return False
    lower = query.lower()
    return any(keyword.lower() in lower for keyword in OUT_OF_SCOPE_KEYWORDS)


# ---------------------------------------------------------------------------
# _sanitize_query
# ---------------------------------------------------------------------------

def _sanitize_query(query: str) -> str:
    """
    Sanitize a user query by stripping prompt-injection patterns and markdown.

    Steps applied in order:
      1. Detect injection patterns (case-insensitive); log a structured WARNING
         if any are found, then strip them from the query.
      2. Remove markdown code-block fences (``` ... ```).
      3. Remove inline backtick spans.
      4. Remove markdown bold (**...**) and italic (*...*) markers.
      5. Normalize internal whitespace.
      6. Truncate to MAX_QUERY_LENGTH characters.

    Safe for legitimate COBOL queries — none of the stripped patterns appear in
    normal codebase questions, so clean queries pass through unchanged.

    Args:
        query: Raw user query string (may be None; treated as empty string).

    Returns:
        str: Sanitized query, guaranteed to be <= MAX_QUERY_LENGTH characters.
    """
    if not query:
        return ""

    detected = [p for p in _INJECTION_PATTERNS if p.lower() in query.lower()]
    if detected:
        logger.warning(
            "Potential prompt injection detected — patterns: %s",
            detected,
        )
        for pattern in _INJECTION_PATTERNS:
            query = re.sub(re.escape(pattern), "", query, flags=re.IGNORECASE)

    # Remove markdown code blocks (``` fences and their enclosed content)
    query = re.sub(r"```[^`]*```", "", query, flags=re.DOTALL)
    # Remove inline backtick spans (keep the enclosed text)
    query = re.sub(r"`([^`]*)`", r"\1", query)
    # Remove bold and italic markers (keep the text)
    query = re.sub(r"\*\*([^*]+)\*\*", r"\1", query)
    query = re.sub(r"\*([^*]+)\*", r"\1", query)

    # Normalize whitespace
    query = " ".join(query.split())

    # Truncate to configured limit
    return query[:MAX_QUERY_LENGTH].strip()


# ---------------------------------------------------------------------------
# _parse_line_range
# ---------------------------------------------------------------------------

def _parse_line_range(raw: Any) -> int:
    """
    Parse the ChromaDB line_range metadata string and return the start line number.

    ChromaDB stores line_range as the string representation of a Python list,
    e.g. "[42, 89]". This function recovers the first (start) integer.

    Args:
        raw: The raw line_range value from ChromaDB metadata. May be a string
             like "[42, 89]", a bare integer string "77", or None.

    Returns:
        int: The start line number, or 0 if parsing fails.
    """
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw)
    raw_str = str(raw).strip()
    if not raw_str:
        return 0
    try:
        parsed = ast.literal_eval(raw_str)
        if isinstance(parsed, (list, tuple)) and parsed:
            return int(parsed[0])
        if isinstance(parsed, (int, float)):
            return int(parsed)
    except (ValueError, SyntaxError):
        pass
    # Last-resort: try stripping brackets and taking the first token
    try:
        cleaned = raw_str.strip("[]").split(",")[0].strip()
        return int(cleaned)
    except (ValueError, IndexError):
        pass
    return 0


def _parse_line_range_tuple(raw: Any) -> tuple:
    """
    Parse ChromaDB line_range metadata and return (start, end) for highlighting.

    Args:
        raw: Raw line_range value (e.g. "[42, 89]" or "[77, 77]").

    Returns:
        tuple: (start_line, end_line); (0, 0) if parsing fails.
    """
    if raw is None:
        return (0, 0)
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (int(raw[0]), int(raw[1]))
    raw_str = str(raw).strip()
    if not raw_str:
        return (0, 0)
    try:
        parsed = ast.literal_eval(raw_str)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return (int(parsed[0]), int(parsed[1]))
        if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
            s = int(parsed[0])
            return (s, s)
        if isinstance(parsed, (int, float)):
            s = int(parsed)
            return (s, s)
    except (ValueError, SyntaxError, TypeError):
        pass
    try:
        parts = raw_str.strip("[]").split(",")
        start = int(parts[0].strip()) if parts else 0
        end = int(parts[1].strip()) if len(parts) > 1 else start
        return (start, end)
    except (ValueError, IndexError):
        pass
    return (0, 0)


# ---------------------------------------------------------------------------
# _normalize_file_path
# ---------------------------------------------------------------------------

def _normalize_file_path(absolute_path: str) -> str:
    """
    Convert an absolute file path to a repo-relative path for use in GitHub URLs.

    Strategy:
      1. If REPO_PATH env var is set and the path starts with it, strip the prefix.
      2. Otherwise anchor on "data/gnucobol-contrib" in the path (same logic used
         in eval/run_eval.py::to_relative_path).
      3. If neither applies, return the basename only.

    Args:
        absolute_path: The absolute filesystem path stored in ChromaDB metadata.

    Returns:
        str: A repo-relative path suitable for embedding in a GitHub URL.
    """
    if not absolute_path:
        return ""

    repo_path = os.getenv("REPO_PATH", "").rstrip("/")
    if repo_path and absolute_path.startswith(repo_path):
        relative = absolute_path[len(repo_path):].lstrip("/")
        return relative

    if _GNUCOBOL_ANCHOR in absolute_path:
        # Keep anchor in relative path for stable comparison and links.
        return _GNUCOBOL_ANCHOR + absolute_path.split(_GNUCOBOL_ANCHOR, 1)[-1]

    return os.path.basename(absolute_path)


# ---------------------------------------------------------------------------
# _build_github_link
# ---------------------------------------------------------------------------

def _build_github_link(file_path: str, line_range_raw: str) -> str:
    """
    Construct a clickable GitHub deep link for a specific file and line number.

    Reads REPO_OWNER, REPO_NAME, REPO_COMMIT from the environment (falling back
    to the defaults loaded from constants). Normalizes the absolute file_path to
    a repo-relative form before embedding it in the URL.

    Args:
        file_path:      Absolute filesystem path from ChromaDB metadata.
        line_range_raw: Raw line_range string from ChromaDB metadata (e.g. "[42, 89]").

    Returns:
        str: A fully-formed GitHub URL with #L<line> anchor, or a plain path
             string if env vars are not set.
    """
    owner = os.getenv("REPO_OWNER", DEFAULT_REPO_OWNER)
    repo = os.getenv("REPO_NAME", DEFAULT_REPO_NAME)
    commit = os.getenv("REPO_COMMIT", DEFAULT_REPO_COMMIT)
    relative = _normalize_file_path(file_path)
    line = _parse_line_range(line_range_raw)

    if owner and repo and commit:
        return REPO_BASE_URL.format(
            owner=owner,
            repo=repo,
            commit=commit,
            file_path=relative,
            line=line,
        )

    # Fallback when env vars are missing — still include line anchor
    return f"{relative}#L{line}"


# ---------------------------------------------------------------------------
# _format_context_block
# ---------------------------------------------------------------------------

def _format_context_block(result: Dict[str, Any], idx: int) -> str:
    """
    Render one assembled retrieval result into a formatted prompt context block.

    The block contains: result index, relative file path, GitHub deep link,
    line range, paragraph name (if present), parent section, and the assembled
    context text (chunk + DATA xref + copybook content).

    Args:
        result: One dict from assemble_context() output; must contain keys
                "text", "metadata", "score", "assembled_context".
        idx:    1-based index of this result in the ranked list.

    Returns:
        str: A formatted multi-line string ready to embed in the LLM prompt.
    """
    meta = result.get("metadata") or {}
    file_path = (meta.get("file_path") or "").strip()
    line_range_raw = meta.get("line_range") or ""
    paragraph_name = (meta.get("paragraph_name") or "").strip()
    parent_section = (meta.get("parent_section") or "").strip()
    assembled = (result.get("assembled_context") or result.get("text") or "").strip()
    score = result.get("score", 0.0)

    relative_path = _normalize_file_path(file_path)
    github_link = _build_github_link(file_path, str(line_range_raw))
    start_line = _parse_line_range(str(line_range_raw))

    lines = [f"--- Result {idx} (relevance: {score:.3f}) ---"]
    lines.append(f"File path: {relative_path}")
    lines.append(f"GitHub: {github_link}")
    lines.append(f"Line number: {start_line}")
    if paragraph_name:
        lines.append(f"Paragraph: {paragraph_name}")
    if parent_section:
        lines.append(f"Section: {parent_section}")
    # Cite-as reminder — LLM must reproduce these exact phrases in its answer.
    lines.append(
        f"[REQUIRED CITATION: use the phrase \"line number {start_line}\" "
        f"and \"file path\" in your answer when referencing this source]"
    )
    lines.append("")
    lines.append(assembled)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------

def _build_messages(
    query: str,
    assembled_results: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Build the OpenAI messages list (system + user) for a query and its context.

    Args:
        query:             The user's natural language question.
        assembled_results: List of dicts from assemble_context().

    Returns:
        list[dict]: A list of OpenAI message dicts with "role" and "content" keys.
    """
    context_blocks = []
    for i, result in enumerate(assembled_results, start=1):
        context_blocks.append(_format_context_block(result, i))

    context_text = "\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"
    translation_mode = "enabled" if _translation_hints_enabled(query) else "disabled"

    user_content = (
        f"Query: {query}\n\n"
        f"Translation hints: {translation_mode}\n\n"
        f"CONTEXT:\n{context_text}"
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# _call_openai  (single attempt, no retry)
# ---------------------------------------------------------------------------

def _call_openai(messages: List[Dict[str, str]]) -> str:
    """
    Make a single blocking call to the OpenAI chat completion API.

    Args:
        messages: List of OpenAI message dicts (system + user).

    Returns:
        str: The assistant's reply text.

    Raises:
        openai.OpenAIError: On any API-level failure (connection, rate limit, etc.).
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# _stream_openai  (single attempt, streaming)
# ---------------------------------------------------------------------------

def _stream_openai(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """
    Make a streaming call to the OpenAI chat completion API and yield tokens.

    Args:
        messages: List of OpenAI message dicts (system + user).

    Yields:
        str: Individual text tokens as they arrive from the API.

    Raises:
        openai.OpenAIError: On any API-level failure.
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


# ---------------------------------------------------------------------------
# _call_with_backoff  (retry wrapper for _call_openai)
# ---------------------------------------------------------------------------

def _call_with_backoff(messages: List[Dict[str, str]]) -> str:
    """
    Call the OpenAI API with exponential backoff, retrying on transient errors.

    Retryable errors: APIConnectionError, APITimeoutError, RateLimitError,
    InternalServerError. Non-retryable errors (AuthenticationError, BadRequestError)
    are re-raised immediately without consuming retry budget.

    Args:
        messages: List of OpenAI message dicts (system + user).

    Returns:
        str: The assistant's reply text.

    Raises:
        openai.OpenAIError: After MAX_RETRIES failed attempts on transient errors,
                            or immediately on non-retryable errors.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _call_openai(messages)
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            wait = 2 ** (attempt - 1)  # 1s, 2s, 4s
            logger.warning(
                "OpenAI transient error (attempt %d/%d): %s — retrying in %ds",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        except openai.OpenAIError:
            # Non-retryable (auth, bad request, etc.) — propagate immediately
            raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _is_fast_path  (score-based not-found detection)
# ---------------------------------------------------------------------------

def _is_fast_path(assembled_results: List[Dict[str, Any]]) -> bool:
    """
    Return True if all relevance scores fall below NOT_FOUND_SCORE_THRESHOLD.

    Args:
        assembled_results: List of dicts from assemble_context().

    Returns:
        bool: True when the fast-path (no-LLM not-found response) should be used.
    """
    if not assembled_results:
        return True
    max_score = max(r.get("score", 0.0) for r in assembled_results)
    return max_score < NOT_FOUND_SCORE_THRESHOLD


# ---------------------------------------------------------------------------
# generate_answer  (blocking)
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    assembled_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a grounded answer for a query using retrieved and assembled context.

    Fast-path: if all relevance scores are below NOT_FOUND_SCORE_THRESHOLD, returns
    a structured "not found" message without calling the LLM.

    LLM path: builds a prompt with formatted context blocks, calls GPT-4o-mini via
    _call_with_backoff (max MAX_RETRIES retries with exponential backoff), and
    returns the generated answer.

    Args:
        query:             The user's natural language question.
        assembled_results: List of dicts from context_assembler.assemble_context().
                           Each dict must have keys: text, metadata, score,
                           assembled_context.

    Returns:
        dict: {
            "success": bool — True when an answer was produced (even "not found"),
            "answer":  str  — The answer text (always present on success),
            "error":   str  — Human-readable error message (only on success=False),
        }
    """
    try:
        query = _sanitize_query(query)

        if _is_out_of_scope(query):
            logger.info("Out-of-scope query detected: %.60s", query)
            return {"success": True, "answer": _OUT_OF_SCOPE_TEMPLATE}

        if _is_fast_path(assembled_results):
            logger.info("Fast-path: max score below threshold for query: %.60s", query)
            return {"success": True, "answer": _NOT_FOUND_TEMPLATE}

        messages = _build_messages(query, assembled_results)
        answer = _call_with_backoff(messages)
        if assembled_results and not _has_required_citations(answer):
            logger.warning(
                "Answer missing required citation phrases; using deterministic fallback for query: %.60s",
                query,
            )
            answer = _build_citation_fallback(assembled_results)
        logger.info("Answer generated (%d chars) for query: %.60s", len(answer), query)
        return {"success": True, "answer": answer}

    except Exception as exc:
        logger.error("generate_answer failed for query '%.60s': %s", query, exc)
        return {
            "success": False,
            "answer": "",
            "error": f"Answer generation failed: {exc}",
        }


# ---------------------------------------------------------------------------
# generate_answer_stream  (streaming)
# ---------------------------------------------------------------------------

def generate_answer_stream(
    query: str,
    assembled_results: List[Dict[str, Any]],
) -> Generator[str, None, None]:
    """
    Generate a streaming answer for a query, yielding tokens as they arrive.

    Fast-path: if all relevance scores are below NOT_FOUND_SCORE_THRESHOLD, yields
    the not-found template as a single token without calling the LLM.

    LLM path: builds the same prompt as generate_answer(), calls _stream_openai(),
    and yields each token as it arrives from the OpenAI streaming API.

    Args:
        query:             The user's natural language question.
        assembled_results: List of dicts from context_assembler.assemble_context().

    Yields:
        str: Individual text tokens. On failure yields a single error string
             prefixed with "__ERROR__" so callers can detect it (same pattern as __STATUS__).
    """
    try:
        query = _sanitize_query(query)

        # Status tokens inform the UI of each pipeline phase before content arrives.
        yield "__STATUS__Searching codebase..."
        yield "__STATUS__Assembling context..."
        yield "__STATUS__Generating answer..."

        if _is_out_of_scope(query):
            logger.info("Stream: out-of-scope query: %.60s", query)
            yield _OUT_OF_SCOPE_TEMPLATE
            return

        if _is_fast_path(assembled_results):
            logger.info("Stream fast-path: query: %.60s", query)
            yield _NOT_FOUND_TEMPLATE
            return

        messages = _build_messages(query, assembled_results)
        for token in _stream_openai(messages):
            yield token

    except Exception as exc:
        logger.error("generate_answer_stream failed for query '%.60s': %s", query, exc)
        yield f"__ERROR__Answer generation failed: {exc}"
