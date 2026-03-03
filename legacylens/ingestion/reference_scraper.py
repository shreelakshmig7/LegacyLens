"""
reference_scraper.py
--------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — COBOL dependency extractor
--------------------------------------------------------------------------------------
Static analysis component that scans COBOL source lines for three structural dependency
keywords and extracts the referenced names as structured metadata.

  CALL  "PROGRAM-NAME"  → external program invocations
  COPY  COPYBOOK-NAME   → copybook (shared code) inclusions
  USING variable-name   → parameter passing between modules

Comment lines (col 7 = * or starting with *>) are skipped so that references in
documentation comments are never mis-classified as real dependencies.

Results are deduplicated — if CALL "PROG-A" appears 10 times in a file, "PROG-A"
appears once in the output list.

Key functions:
    scrape_lines(lines)                -> dict  (scrape from in-memory lines)
    scrape_dependencies(file_path)     -> dict  (read + scrape from disk)
    attach_dependencies(chunks)        -> dict  (enrich chunk metadata dicts)

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import copy
import logging
import pathlib
import re
from typing import Any, Dict, List

from legacylens.config.constants import COBOL_DEPENDENCY_KEYWORDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for each dependency keyword
# ---------------------------------------------------------------------------

# CALL "PROGRAM" or CALL 'PROGRAM' — capture quoted literal only (dynamic CALL
# with a variable name is not captured since we cannot resolve it statically)
_CALL_RE = re.compile(
    r'\bCALL\s+["\']([A-Z0-9][A-Z0-9\-]*)["\']',
    re.IGNORECASE,
)

# COPY COPYBOOK. or COPY "COPYBOOK.cpy". or COPY COPYBOOK (no period)
_COPY_RE = re.compile(
    r'\bCOPY\s+["\']?([A-Z0-9][A-Z0-9\-\.]*[A-Z0-9])["\']?',
    re.IGNORECASE,
)

# USING var1 var2 ... — capture all space-separated identifiers after USING
# until end of line or another keyword
_USING_RE = re.compile(
    r'\bUSING\s+((?:[A-Z][A-Z0-9\-]*\s*)+)',
    re.IGNORECASE,
)

# Comment line detection — raw line, col 7 (index 6) = * or line stripped starts with *>
_COMMENT_COL7_RE = re.compile(r"^.{6}\*")


def _is_comment_line(line: str) -> bool:
    """
    Return True if this raw COBOL line is a comment and should be skipped.

    Args:
        line: A raw COBOL source line (may still have sequence numbers).

    Returns:
        bool: True when the line is a comment line.
    """
    stripped = line.strip()
    if stripped.startswith("*>") or stripped.startswith("*"):
        return True
    if len(line) > 6 and line[6] in ("*", "/"):
        return True
    return False


def scrape_lines(lines: List[str]) -> dict:
    """
    Scan a list of COBOL source lines and extract CALL, COPY, and USING references.

    Comment lines are skipped. Results are deduplicated. The function does not
    modify its input.

    Args:
        lines: List of COBOL source lines (raw or preprocessed).

    Returns:
        dict: {
            "success": bool,
            "data": {
                "calls":           list[str] — deduplicated CALL targets,
                "copies":          list[str] — deduplicated COPY targets,
                "usings":          list[str] — deduplicated USING parameters,
                "all_dependencies": list[str] — union of calls + copies + usings,
            },
            "error": None | str,
        }
    """
    try:
        seen_calls: set = set()
        seen_copies: set = set()
        seen_usings: set = set()

        for line in lines:
            if _is_comment_line(line):
                continue

            # Apply regexes to the full line — patterns are specific enough that
            # they will not false-match against sequence number digits. Both raw
            # COBOL (with 6-digit sequence numbers) and preprocessed lines (already
            # stripped) are handled correctly this way.
            code = line

            # CALL extraction
            for m in _CALL_RE.finditer(code):
                seen_calls.add(m.group(1).upper())

            # COPY extraction — strip trailing period if present
            for m in _COPY_RE.finditer(code):
                name = m.group(1).rstrip(".")
                seen_copies.add(name)

            # USING extraction — split on whitespace, filter valid identifiers
            for m in _USING_RE.finditer(code):
                params = m.group(1).split()
                for p in params:
                    clean = p.strip().rstrip(".,")
                    if re.match(r"^[A-Z][A-Z0-9\-]+$", clean, re.IGNORECASE):
                        seen_usings.add(clean.upper())

        calls = sorted(seen_calls)
        copies = sorted(seen_copies)
        usings = sorted(seen_usings)
        all_dependencies = sorted(seen_calls | seen_copies | seen_usings)

        return {
            "success": True,
            "data": {
                "calls": calls,
                "copies": copies,
                "usings": usings,
                "all_dependencies": all_dependencies,
            },
            "error": None,
        }

    except Exception as exc:
        logger.exception("Unexpected error in scrape_lines: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }


def scrape_dependencies(file_path: str) -> dict:
    """
    Read a COBOL source file from disk and extract its dependency references.

    Args:
        file_path: Path to a COBOL source file (.cbl, .cob, or .cpy).

    Returns:
        dict: Same structure as scrape_lines — see that function's docstring.
    """
    try:
        path = pathlib.Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"File not found: {file_path}",
            }

        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        logger.debug("Scraping dependencies in %s (%d lines)", file_path, len(lines))
        result = scrape_lines(lines)

        if result["success"]:
            data = result["data"]
            logger.debug(
                "Scraped %s: %d CALL, %d COPY, %d USING",
                file_path,
                len(data["calls"]),
                len(data["copies"]),
                len(data["usings"]),
            )

        return result

    except OSError as exc:
        logger.error("Failed to read %s for dependency scraping: %s", file_path, exc)
        return {
            "success": False,
            "data": None,
            "error": f"File read error: {str(exc)}",
        }
    except Exception as exc:
        logger.exception("Unexpected error scraping %s: %s", file_path, exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }


def attach_dependencies(chunks: List[Dict[str, Any]]) -> dict:
    """
    Enrich a list of chunk dicts by scraping each chunk's text for dependencies.

    Each chunk's dependencies field is populated with the union of CALL targets,
    COPY targets, and USING parameters found in that specific chunk's text.

    This function does NOT mutate the input chunk dicts — it returns new dicts
    in the result data.

    Args:
        chunks: List of chunk dicts as produced by the chunker. Each dict must
                have at minimum a "text" key and a "dependencies" key.

    Returns:
        dict: {
            "success": bool,
            "data": {
                "chunks": list[dict] — new chunk dicts with dependencies populated,
            },
            "error": None | str,
        }
    """
    try:
        enriched: List[Dict[str, Any]] = []

        for chunk in chunks:
            new_chunk = copy.deepcopy(chunk)
            text = chunk.get("text", "")
            lines = text.splitlines()

            result = scrape_lines(lines)
            if result["success"]:
                new_chunk["dependencies"] = result["data"]["all_dependencies"]
            else:
                new_chunk["dependencies"] = []
                logger.warning(
                    "Dependency scraping failed for chunk in %s: %s",
                    chunk.get("file_path", "unknown"),
                    result.get("error"),
                )

            enriched.append(new_chunk)

        return {
            "success": True,
            "data": {"chunks": enriched},
            "error": None,
        }

    except Exception as exc:
        logger.exception("Unexpected error in attach_dependencies: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }
