"""
preprocessor.py
---------------
LegacyLens — RAG System for Legacy Enterprise Codebases — COBOL source preprocessor
-------------------------------------------------------------------------------------
Cleans raw COBOL source files before chunking and embedding. Applies the full
COBOL 72-column fixed-format rules:

  Cols 1-6  (index 0-5):  Sequence numbers — stripped
  Col  7    (index 6):    Indicator area — drives special handling:
                            ' ' (space)  → normal code line
                            '*' or '*>' → comment line → extracted to metadata, not embedded
                            '-'          → continuation line → joined to previous code line
                            'D' or 'd'   → debug line → discarded (treated as comment)
                            '/'          → page eject comment → discarded
  Cols 8-72 (index 7-71): Area A + Area B — the actual code content
  Cols 73-80 (index 72+): Identification area — stripped

Blank lines and whitespace-only lines after stripping are excluded from code_lines.
Comment text is preserved in a separate list for metadata use — it is never embedded.

Key functions:
    preprocess_file(file_path)     -> dict
    preprocess_lines(raw_lines)    -> dict

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import hashlib
import logging
import pathlib
import re
from typing import List, Tuple

from legacylens.config.constants import (
    COBOL_IDENTIFICATION_COLS,
    COBOL_SEQUENCE_COLS,
)

logger = logging.getLogger(__name__)

_SEQ_START, _SEQ_END = COBOL_SEQUENCE_COLS          # (0, 6)
_ID_START = COBOL_IDENTIFICATION_COLS[0]             # 72
_INDICATOR_COL = _SEQ_END                            # col index 6 (0-based)
_CODE_START = _SEQ_END + 1                           # col index 7

# ---------------------------------------------------------------------------
# PII detection patterns (recommended PRD §3.4 / §12.3)
# ---------------------------------------------------------------------------
# Patterns are applied to preprocessed code area lines (cols 7-71, after sequence
# and identification area stripping). Matched literal values are replaced with
# [REDACTED] before the text is sent to any external embedding API.

_PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # IPv4 addresses — must be inside a string literal (single or double quotes) AND
    # each octet must be 0-255. This prevents false positives from COBOL version strings
    # (e.g. 2.1.4.0), unquoted initialiser values (0.0.0.0), and period-terminated
    # decimal literals which are common in fixed-format COBOL source.
    (
        re.compile(
            r"""(?<=["'])"""
            r"""(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"""
            r"""(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}"""
            r"""(?=["'])"""
        ),
        "[REDACTED-IP]",
    ),
    # SSN-like literals — must be inside a string literal to avoid matching
    # insertion-edited PIC clauses (e.g. PIC 999-99-9999) which have the same
    # digit pattern but are not actual SSN values.
    (re.compile(r"""(?<=["'])\d{3}-\d{2}-\d{4}(?=["'])"""), "[REDACTED-SSN]"),
    # Hardcoded password keyword followed by a quoted value — context already scoped
    # Matches: PASSWORD = 'secret', PASSWORD: "abc123"
    (re.compile(r'(?i)\bPASSWORD\b\s*[:=]\s*["\'][^"\']+["\']'), "[REDACTED-PWD]"),
]


def _scan_and_redact_pii(line: str) -> Tuple[str, bool]:
    """
    Scan a preprocessed code line for PII patterns and replace matches with placeholders.

    Args:
        line: A single preprocessed code line (sequence/ID area already stripped).

    Returns:
        tuple[str, bool]: (redacted_line, pii_found) where pii_found is True if any
                          pattern matched and a substitution was made.
    """
    found = False
    for pattern, placeholder in _PII_PATTERNS:
        new_line, count = pattern.subn(placeholder, line)
        if count > 0:
            line = new_line
            found = True
    return line, found


_DEAD_CODE_HINTS = (
    "MOVE ",
    "IF ",
    "PERFORM ",
    "CALL ",
    "COPY ",
    "OPEN ",
    "READ ",
    "WRITE ",
    "REWRITE ",
    "PROCEDURE DIVISION",
)


def _looks_like_dead_code(comment_text: str) -> bool:
    """
    Heuristic check for commented-out code lines.

    Args:
        comment_text: Comment content with COBOL columns removed.

    Returns:
        bool: True when the comment resembles executable COBOL.
    """
    text = (comment_text or "").strip().upper()
    if not text:
        return False
    if text[:2].isdigit():
        return True
    return any(hint in text for hint in _DEAD_CODE_HINTS)


def preprocess_lines(raw_lines: List[str]) -> dict:
    """
    Preprocess a list of raw COBOL source lines in memory.

    Strips sequence numbers and identification area, handles the indicator
    column (comments, continuations, debug lines), joins continuation lines,
    and removes blank lines.

    Args:
        raw_lines: List of raw strings as read from a COBOL source file.
                   Lines may or may not have a trailing newline.

    Returns:
        dict: {
            "success": bool,
            "data": {
                "code_lines": list[str]  — cleaned, embeddable code lines,
                "comments":   list[str]  — extracted comment text (not embedded),
                "line_count": int        — len(code_lines),
            },
            "error": None | str,
        }
    """
    try:
        code_lines: List[str] = []
        comments: List[str] = []
        dead_code_count = 0
        security_flag = False

        for raw in raw_lines:
            # Strip trailing newline for uniform processing
            line = raw.rstrip("\n\r")

            # Lines shorter than 7 chars have no indicator — treat as code
            if len(line) <= _SEQ_END:
                content = line[_SEQ_START:].rstrip()
                if content:
                    content, pii_found = _scan_and_redact_pii(content)
                    if pii_found:
                        security_flag = True
                    code_lines.append(content)
                continue

            indicator = line[_INDICATOR_COL] if len(line) > _INDICATOR_COL else " "

            # Extract code area: cols 7-71 (strip sequence + identification)
            raw_code = line[_CODE_START:_ID_START].rstrip()

            if indicator in ("*", "/"):
                # Standard comment or page-eject — extract to comments, skip embedding
                comment_text = raw_code.lstrip("*>").strip()
                if comment_text:
                    comments.append(comment_text)
                    if _looks_like_dead_code(comment_text):
                        dead_code_count += 1

            elif indicator in ("D", "d"):
                # Debug line — discard entirely
                pass

            elif indicator == "-":
                # Continuation line — join to previous code line (PII scan after join)
                continuation = raw_code.lstrip()
                if code_lines:
                    joined = code_lines[-1] + continuation
                    joined, pii_found = _scan_and_redact_pii(joined)
                    if pii_found:
                        security_flag = True
                    code_lines[-1] = joined
                else:
                    # Continuation with no preceding line — treat as standalone
                    if continuation:
                        continuation, pii_found = _scan_and_redact_pii(continuation)
                        if pii_found:
                            security_flag = True
                        code_lines.append(continuation)

            else:
                # Normal code line (indicator is space or any unrecognised char)
                if raw_code.strip():
                    raw_code, pii_found = _scan_and_redact_pii(raw_code)
                    if pii_found:
                        security_flag = True
                    code_lines.append(raw_code)

        return {
            "success": True,
            "data": {
                "code_lines": code_lines,
                "comments": comments,
                "line_count": len(code_lines),
                "comment_count": len(comments),
                "comment_density": (len(comments) / len(code_lines)) if code_lines else 0.0,
                "dead_code_count": dead_code_count,
                "dead_code_flag": dead_code_count > 0,
                "security_flag": security_flag,
            },
            "error": None,
        }

    except Exception as exc:
        logger.exception("Unexpected error in preprocess_lines: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }


def _compute_file_hash(path: pathlib.Path) -> str:
    """
    Compute the SHA-256 hash of raw file bytes for version integrity tracking.

    Reading in binary mode so the hash is independent of text encoding and
    line-ending normalisation — the same physical file always produces the same hash.

    Args:
        path: Absolute or relative path to any file.

    Returns:
        str: 64-character lowercase hex digest, or empty string if the file cannot be read.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        logger.warning("Could not compute file hash for %s: %s", path, exc)
        return ""


def preprocess_file(file_path: str) -> dict:
    """
    Read a COBOL source file from disk and preprocess it.

    Handles encoding errors gracefully by replacing undecodable bytes so that
    files with legacy character sets or isolated bad bytes do not abort ingestion.

    Args:
        file_path: Absolute or relative path to a COBOL source file.

    Returns:
        dict: Same structure as preprocess_lines — see that function's docstring.

    Raises:
        Nothing — all errors are returned as {"success": False, "error": "..."}.
    """
    try:
        path = pathlib.Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"File not found: {file_path}",
            }

        if not path.is_file():
            return {
                "success": False,
                "data": None,
                "error": f"Path is not a file: {file_path}",
            }

        file_hash = _compute_file_hash(path)

        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()

        logger.debug("Preprocessing %s (%d raw lines)", file_path, len(raw_lines))
        result = preprocess_lines(raw_lines)

        if result["success"]:
            result["data"]["file_hash"] = file_hash
            logger.debug(
                "Preprocessed %s: %d code lines, %d comments, hash=%s",
                file_path,
                result["data"]["line_count"],
                len(result["data"]["comments"]),
                file_hash[:8],
            )

        return result

    except OSError as exc:
        logger.error("Failed to read file %s: %s", file_path, exc)
        return {
            "success": False,
            "data": None,
            "error": f"File read error: {str(exc)}",
        }
    except Exception as exc:
        logger.exception("Unexpected error preprocessing %s: %s", file_path, exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }
