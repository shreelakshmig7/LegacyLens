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

import logging
import pathlib
from typing import List

from legacylens.config.constants import (
    COBOL_IDENTIFICATION_COLS,
    COBOL_SEQUENCE_COLS,
)

logger = logging.getLogger(__name__)

_SEQ_START, _SEQ_END = COBOL_SEQUENCE_COLS          # (0, 6)
_ID_START = COBOL_IDENTIFICATION_COLS[0]             # 72
_INDICATOR_COL = _SEQ_END                            # col index 6 (0-based)
_CODE_START = _SEQ_END + 1                           # col index 7


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

        for raw in raw_lines:
            # Strip trailing newline for uniform processing
            line = raw.rstrip("\n\r")

            # Lines shorter than 7 chars have no indicator — treat as code
            if len(line) <= _SEQ_END:
                content = line[_SEQ_START:].rstrip()
                if content:
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

            elif indicator in ("D", "d"):
                # Debug line — discard entirely
                pass

            elif indicator == "-":
                # Continuation line — join to previous code line
                continuation = raw_code.lstrip()
                if code_lines:
                    code_lines[-1] = code_lines[-1] + continuation
                else:
                    # Continuation with no preceding line — treat as standalone
                    if continuation:
                        code_lines.append(continuation)

            else:
                # Normal code line (indicator is space or any unrecognised char)
                # Also handles *> inline comments embedded in code lines
                if raw_code.strip():
                    code_lines.append(raw_code)

        return {
            "success": True,
            "data": {
                "code_lines": code_lines,
                "comments": comments,
                "line_count": len(code_lines),
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

        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()

        logger.debug("Preprocessing %s (%d raw lines)", file_path, len(raw_lines))
        result = preprocess_lines(raw_lines)

        if result["success"]:
            logger.debug(
                "Preprocessed %s: %d code lines, %d comments",
                file_path,
                result["data"]["line_count"],
                len(result["data"]["comments"]),
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
