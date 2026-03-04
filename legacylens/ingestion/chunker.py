"""
chunker.py
----------
LegacyLens — RAG System for Legacy Enterprise Codebases — COBOL source chunker
-------------------------------------------------------------------------------
Splits preprocessed COBOL source into semantically meaningful chunks ready for
embedding. Applies chunking strategies in priority order:

  1. Paragraph-level (primary)
     Splits at COBOL PARAGRAPH boundaries in the PROCEDURE DIVISION.
     Each named paragraph becomes one chunk. The paragraph header line is
     included as the first line of the chunk for full context.

  2. Section-level (secondary)
     If a section in the PROCEDURE DIVISION is too large to be embedded as
     a single chunk (> MAX_CHUNK_TOKENS tokens), it is split at its internal
     paragraph boundaries and each sub-paragraph becomes a chunk.

  3. Fixed-size with overlap (fallback)
     Used for DATA DIVISION content, copybook body content, and any code that
     lacks recognisable paragraph boundaries. Chunks are capped at
     MAX_CHUNK_TOKENS tokens with CHUNK_OVERLAP_TOKENS tokens of overlap
     between adjacent chunks.

Copybook files (.cpy) are chunked with the same strategies as .cbl/.cob files
but every chunk receives type=CHUNK_TYPE_COPYBOOK.

Key functions:
    chunk_file(file_path)                                -> dict
    chunk_code_lines(code_lines, file_path, is_copybook) -> dict

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple


def _file_name_from_path(file_path: str) -> str:
    """Return the uppercase stem of a file path (e.g. '/repo/ocesql.cbl' -> 'OCESQL')."""
    return pathlib.Path(file_path).stem.upper()

from legacylens.config.constants import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TYPE_COPYBOOK,
    CHUNK_TYPE_DATA,
    CHUNK_TYPE_PROCEDURE,
    MAX_CHUNK_TOKENS,
)
from legacylens.ingestion.preprocessor import preprocess_file

logger = logging.getLogger(__name__)

# COBOL paragraph header pattern: Area A (first 4 chars after stripping),
# an identifier made of letters/digits/hyphens, terminated with a period.
# Must not match DIVISION or SECTION keywords (those are structural, not paragraph names).
_PARA_RE = re.compile(
    r"^([A-Z][A-Z0-9\-]*)\.\s*$",
    re.IGNORECASE,
)
_DIVISION_RE = re.compile(r"\b(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"\b[A-Z][A-Z0-9\-]*\s+SECTION\b", re.IGNORECASE)
_PROC_DIVISION_RE = re.compile(r"\bPROCEDURE\s+DIVISION\b", re.IGNORECASE)
_DATA_DIVISION_RE = re.compile(r"\b(DATA|WORKING-STORAGE|LOCAL-STORAGE|LINKAGE|FILE)\s+(DIVISION|SECTION)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Token estimation (word-count proxy — no tokeniser dependency at this layer)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """
    Estimate the token count of a text string using a word-count heuristic.

    Args:
        text: The text to estimate tokens for.

    Returns:
        int: Estimated token count (words / 0.75, rounded up).
    """
    words = len(text.split())
    return max(1, int(words / 0.75))


# ---------------------------------------------------------------------------
# Fixed-size chunking with overlap
# ---------------------------------------------------------------------------

def _fixed_size_chunks(
    lines: List[str],
    start_line: int,
    file_path: str,
    chunk_type: str,
    parent_section: str,
) -> List[Dict[str, Any]]:
    """
    Split a list of code lines into fixed-size overlapping chunks.

    Args:
        lines:          Code lines to chunk (already preprocessed, no raw COBOL cols).
        start_line:     1-based line number of the first line in the source file.
        file_path:      Absolute path to the source file (stored in metadata).
        chunk_type:     One of CHUNK_TYPE_PROCEDURE, CHUNK_TYPE_DATA, CHUNK_TYPE_COPYBOOK.
        parent_section: Name of the enclosing DIVISION or SECTION.

    Returns:
        list[dict]: List of chunk dicts, each with text + full metadata schema.
    """
    chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        # Collect lines until we reach MAX_CHUNK_TOKENS
        chunk_lines: List[str] = []
        token_count = 0
        j = i
        while j < len(lines):
            line_tokens = _estimate_tokens(lines[j])
            if chunk_lines and token_count + line_tokens > MAX_CHUNK_TOKENS:
                break
            chunk_lines.append(lines[j])
            token_count += line_tokens
            j += 1

        if not chunk_lines:
            break

        chunk_start = start_line + i - 1
        chunk_end = start_line + j - 2

        chunks.append({
            "text": "\n".join(chunk_lines),
            "file_path": file_path,
            "file_name": _file_name_from_path(file_path),
            "line_range": [chunk_start, chunk_end],
            "type": chunk_type,
            "parent_section": parent_section,
            "paragraph_name": "",
            "dependencies": [],
        })

        # Advance i, keeping CHUNK_OVERLAP_TOKENS worth of lines for the next chunk.
        # Compute how many lines approximate the overlap token budget, then step back
        # from j by that amount — but ALWAYS guarantee i moves forward past its old
        # value so we cannot loop infinitely even when a single line fills a chunk.
        avg_tokens_per_line = token_count / max(1, len(chunk_lines))
        overlap_lines = int(CHUNK_OVERLAP_TOKENS / max(1, avg_tokens_per_line))
        new_i = j - overlap_lines
        i = max(i + 1, new_i)  # guaranteed forward progress

    return chunks


# ---------------------------------------------------------------------------
# Paragraph detection helpers
# ---------------------------------------------------------------------------

def _is_paragraph_header(line: str) -> Optional[str]:
    """
    Detect if a preprocessed code line is a COBOL paragraph header.

    Args:
        line: A single preprocessed code line (no sequence numbers, no id area).

    Returns:
        str: The paragraph name (without trailing period) if detected, else None.
    """
    stripped = line.strip()
    if not stripped:
        return None
    # Must not be a DIVISION or SECTION header
    if _DIVISION_RE.search(stripped) or _SECTION_RE.search(stripped):
        return None
    # Must not be a data item (starts with level number)
    if re.match(r"^\d+\s", stripped):
        return None
    m = _PARA_RE.match(stripped)
    if m:
        return m.group(1).upper()
    return None


def _is_procedure_division_start(line: str) -> bool:
    """Return True if line marks the start of the PROCEDURE DIVISION."""
    return bool(_PROC_DIVISION_RE.search(line))


def _is_data_related(line: str) -> bool:
    """Return True if line is a DATA DIVISION, WORKING-STORAGE, LINKAGE, or FILE SECTION."""
    return bool(_DATA_DIVISION_RE.search(line))


# ---------------------------------------------------------------------------
# Core paragraph-level chunking
# ---------------------------------------------------------------------------

def _paragraph_chunks(
    code_lines: List[str],
    file_path: str,
    chunk_type_procedure: str,
    chunk_type_data: str,
) -> List[Dict[str, Any]]:
    """
    Split code lines into paragraph-level chunks for PROCEDURE content and
    fixed-size chunks for DATA content.

    Args:
        code_lines:          Preprocessed code lines.
        file_path:           Source file path (for metadata).
        chunk_type_procedure: Chunk type for procedure chunks.
        chunk_type_data:      Chunk type for data chunks.

    Returns:
        list[dict]: All chunks produced from the file.
    """
    chunks: List[Dict[str, Any]] = []
    in_procedure = False
    current_section = ""
    current_para_name: Optional[str] = None
    current_para_lines: List[str] = []
    current_para_start = 1
    data_lines: List[str] = []
    data_start = 1

    def flush_paragraph(para_name: Optional[str], lines: List[str], start: int, end: int) -> None:
        if not lines:
            return
        text = "\n".join(lines)
        chunks.append({
            "text": text,
            "file_path": file_path,
            "file_name": _file_name_from_path(file_path),
            "line_range": [start, end],
            "type": chunk_type_procedure,
            "parent_section": current_section,
            "paragraph_name": para_name or "",
            "dependencies": [],
        })

    def flush_data(lines: List[str], start: int) -> None:
        if not lines:
            return
        sub = _fixed_size_chunks(lines, start, file_path, chunk_type_data, current_section)
        chunks.extend(sub)

    for idx, line in enumerate(code_lines, 1):
        stripped = line.strip()

        # Track division/section transitions
        if _DIVISION_RE.search(stripped):
            if _is_procedure_division_start(stripped):
                # Flush any accumulated data before switching
                flush_data(data_lines, data_start)
                data_lines = []
                in_procedure = True
                current_section = "PROCEDURE DIVISION"
                current_para_name = None
                current_para_lines = []
                current_para_start = idx
            else:
                # Non-procedure division — flush any open paragraph
                if in_procedure:
                    flush_paragraph(current_para_name, current_para_lines, current_para_start, idx - 1)
                    current_para_lines = []
                else:
                    flush_data(data_lines, data_start)
                    data_lines = []
                in_procedure = False
                current_section = stripped
                data_start = idx
            continue

        if _SECTION_RE.search(stripped):
            if in_procedure:
                # New section in PROCEDURE — flush current paragraph
                flush_paragraph(current_para_name, current_para_lines, current_para_start, idx - 1)
                current_para_lines = []
                current_para_name = None
                current_para_start = idx
            current_section = stripped
            continue

        if in_procedure:
            para_name = _is_paragraph_header(line)
            if para_name:
                # Flush previous paragraph
                flush_paragraph(current_para_name, current_para_lines, current_para_start, idx - 1)
                current_para_name = para_name
                current_para_lines = [line]
                current_para_start = idx
            else:
                current_para_lines.append(line)
        else:
            if not data_lines:
                data_start = idx
            data_lines.append(line)

    # Flush any remaining content
    if in_procedure:
        flush_paragraph(current_para_name, current_para_lines, current_para_start, len(code_lines))
    else:
        flush_data(data_lines, data_start)

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_code_lines(
    code_lines: List[str],
    file_path: str = "unknown.cbl",
    is_copybook: bool = False,
) -> dict:
    """
    Chunk a list of preprocessed COBOL code lines into embedding-ready segments.

    Args:
        code_lines:  List of cleaned code lines (output of preprocessor).
        file_path:   Path to the source file — stored in chunk metadata.
        is_copybook: If True, all chunks receive type=CHUNK_TYPE_COPYBOOK regardless
                     of division. Copybooks define shared data and logic that is
                     included by parent programs via COPY statements.

    Returns:
        dict: {
            "success": bool,
            "data": {
                "chunks": list[dict]  — each dict has text + full metadata schema,
                "chunk_count": int,
            },
            "error": None | str,
        }
    """
    try:
        if not code_lines:
            return {
                "success": True,
                "data": {"chunks": [], "chunk_count": 0},
                "error": None,
            }

        if is_copybook:
            # Copybooks: chunk with fixed-size strategy, all chunks get COPYBOOK type
            raw_chunks = _fixed_size_chunks(
                code_lines,
                start_line=1,
                file_path=file_path,
                chunk_type=CHUNK_TYPE_COPYBOOK,
                parent_section="COPYBOOK",
            )
        else:
            raw_chunks = _paragraph_chunks(
                code_lines,
                file_path=file_path,
                chunk_type_procedure=CHUNK_TYPE_PROCEDURE,
                chunk_type_data=CHUNK_TYPE_DATA,
            )

        # Filter out empty chunks that might have slipped through
        chunks = [c for c in raw_chunks if c["text"].strip()]

        logger.debug("Chunked %s: %d chunks produced", file_path, len(chunks))

        return {
            "success": True,
            "data": {"chunks": chunks, "chunk_count": len(chunks)},
            "error": None,
        }

    except Exception as exc:
        logger.exception("Unexpected error chunking %s: %s", file_path, exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }


def chunk_file(file_path: str) -> dict:
    """
    Preprocess and chunk a COBOL source file from disk.

    Automatically detects copybooks by file extension (.cpy) and applies the
    CHUNK_TYPE_COPYBOOK type to all resulting chunks.

    Args:
        file_path: Absolute or relative path to a COBOL source file.

    Returns:
        dict: Same structure as chunk_code_lines — see that function's docstring.
    """
    try:
        path = pathlib.Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"File not found: {file_path}",
            }

        is_copybook = path.suffix.lower() == ".cpy"

        preprocess_result = preprocess_file(file_path)
        if not preprocess_result["success"]:
            return {
                "success": False,
                "data": None,
                "error": f"Preprocessing failed: {preprocess_result['error']}",
            }

        code_lines = preprocess_result["data"]["code_lines"]
        return chunk_code_lines(code_lines, file_path=str(path.resolve()), is_copybook=is_copybook)

    except Exception as exc:
        logger.exception("Unexpected error in chunk_file for %s: %s", file_path, exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }
