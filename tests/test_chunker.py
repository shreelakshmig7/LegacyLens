"""
test_chunker.py
---------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for ingestion/chunker.py
------------------------------------------------------------------------------------------
Validates COBOL chunking against real source files from data/gnucobol-contrib.
Tests cover: paragraph-level chunking (primary), section-level fallback, fixed-size
fallback for non-standard files, copybook (.cpy) handling, metadata schema completeness,
no mid-paragraph splits, and structured return format.

Reference files used:
  samples/DBsample/PostgreSQL/example1/PGMOD1.cbl — clean named paragraphs
  tools/Dump-Tools/dumphex.cbl                     — mixed structure
  Any .cpy file in the repo                        — copybook chunking

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import pathlib

import pytest

REPO_PATH = pathlib.Path(__file__).parent.parent / "data" / "gnucobol-contrib"
PGMOD1 = REPO_PATH / "samples" / "DBsample" / "PostgreSQL" / "example1" / "PGMOD1.cbl"
DUMPHEX = REPO_PATH / "tools" / "Dump-Tools" / "dumphex.cbl"

pytestmark = pytest.mark.skipif(
    not REPO_PATH.exists(),
    reason="data/gnucobol-contrib not found — clone the repo before running tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_file(path: pathlib.Path) -> dict:
    from legacylens.ingestion.chunker import chunk_file
    return chunk_file(str(path))


def _chunk_lines(code_lines: list, file_path: str = "test.cbl", is_copybook: bool = False) -> dict:
    from legacylens.ingestion.chunker import chunk_code_lines
    return chunk_code_lines(code_lines, file_path=file_path, is_copybook=is_copybook)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

def test_chunk_file_returns_success_dict() -> None:
    """chunk_file must return a structured dict with success and data keys."""
    result = _chunk_file(PGMOD1)
    assert isinstance(result, dict)
    assert result["success"] is True
    assert "data" in result
    assert result.get("error") is None


def test_data_contains_chunks_list() -> None:
    """Result data must contain a non-empty chunks list."""
    result = _chunk_file(PGMOD1)
    assert "chunks" in result["data"]
    assert isinstance(result["data"]["chunks"], list)
    assert len(result["data"]["chunks"]) > 0


def test_nonexistent_file_returns_failure() -> None:
    """chunk_file must return success=False for a missing file."""
    from legacylens.ingestion.chunker import chunk_file
    result = chunk_file("/nonexistent/file.cbl")
    assert result["success"] is False
    assert result.get("error") is not None


# ---------------------------------------------------------------------------
# Metadata schema — every chunk must have all required fields
# ---------------------------------------------------------------------------

def test_every_chunk_has_required_metadata() -> None:
    """Every chunk must have all metadata schema fields populated."""
    from legacylens.config.constants import METADATA_FIELDS
    result = _chunk_file(PGMOD1)
    for chunk in result["data"]["chunks"]:
        for field in METADATA_FIELDS:
            assert field in chunk, f"Missing metadata field '{field}' in chunk: {chunk}"


def test_chunk_text_field_present() -> None:
    """Every chunk must include a 'text' field with non-empty content."""
    result = _chunk_file(PGMOD1)
    for chunk in result["data"]["chunks"]:
        assert "text" in chunk
        assert isinstance(chunk["text"], str)
        assert len(chunk["text"].strip()) > 0


def test_line_range_is_valid_tuple() -> None:
    """line_range must be a list of [start, end] with start <= end."""
    result = _chunk_file(PGMOD1)
    for chunk in result["data"]["chunks"]:
        lr = chunk["line_range"]
        assert isinstance(lr, list) and len(lr) == 2, f"Bad line_range: {lr}"
        assert lr[0] <= lr[1], f"line_range start > end: {lr}"
        assert lr[0] >= 1, f"line_range start < 1: {lr}"


def test_file_path_in_metadata() -> None:
    """file_path metadata must match the input file path."""
    result = _chunk_file(PGMOD1)
    for chunk in result["data"]["chunks"]:
        assert PGMOD1.name in chunk["file_path"] or str(PGMOD1) in chunk["file_path"]


# ---------------------------------------------------------------------------
# Paragraph-level chunking (primary strategy)
# ---------------------------------------------------------------------------

def test_paragraph_names_captured() -> None:
    """PGMOD1.cbl has named paragraphs — paragraph_name must be non-empty for PROCEDURE chunks."""
    result = _chunk_file(PGMOD1)
    proc_chunks = [c for c in result["data"]["chunks"] if c["type"] == "PROCEDURE"]
    named = [c for c in proc_chunks if c["paragraph_name"]]
    assert len(named) > 0, "No PROCEDURE chunks have paragraph_name populated"


def test_known_paragraph_present() -> None:
    """MAIN-PGMOD1-EX is a known paragraph in PGMOD1.cbl — must appear in chunks."""
    result = _chunk_file(PGMOD1)
    names = [c["paragraph_name"] for c in result["data"]["chunks"]]
    assert "MAIN-PGMOD1-EX" in names, (
        f"Expected MAIN-PGMOD1-EX in chunks. Found: {names}"
    )


def test_no_mid_paragraph_split() -> None:
    """Paragraph names must not appear in the middle of a chunk's text."""
    result = _chunk_file(PGMOD1)
    for chunk in result["data"]["chunks"]:
        if chunk["paragraph_name"]:
            # The paragraph name should appear at the start of the chunk, not mid-text
            text_lines = chunk["text"].strip().splitlines()
            if len(text_lines) > 1:
                # paragraph header is first line — it must not reappear in body
                body = "\n".join(text_lines[1:])
                name = chunk["paragraph_name"] + "."
                assert name not in body, (
                    f"Paragraph name {name!r} found mid-chunk — possible split error"
                )


# ---------------------------------------------------------------------------
# Chunk type values — must use constants, not magic strings
# ---------------------------------------------------------------------------

def test_chunk_types_are_valid() -> None:
    """All chunk type values must be one of the defined CHUNK_TYPE_* constants."""
    from legacylens.config.constants import (
        CHUNK_TYPE_COPYBOOK,
        CHUNK_TYPE_DATA,
        CHUNK_TYPE_PROCEDURE,
    )
    valid = {CHUNK_TYPE_PROCEDURE, CHUNK_TYPE_DATA, CHUNK_TYPE_COPYBOOK}
    result = _chunk_file(PGMOD1)
    for chunk in result["data"]["chunks"]:
        assert chunk["type"] in valid, f"Invalid chunk type: {chunk['type']!r}"


# ---------------------------------------------------------------------------
# Copybook (.cpy) chunking
# ---------------------------------------------------------------------------

def test_copybook_chunks_have_copybook_type() -> None:
    """Chunks from a .cpy file must have type=CHUNK_TYPE_COPYBOOK."""
    from legacylens.config.constants import CHUNK_TYPE_COPYBOOK
    # Find any .cpy file in the repo
    cpy_files = list(REPO_PATH.rglob("*.cpy"))
    assert len(cpy_files) > 0, "No .cpy files found in repo"
    result = _chunk_file(cpy_files[0])
    assert result["success"] is True
    for chunk in result["data"]["chunks"]:
        assert chunk["type"] == CHUNK_TYPE_COPYBOOK, (
            f"Expected CHUNK_TYPE_COPYBOOK, got {chunk['type']!r}"
        )


# ---------------------------------------------------------------------------
# Fixed-size fallback chunking
# ---------------------------------------------------------------------------

def test_fixed_size_fallback_via_lines() -> None:
    """When no paragraph boundaries are found, fallback to fixed-size chunks."""
    # Provide code lines with no paragraph markers — pure DATA DIVISION content
    code_lines = [f"05 WS-VAR-{i:04d}        PIC X(10)." for i in range(600)]
    result = _chunk_lines(code_lines, file_path="test.cbl")
    assert result["success"] is True
    chunks = result["data"]["chunks"]
    # Must produce multiple chunks from 600 lines
    assert len(chunks) > 1, "Expected multiple chunks from 600 lines via fallback"


def test_fixed_size_chunks_respect_max_token_limit() -> None:
    """No fixed-size chunk may exceed MAX_CHUNK_TOKENS worth of text."""
    from legacylens.config.constants import MAX_CHUNK_TOKENS
    code_lines = [f"05 WS-VAR-{i:04d}        PIC X(10)." for i in range(600)]
    result = _chunk_lines(code_lines, file_path="test.cbl")
    assert result["success"] is True
    for chunk in result["data"]["chunks"]:
        # Rough token estimate: words / 0.75 — must stay under limit
        approx_tokens = len(chunk["text"].split()) / 0.75
        assert approx_tokens <= MAX_CHUNK_TOKENS * 1.1, (
            f"Chunk exceeds token limit: ~{approx_tokens:.0f} tokens"
        )


# ---------------------------------------------------------------------------
# Overlap between fixed-size chunks
# ---------------------------------------------------------------------------

def test_fixed_size_chunks_have_overlap() -> None:
    """Adjacent fixed-size chunks must share some content (overlap)."""
    code_lines = [f"05 WS-VAR-{i:04d}        PIC X(10)." for i in range(600)]
    result = _chunk_lines(code_lines, file_path="test.cbl")
    chunks = result["data"]["chunks"]
    if len(chunks) >= 2:
        text0_lines = set(chunks[0]["text"].strip().splitlines())
        text1_lines = set(chunks[1]["text"].strip().splitlines())
        overlap = text0_lines & text1_lines
        assert len(overlap) > 0, "No overlap found between adjacent fixed-size chunks"


# ---------------------------------------------------------------------------
# Oversized procedure paragraph splitting (no drop — preserve retrieval precision)
# ---------------------------------------------------------------------------

def test_oversized_paragraph_split_into_subchunks() -> None:
    """An oversized procedure paragraph must be split into multiple chunks, not dropped."""
    from legacylens.config.constants import MAX_CHUNK_TOKENS
    # Build procedure content with one paragraph that exceeds MAX_CHUNK_TOKENS.
    # ~500 tokens ≈ 375 words; use enough lines so one paragraph is oversized.
    lines_per_chunk = max(80, (MAX_CHUNK_TOKENS * 3) // 10)  # well over limit
    body_lines = [f"           MOVE WS-{i:04d} TO OUT-{i:04d}." for i in range(lines_per_chunk)]
    code_lines = [
        "       IDENTIFICATION DIVISION.",
        "       PROGRAM-ID. BIGPROG.",
        "       PROCEDURE DIVISION.",
        "       BIG-PARA.",
    ] + body_lines
    result = _chunk_lines(code_lines, file_path="big.cbl")
    assert result["success"] is True
    chunks = result["data"]["chunks"]
    proc_chunks = [c for c in chunks if c.get("type") == "PROCEDURE"]
    assert len(proc_chunks) >= 2, (
        "Oversized paragraph must be split into at least 2 procedure sub-chunks"
    )
    # All sub-chunks must keep the same paragraph_name
    for c in proc_chunks:
        assert c.get("paragraph_name") == "BIG-PARA", (
            f"Sub-chunk must preserve paragraph_name BIG-PARA, got {c.get('paragraph_name')!r}"
        )
    # Combined line ranges should cover the paragraph (no content dropped)
    all_lines_covered = set()
    for c in proc_chunks:
        lr = c["line_range"]
        for line_no in range(lr[0], lr[1] + 1):
            all_lines_covered.add(line_no)
    assert len(all_lines_covered) >= lines_per_chunk, (
        "Sub-chunks must cover the full paragraph line range"
    )


def test_oversized_procedure_paragraph_never_produces_overlimit_chunk() -> None:
    """Oversized PROCEDURE paragraphs must use fixed-size fallback and stay <= MAX_CHUNK_TOKENS."""
    from legacylens.config.constants import MAX_CHUNK_TOKENS

    body_lines = [f"           MOVE INPUT-{i:04d} TO OUTPUT-{i:04d}." for i in range(650)]
    code_lines = [
        "       IDENTIFICATION DIVISION.",
        "       PROGRAM-ID. NOSKIP.",
        "       PROCEDURE DIVISION.",
        "       HUGE-PARA.",
    ] + body_lines

    result = _chunk_lines(code_lines, file_path="noskip.cbl")
    assert result["success"] is True
    chunks = [c for c in result["data"]["chunks"] if c.get("type") == "PROCEDURE"]
    assert len(chunks) >= 2, "Expected fixed-size fallback to split oversized paragraph"

    for chunk in chunks:
        approx_tokens = len(chunk["text"].split()) / 0.75
        assert approx_tokens <= MAX_CHUNK_TOKENS * 1.02, (
            f"Procedure sub-chunk exceeds token limit: ~{approx_tokens:.0f} tokens"
        )


# ---------------------------------------------------------------------------
# Chunk count sanity on real file
# ---------------------------------------------------------------------------

def test_chunk_count_reasonable_for_dumphex() -> None:
    """dumphex.cbl (~200 lines) should produce at least 1 and at most 50 chunks."""
    result = _chunk_file(DUMPHEX)
    assert result["success"] is True
    count = len(result["data"]["chunks"])
    assert 1 <= count <= 50, f"Unexpected chunk count for dumphex.cbl: {count}"


# ---------------------------------------------------------------------------
# file_hash propagated from preprocessor to every chunk
# ---------------------------------------------------------------------------

def test_chunk_file_attaches_file_hash() -> None:
    """Every chunk produced by chunk_file must carry a non-empty file_hash field."""
    result = _chunk_file(PGMOD1)
    assert result["success"] is True
    for chunk in result["data"]["chunks"]:
        assert "file_hash" in chunk, "chunk must have a file_hash field"
        assert isinstance(chunk["file_hash"], str)
        assert len(chunk["file_hash"]) == 64, (
            f"file_hash must be a 64-char SHA-256 hex digest, got: {chunk['file_hash']!r}"
        )


def test_all_chunks_from_same_file_share_file_hash() -> None:
    """All chunks from the same source file must have the same file_hash."""
    result = _chunk_file(PGMOD1)
    assert result["success"] is True
    chunks = result["data"]["chunks"]
    hashes = {c["file_hash"] for c in chunks}
    assert len(hashes) == 1, f"Expected one unique hash per file, found {len(hashes)}"


# ---------------------------------------------------------------------------
# security_flag propagated from preprocessor to every chunk
# ---------------------------------------------------------------------------

def test_chunk_file_attaches_security_flag() -> None:
    """Every chunk produced by chunk_file must carry a security_flag bool field."""
    result = _chunk_file(PGMOD1)
    assert result["success"] is True
    for chunk in result["data"]["chunks"]:
        assert "security_flag" in chunk, "chunk must have a security_flag field"
        assert isinstance(chunk["security_flag"], bool)


# ---------------------------------------------------------------------------
# parent_section fallback — top-level DATA items must get "DATA DIVISION"
# ---------------------------------------------------------------------------

def test_data_chunks_before_any_section_get_data_division_fallback() -> None:
    """DATA items that appear with no DIVISION or SECTION header above them must get
    parent_section='DATA DIVISION' as fallback, not an empty string.
    This is the exact production case: current_section='' (initial) is never updated
    before the flush, so the fallback must fire."""
    from legacylens.ingestion.chunker import chunk_code_lines
    # Pure data lines, no DIVISION or SECTION headers at all
    code_lines = [
        "       01 WS-COUNTER PIC 9(4).",
        "       01 WS-FLAG PIC X.",
        "       01 WS-NAME PIC X(30).",
    ]
    result = chunk_code_lines(code_lines, file_path="test.cbl")
    assert result["success"] is True
    data_chunks = [c for c in result["data"]["chunks"] if c["type"] == "DATA"]
    assert len(data_chunks) > 0, "Expected at least one DATA chunk"
    for chunk in data_chunks:
        assert chunk["parent_section"] != "", (
            "parent_section must not be empty when no section header was seen"
        )
        assert chunk["parent_section"] == "DATA DIVISION", (
            f"Expected 'DATA DIVISION' fallback, got {chunk['parent_section']!r}"
        )


def test_data_chunks_with_explicit_section_keep_section_name() -> None:
    """DATA chunks under an explicit WORKING-STORAGE SECTION must keep that section name;
    the 'DATA DIVISION' fallback must not override an already-set section."""
    from legacylens.ingestion.chunker import chunk_code_lines
    # Only WORKING-STORAGE header and data items — section is set before data lines
    code_lines = [
        "       WORKING-STORAGE SECTION.",
        "       01 WS-COUNTER PIC 9(4).",
        "       01 WS-FLAG PIC X.",
    ]
    result = chunk_code_lines(code_lines, file_path="test.cbl")
    assert result["success"] is True
    data_chunks = [c for c in result["data"]["chunks"] if c["type"] == "DATA"]
    assert len(data_chunks) > 0, "Expected at least one DATA chunk"
    for chunk in data_chunks:
        assert "WORKING-STORAGE" in chunk["parent_section"], (
            f"Explicit WORKING-STORAGE SECTION must be preserved, got {chunk['parent_section']!r}"
        )
