"""
test_preprocessor.py
--------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for ingestion/preprocessor.py
----------------------------------------------------------------------------------------------
Validates COBOL preprocessing against real source files from data/gnucobol-contrib.
Tests cover: sequence number stripping (cols 1-6), identification area stripping (cols 73-80),
comment line extraction (col 7 = * or *>), continuation line joining (col 7 = -),
debug line handling (col 7 = D), whitespace normalisation, encoding resilience,
and the structured return format.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import pathlib

import pytest

REPO_PATH = pathlib.Path(__file__).parent.parent / "data" / "gnucobol-contrib"
DUMPHEX = REPO_PATH / "tools" / "Dump-Tools" / "dumphex.cbl"

pytestmark = pytest.mark.skipif(
    not REPO_PATH.exists(),
    reason="data/gnucobol-contrib not found — clone the repo before running tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess(path: pathlib.Path) -> dict:
    from legacylens.ingestion.preprocessor import preprocess_file
    return preprocess_file(str(path))


def _preprocess_lines(raw_lines: list) -> dict:
    from legacylens.ingestion.preprocessor import preprocess_lines
    return preprocess_lines(raw_lines)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

def test_preprocess_returns_success_dict() -> None:
    """preprocess_file must return a structured dict."""
    result = _preprocess(DUMPHEX)
    assert isinstance(result, dict)
    assert result["success"] is True
    assert "data" in result
    assert result.get("error") is None


def test_data_has_required_keys() -> None:
    """Result data must include code_lines, comments, and line_count."""
    result = _preprocess(DUMPHEX)
    data = result["data"]
    assert "code_lines" in data
    assert "comments" in data
    assert "line_count" in data


def test_nonexistent_file_returns_failure() -> None:
    """preprocess_file must return success=False for a missing file."""
    from legacylens.ingestion.preprocessor import preprocess_file
    result = preprocess_file("/nonexistent/file.cbl")
    assert result["success"] is False
    assert result.get("error") is not None


# ---------------------------------------------------------------------------
# Sequence number stripping (cols 1-6)
# ---------------------------------------------------------------------------

def test_sequence_numbers_stripped() -> None:
    """Sequence numbers (cols 0-5) must be removed from code lines on known input."""
    import re
    # Use controlled input — real COBOL files can have 6-digit numeric literals in code area
    raw = [
        "000010 IDENTIFICATION DIVISION.\n",
        "000020 PROGRAM-ID. TESTPROG.\n",
        "000030 DATA DIVISION.\n",
    ]
    result = _preprocess_lines(raw)
    seq_pattern = re.compile(r"^\d{6}\s")
    for line in result["data"]["code_lines"]:
        assert not seq_pattern.match(line), (
            f"Sequence number + space not stripped from: {line!r}"
        )


def test_sequence_strip_via_lines() -> None:
    """preprocess_lines must strip cols 0-5 from each raw COBOL line."""
    raw = [
        "000010 IDENTIFICATION DIVISION.\n",
        "000020 PROGRAM-ID. TESTPROG.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    for line in result["data"]["code_lines"]:
        assert not line[:1].isdigit(), f"Digits still present at start: {line!r}"


# ---------------------------------------------------------------------------
# Identification area stripping (cols 73-80)
# ---------------------------------------------------------------------------

def test_identification_area_stripped() -> None:
    """No code line may have content beyond column 72."""
    result = _preprocess(DUMPHEX)
    for line in result["data"]["code_lines"]:
        assert len(line.rstrip("\n")) <= 72, (
            f"Line exceeds 72 chars: {line!r}"
        )


def test_identification_area_via_lines() -> None:
    """preprocess_lines must strip cols 72+ (identification area) from 80-col lines."""
    # Identification area starts at index 72 (col 73, 1-indexed).
    # Construct a line where MYPROG occupies indices 72-77 — the identification area.
    # Format: 6-char seq + 1-char indicator + 65-char code area + "MYPROG  "
    code_area = "IDENTIFICATION DIVISION." + " " * (65 - len("IDENTIFICATION DIVISION."))
    raw = ["000010 " + code_area + "MYPROG  \n"]
    assert len(raw[0].rstrip("\n")) >= 72, "Test line too short — MYPROG not in id area"
    result = _preprocess_lines(raw)
    assert result["success"] is True
    for line in result["data"]["code_lines"]:
        assert "MYPROG" not in line, f"Identification area not stripped: {line!r}"


# ---------------------------------------------------------------------------
# Comment line extraction (col 7 = * or *>)
# ---------------------------------------------------------------------------

def test_comment_lines_not_in_code() -> None:
    """Lines with * or *> in col 7 must not appear in code_lines."""
    result = _preprocess(DUMPHEX)
    for line in result["data"]["code_lines"]:
        stripped = line.lstrip()
        assert not stripped.startswith("*"), (
            f"Comment line leaked into code_lines: {line!r}"
        )


def test_comment_lines_extracted() -> None:
    """Comment lines must be captured in the comments list."""
    result = _preprocess(DUMPHEX)
    assert len(result["data"]["comments"]) > 0, (
        "Expected comments to be extracted from dumphex.cbl"
    )


def test_comment_extraction_via_lines() -> None:
    """preprocess_lines must separate * comment lines into comments list."""
    raw = [
        "000010 IDENTIFICATION DIVISION.\n",
        "000020*> This is a comment line\n",
        "000030 PROGRAM-ID. TEST.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    assert len(result["data"]["comments"]) >= 1
    assert len(result["data"]["code_lines"]) == 2


# ---------------------------------------------------------------------------
# Continuation line joining (col 7 = -)
# ---------------------------------------------------------------------------

def test_continuation_lines_joined() -> None:
    """A line with '-' in col 7 must be joined to the preceding code line."""
    raw = [
        "000010 MOVE 'HELLO '\n",
        "000020-    'WORLD' TO WS-GREETING.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    code = result["data"]["code_lines"]
    # Should produce one joined line, not two separate lines
    assert len(code) == 1
    joined = code[0]
    assert "HELLO" in joined and "WORLD" in joined


# ---------------------------------------------------------------------------
# Debug line handling (col 7 = D)
# ---------------------------------------------------------------------------

def test_debug_lines_not_in_code() -> None:
    """Lines with 'D' in col 7 must not appear in code_lines."""
    raw = [
        "000010 IDENTIFICATION DIVISION.\n",
        "000020D    DISPLAY 'DEBUG OUTPUT'.\n",
        "000030 PROGRAM-ID. TEST.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    for line in result["data"]["code_lines"]:
        assert "DEBUG OUTPUT" not in line


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------

def test_no_trailing_whitespace_in_code_lines() -> None:
    """Each code line must have trailing whitespace stripped."""
    result = _preprocess(DUMPHEX)
    for line in result["data"]["code_lines"]:
        assert line == line.rstrip(), (
            f"Trailing whitespace not stripped: {line!r}"
        )


def test_blank_lines_excluded() -> None:
    """Blank lines (whitespace-only after stripping) must not appear in code_lines."""
    result = _preprocess(DUMPHEX)
    for line in result["data"]["code_lines"]:
        assert line.strip() != "", "Blank line found in code_lines"


# ---------------------------------------------------------------------------
# Encoding resilience
# ---------------------------------------------------------------------------

def test_preprocess_handles_encoding_errors() -> None:
    """preprocess_file must succeed (not crash) on files with encoding issues."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".cbl", delete=False, mode="wb") as tf:
        tf.write(b"000010 IDENTIFICATION DIVISION.\n")
        tf.write(b"000020 PROGRAM-ID. \xff\xfe TEST.\n")  # invalid UTF-8 bytes
        tmp_path = tf.name
    try:
        from legacylens.ingestion.preprocessor import preprocess_file
        result = preprocess_file(tmp_path)
        assert result["success"] is True
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# line_count accuracy
# ---------------------------------------------------------------------------

def test_line_count_is_accurate() -> None:
    """line_count must equal len(code_lines) — counts only usable code lines."""
    result = _preprocess(DUMPHEX)
    data = result["data"]
    assert data["line_count"] == len(data["code_lines"])


# ---------------------------------------------------------------------------
# file_hash (SHA-256 of raw file bytes)
# ---------------------------------------------------------------------------

def test_file_hash_present_in_result() -> None:
    """preprocess_file must return a file_hash key in its data dict."""
    result = _preprocess(DUMPHEX)
    assert "file_hash" in result["data"], "file_hash must be returned by preprocess_file"


def test_file_hash_is_valid_sha256_hex() -> None:
    """file_hash must be a 64-character lowercase hex string (SHA-256 digest)."""
    result = _preprocess(DUMPHEX)
    fh = result["data"]["file_hash"]
    assert isinstance(fh, str)
    assert len(fh) == 64, f"Expected 64-char hex, got len={len(fh)}"
    assert all(c in "0123456789abcdef" for c in fh), "file_hash contains non-hex chars"


def test_file_hash_stable_across_calls() -> None:
    """The same file must produce the same hash on repeated calls."""
    r1 = _preprocess(DUMPHEX)
    r2 = _preprocess(DUMPHEX)
    assert r1["data"]["file_hash"] == r2["data"]["file_hash"]


# ---------------------------------------------------------------------------
# security_flag — PII detection and redaction
# ---------------------------------------------------------------------------

def test_security_flag_false_for_clean_code() -> None:
    """security_flag must be False when no PII patterns are present."""
    raw = [
        "000010 IDENTIFICATION DIVISION.\n",
        "000020 PROGRAM-ID. CLEAN.\n",
        "000030 DATA DIVISION.\n",
        "000040 WORKING-STORAGE SECTION.\n",
        "000050 01 WS-COUNTER PIC 9(4).\n",
        "000060 PROCEDURE DIVISION.\n",
        "000070 MAIN.\n",
        "000080     MOVE 0 TO WS-COUNTER.\n",
        "000090     STOP RUN.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    assert result["data"]["security_flag"] is False


def test_pii_ipv4_address_detected() -> None:
    """An IPv4 address literal in code lines must set security_flag=True."""
    raw = [
        "000010 IDENTIFICATION DIVISION.\n",
        "000020 DATA DIVISION.\n",
        "000030 WORKING-STORAGE SECTION.\n",
        "000040 01 WS-HOST PIC X(20) VALUE '192.168.1.100'.\n",
        "000050 PROCEDURE DIVISION.\n",
        "000060 MAIN. STOP RUN.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    assert result["data"]["security_flag"] is True, (
        "security_flag must be True when an IPv4 address is present in source"
    )


def test_pii_ssn_pattern_detected() -> None:
    """An SSN-like literal (NNN-NN-NNNN) must set security_flag=True."""
    raw = [
        "000010 IDENTIFICATION DIVISION.\n",
        "000020 DATA DIVISION.\n",
        "000030 WORKING-STORAGE SECTION.\n",
        "000040 01 WS-SSN PIC X(11) VALUE '123-45-6789'.\n",
        "000050 PROCEDURE DIVISION.\n",
        "000060 MAIN. STOP RUN.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    assert result["data"]["security_flag"] is True


def test_pii_redaction_replaces_ip_in_code_lines() -> None:
    """The IPv4 address must be replaced with [REDACTED] in the code_lines output."""
    raw = [
        "000010 DATA DIVISION.\n",
        "000020 WORKING-STORAGE SECTION.\n",
        "000030 01 WS-HOST PIC X(20) VALUE '10.0.0.1'.\n",
        "000040 PROCEDURE DIVISION.\n",
        "000050 MAIN. STOP RUN.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    combined = " ".join(result["data"]["code_lines"])
    assert "10.0.0.1" not in combined, "Raw IP address must not appear in code_lines after redaction"
    assert "[REDACTED-IP]" in combined, "IP redaction placeholder must appear in code_lines"


# ---------------------------------------------------------------------------
# Tightened IP regex — must NOT fire on COBOL version/dotted numeric literals
# ---------------------------------------------------------------------------

def test_bare_version_string_not_flagged_as_ip() -> None:
    """COBOL dotted version literals (e.g. 2.1.4.0) without quotes must NOT set security_flag."""
    raw = [
        "000010 IDENTIFICATION DIVISION.\n",
        "000020 PROGRAM-ID. VTEST.\n",
        "000030 DATA DIVISION.\n",
        "000040 WORKING-STORAGE SECTION.\n",
        "000050 01 WS-VER PIC X(10) VALUE 2.1.4.0.\n",
        "000060 PROCEDURE DIVISION.\n",
        "000070 MAIN. STOP RUN.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    assert result["data"]["security_flag"] is False, (
        "COBOL version string 2.1.4.0 (unquoted) must not trigger IP detection"
    )


def test_unquoted_dotted_numeric_not_flagged() -> None:
    """An unquoted dotted-decimal like 0.0.0.0 (e.g. initialiser) must NOT trigger IP flag."""
    raw = [
        "000010 DATA DIVISION.\n",
        "000020 WORKING-STORAGE SECTION.\n",
        "000030 01 WS-ADDR PIC X(15) VALUE 0.0.0.0.\n",
        "000040 PROCEDURE DIVISION.\n",
        "000050 MAIN. STOP RUN.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    assert result["data"]["security_flag"] is False, (
        "Unquoted 0.0.0.0 must not trigger IP detection"
    )


def test_ip_inside_string_literal_is_detected() -> None:
    """A real IP address inside a quoted string literal must set security_flag=True."""
    raw = [
        "000010 DATA DIVISION.\n",
        "000020 WORKING-STORAGE SECTION.\n",
        "000030 01 WS-HOST PIC X(15) VALUE '10.10.50.200'.\n",
        "000040 PROCEDURE DIVISION.\n",
        "000050 MAIN. STOP RUN.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    assert result["data"]["security_flag"] is True, (
        "IP address inside a quoted literal must set security_flag=True"
    )


# ---------------------------------------------------------------------------
# Tightened SSN regex — must NOT fire on unquoted PIC-style patterns
# ---------------------------------------------------------------------------

def test_pic_style_ssn_pattern_not_flagged() -> None:
    """An SSN-like pattern in a PIC clause without quotes must NOT set security_flag."""
    raw = [
        "000010 DATA DIVISION.\n",
        "000020 WORKING-STORAGE SECTION.\n",
        "000030 01 WS-SSN PIC 999-99-9999.\n",
        "000040 PROCEDURE DIVISION.\n",
        "000050 MAIN. STOP RUN.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    assert result["data"]["security_flag"] is False, (
        "SSN-like PIC pattern without quotes must not set security_flag"
    )


def test_ssn_inside_string_literal_is_detected() -> None:
    """An SSN inside a quoted VALUE literal must set security_flag=True."""
    raw = [
        "000010 DATA DIVISION.\n",
        "000020 WORKING-STORAGE SECTION.\n",
        "000030 01 WS-SSN PIC X(11) VALUE '987-65-4321'.\n",
        "000040 PROCEDURE DIVISION.\n",
        "000050 MAIN. STOP RUN.\n",
    ]
    result = _preprocess_lines(raw)
    assert result["success"] is True
    assert result["data"]["security_flag"] is True, (
        "SSN inside a quoted literal must set security_flag=True"
    )
