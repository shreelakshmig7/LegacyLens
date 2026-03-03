"""
test_reference_scraper.py
-------------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for ingestion/reference_scraper.py
---------------------------------------------------------------------------------------------------
Validates static dependency extraction against real COBOL source files from
data/gnucobol-contrib. Tests cover: CALL extraction, COPY extraction, USING extraction,
comment-line exclusion, deduplication, structured return format, and attachment of
dependency lists to chunk metadata dicts.

Reference file: samples/DBsample/PostgreSQL/example1/PGMOD1.cbl
  Known CALL:  PGSQLMSG
  Known COPY:  LNSQLMSG.cpy, sqlca.cbl, LNMOD1.cpy
  Known USING: present in PROCEDURE DIVISION header and CALL statement

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import pathlib

import pytest

REPO_PATH = pathlib.Path(__file__).parent.parent / "data" / "gnucobol-contrib"
PGMOD1 = REPO_PATH / "samples" / "DBsample" / "PostgreSQL" / "example1" / "PGMOD1.cbl"

pytestmark = pytest.mark.skipif(
    not REPO_PATH.exists(),
    reason="data/gnucobol-contrib not found — clone the repo before running tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scrape(path: pathlib.Path) -> dict:
    from legacylens.ingestion.reference_scraper import scrape_dependencies
    return scrape_dependencies(str(path))


def _scrape_lines(lines: list) -> dict:
    from legacylens.ingestion.reference_scraper import scrape_lines
    return scrape_lines(lines)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

def test_scrape_returns_success_dict() -> None:
    """scrape_dependencies must return a structured dict."""
    result = _scrape(PGMOD1)
    assert isinstance(result, dict)
    assert result["success"] is True
    assert "data" in result
    assert result.get("error") is None


def test_data_has_required_keys() -> None:
    """Result data must contain calls, copies, usings, and all_dependencies keys."""
    result = _scrape(PGMOD1)
    data = result["data"]
    for key in ("calls", "copies", "usings", "all_dependencies"):
        assert key in data, f"Missing key in result data: {key}"


def test_nonexistent_file_returns_failure() -> None:
    """scrape_dependencies must return success=False for a missing file."""
    from legacylens.ingestion.reference_scraper import scrape_dependencies
    result = scrape_dependencies("/nonexistent/file.cbl")
    assert result["success"] is False
    assert result.get("error") is not None


# ---------------------------------------------------------------------------
# CALL extraction
# ---------------------------------------------------------------------------

def test_call_targets_extracted() -> None:
    """CALL statements must be extracted with the called program name."""
    result = _scrape(PGMOD1)
    calls = result["data"]["calls"]
    assert isinstance(calls, list)
    assert len(calls) > 0


def test_known_call_target_present() -> None:
    """PGSQLMSG is a known CALL target in PGMOD1.cbl."""
    result = _scrape(PGMOD1)
    calls = result["data"]["calls"]
    assert "PGSQLMSG" in calls, f"Expected PGSQLMSG in calls. Got: {calls}"


def test_call_via_lines() -> None:
    """scrape_lines must extract program name from CALL statement."""
    raw = [
        '      CALL "MYPROGRAM" USING WS-DATA',
        "      CALL 'ANOTHER-PROG'",
        "      CALL WS-PROG-NAME",          # dynamic CALL — variable, not literal
    ]
    result = _scrape_lines(raw)
    assert result["success"] is True
    calls = result["data"]["calls"]
    assert "MYPROGRAM" in calls
    assert "ANOTHER-PROG" in calls


def test_call_strips_quotes() -> None:
    """CALL target must be stored without surrounding quotes."""
    raw = ['      CALL "QUOTED-PROG" USING WS-DATA']
    result = _scrape_lines(raw)
    calls = result["data"]["calls"]
    assert "QUOTED-PROG" in calls
    assert '"QUOTED-PROG"' not in calls


# ---------------------------------------------------------------------------
# COPY extraction
# ---------------------------------------------------------------------------

def test_copy_targets_extracted() -> None:
    """COPY statements must be extracted with the copybook name."""
    result = _scrape(PGMOD1)
    copies = result["data"]["copies"]
    assert isinstance(copies, list)
    assert len(copies) > 0


def test_known_copy_target_present() -> None:
    """LNSQLMSG.cpy is a known COPY target in PGMOD1.cbl."""
    result = _scrape(PGMOD1)
    copies = result["data"]["copies"]
    # Stored with or without extension — check both
    found = any("LNSQLMSG" in c for c in copies)
    assert found, f"Expected LNSQLMSG in copies. Got: {copies}"


def test_copy_via_lines() -> None:
    """scrape_lines must extract copybook name from COPY statement."""
    raw = [
        '      COPY "MYBOOK.cpy".',
        "      COPY ANOTHERBOOK.",
    ]
    result = _scrape_lines(raw)
    copies = result["data"]["copies"]
    assert any("MYBOOK" in c for c in copies)
    assert any("ANOTHERBOOK" in c for c in copies)


# ---------------------------------------------------------------------------
# USING extraction
# ---------------------------------------------------------------------------

def test_using_targets_extracted() -> None:
    """USING clauses must be extracted."""
    result = _scrape(PGMOD1)
    usings = result["data"]["usings"]
    assert isinstance(usings, list)
    assert len(usings) > 0


def test_using_via_lines() -> None:
    """scrape_lines must extract parameter from USING clause."""
    raw = [
        "      PROCEDURE DIVISION USING LN-INPUT LN-OUTPUT.",
        "      CALL 'PROG' USING WS-PARAM-1 WS-PARAM-2",
    ]
    result = _scrape_lines(raw)
    usings = result["data"]["usings"]
    assert len(usings) > 0


# ---------------------------------------------------------------------------
# Comment line exclusion
# ---------------------------------------------------------------------------

def test_comment_lines_not_scraped() -> None:
    """CALL/COPY/USING in comment lines (col 7 = *) must not be extracted."""
    raw = [
        "*> CALL 'NOTREAL' USING FAKE-DATA",
        "      CALL 'REALPROGRAM' USING WS-REAL",
    ]
    result = _scrape_lines(raw)
    calls = result["data"]["calls"]
    assert "NOTREAL" not in calls, "Comment-line CALL must not be extracted"
    assert "REALPROGRAM" in calls


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_no_duplicate_dependencies() -> None:
    """Each dependency must appear at most once in the extracted lists."""
    raw = [
        '      CALL "MYPROG" USING WS-DATA',
        '      CALL "MYPROG" USING WS-OTHER',    # same CALL target repeated
        '      COPY "MYBOOK".',
        '      COPY "MYBOOK".',                   # same COPY target repeated
    ]
    result = _scrape_lines(raw)
    calls = result["data"]["calls"]
    copies = result["data"]["copies"]
    assert calls.count("MYPROG") == 1, "Duplicate CALL target found"
    assert any(copies.count(c) == 1 for c in copies), "Duplicate COPY target found"


# ---------------------------------------------------------------------------
# all_dependencies list
# ---------------------------------------------------------------------------

def test_all_dependencies_is_union() -> None:
    """all_dependencies must contain every entry from calls + copies + usings."""
    result = _scrape(PGMOD1)
    data = result["data"]
    all_deps = set(data["all_dependencies"])
    for item in data["calls"] + data["copies"]:
        assert item in all_deps, f"Item missing from all_dependencies: {item}"


# ---------------------------------------------------------------------------
# attach_dependencies — enriches chunk dicts with scraped dependency data
# ---------------------------------------------------------------------------

def test_attach_dependencies_populates_chunks() -> None:
    """attach_dependencies must set the dependencies field on each chunk dict."""
    from legacylens.ingestion.reference_scraper import attach_dependencies

    chunks = [
        {"text": '      CALL "PROG-A" USING WS-X', "dependencies": [], "file_path": "test.cbl",
         "line_range": [1, 1], "type": "PROCEDURE", "parent_section": "", "paragraph_name": "MAIN"},
        {"text": "      MOVE WS-A TO WS-B", "dependencies": [], "file_path": "test.cbl",
         "line_range": [2, 2], "type": "PROCEDURE", "parent_section": "", "paragraph_name": "CALC"},
    ]
    result = attach_dependencies(chunks)
    assert result["success"] is True
    enriched = result["data"]["chunks"]
    # First chunk has a CALL — dependencies must be non-empty
    assert len(enriched[0]["dependencies"]) > 0
    # Second chunk has no CALL/COPY/USING — dependencies may be empty list
    assert isinstance(enriched[1]["dependencies"], list)


def test_attach_dependencies_does_not_mutate_input() -> None:
    """attach_dependencies must not mutate the original chunk dicts."""
    from legacylens.ingestion.reference_scraper import attach_dependencies

    original = {"text": '      CALL "PROG-A"', "dependencies": [], "file_path": "test.cbl",
                "line_range": [1, 1], "type": "PROCEDURE", "parent_section": "", "paragraph_name": ""}
    chunks = [original]
    attach_dependencies(chunks)
    # Original dict dependencies field unchanged
    assert original["dependencies"] == []
