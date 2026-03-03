"""
test_file_discovery.py
----------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for ingestion/file_discovery.py
------------------------------------------------------------------------------------------------
Validates file discovery against the real gnucobol-contrib repository cloned at
data/gnucobol-contrib. Tests cover: correct extension filtering, path deduplication,
path traversal rejection, LOC counting, and structured return format.

All tests require data/gnucobol-contrib to exist. If the repo is not cloned,
tests will be skipped with a clear message rather than failing for the wrong reason.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import pathlib

import pytest

REPO_PATH = pathlib.Path(__file__).parent.parent / "data" / "gnucobol-contrib"

# Skip every test in this file if the repo has not been cloned
pytestmark = pytest.mark.skipif(
    not REPO_PATH.exists(),
    reason="data/gnucobol-contrib not found — clone the repo before running tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover(repo_path: pathlib.Path = REPO_PATH) -> dict:
    from legacylens.ingestion.file_discovery import discover_files
    return discover_files(str(repo_path))


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

def test_discover_returns_success_dict() -> None:
    """discover_files must return a structured dict with success, data, and error keys."""
    result = _discover()
    assert isinstance(result, dict)
    assert "success" in result
    assert result["success"] is True
    assert "data" in result
    assert result.get("error") is None


def test_data_contains_required_keys() -> None:
    """Result data must include files, file_count, total_lines keys."""
    result = _discover()
    data = result["data"]
    assert "files" in data
    assert "file_count" in data
    assert "total_lines" in data


# ---------------------------------------------------------------------------
# Extension filtering
# ---------------------------------------------------------------------------

def test_only_cobol_extensions_returned() -> None:
    """Every discovered file must have a .cbl, .cob, or .cpy extension."""
    result = _discover()
    files = result["data"]["files"]
    valid_extensions = {".cbl", ".cob", ".cpy"}
    for f in files:
        assert pathlib.Path(f).suffix.lower() in valid_extensions, (
            f"Non-COBOL file included: {f}"
        )


def test_minimum_file_count() -> None:
    """Must discover at least 50 COBOL files (project minimum requirement)."""
    result = _discover()
    assert result["data"]["file_count"] >= 50


def test_minimum_loc_count() -> None:
    """Must count at least 10,000 lines of code (project minimum requirement)."""
    result = _discover()
    assert result["data"]["total_lines"] >= 10_000


def test_actual_known_file_discovered() -> None:
    """A known file from the repo must appear in the discovered list."""
    result = _discover()
    files = result["data"]["files"]
    # Use relative paths for comparison — file paths should be relative to repo root
    rel_paths = [pathlib.Path(f).name for f in files]
    # dumphex.cbl is a stable file in tools/Dump-Tools/
    assert "dumphex.cbl" in rel_paths, "Expected dumphex.cbl to be discovered"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_no_duplicate_paths() -> None:
    """No file path may appear more than once — deduplication by resolved path."""
    result = _discover()
    files = result["data"]["files"]
    resolved = [str(pathlib.Path(f).resolve()) for f in files]
    assert len(resolved) == len(set(resolved)), (
        f"Duplicate paths found: {len(resolved) - len(set(resolved))} duplicates"
    )


def test_file_count_matches_list_length() -> None:
    """file_count must equal len(files) — no off-by-one in the count."""
    result = _discover()
    assert result["data"]["file_count"] == len(result["data"]["files"])


# ---------------------------------------------------------------------------
# Path safety — traversal rejection
# ---------------------------------------------------------------------------

def test_all_paths_within_repo_root() -> None:
    """Every discovered path must resolve within the repo base directory."""
    result = _discover()
    repo_resolved = REPO_PATH.resolve()
    for f in result["data"]["files"]:
        resolved = pathlib.Path(f).resolve()
        assert str(resolved).startswith(str(repo_resolved)), (
            f"Path escape detected: {f} is outside {repo_resolved}"
        )


def test_path_traversal_rejected() -> None:
    """discover_files must fail safely when given a path containing traversal."""
    from legacylens.ingestion.file_discovery import discover_files
    # Pass a path that tries to escape via ../
    malicious = str(REPO_PATH) + "/../../../etc"
    result = discover_files(malicious)
    # Must return failure, not silently walk the filesystem
    assert result["success"] is False
    assert "error" in result


def test_nonexistent_path_returns_failure() -> None:
    """discover_files must return success=False for a path that does not exist."""
    from legacylens.ingestion.file_discovery import discover_files
    result = discover_files("/nonexistent/path/that/does/not/exist")
    assert result["success"] is False
    assert result.get("error") is not None


# ---------------------------------------------------------------------------
# File path format
# ---------------------------------------------------------------------------

def test_files_are_absolute_paths() -> None:
    """All returned file paths must be absolute paths (easy to open without chdir)."""
    result = _discover()
    for f in result["data"]["files"]:
        assert pathlib.Path(f).is_absolute(), f"Non-absolute path returned: {f}"
