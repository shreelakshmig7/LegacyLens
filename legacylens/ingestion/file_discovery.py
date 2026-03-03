"""
file_discovery.py
-----------------
LegacyLens — RAG System for Legacy Enterprise Codebases — COBOL file discovery
-------------------------------------------------------------------------------
Recursively scans a target repository directory and returns all COBOL source
files (.cbl, .cob, .cpy) as a deduplicated list of absolute paths. Enforces
path safety (no traversal outside the repo root), counts total lines of code,
and returns a structured result dict — never raises exceptions to the caller.

Key functions:
    discover_files(repo_path) -> dict

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import pathlib
from typing import List

from legacylens.config.constants import COBOL_EXTENSIONS

logger = logging.getLogger(__name__)


def _is_safe_path(base: pathlib.Path, target: pathlib.Path) -> bool:
    """
    Return True if target resolves to a path inside base.

    Args:
        base:   The trusted root directory (already resolved).
        target: The candidate path to validate.

    Returns:
        bool: True when target is within base, False on traversal attempt.
    """
    try:
        target.resolve().relative_to(base)
        return True
    except ValueError:
        return False


def _count_lines(path: pathlib.Path) -> int:
    """
    Count the number of lines in a file, handling encoding errors gracefully.

    Args:
        path: Absolute path to the file.

    Returns:
        int: Number of lines, or 0 if the file cannot be read.
    """
    try:
        with open(path, "r", errors="ignore", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def discover_files(repo_path: str) -> dict:
    """
    Recursively discover all COBOL source files within a repository directory.

    Filters by extension (.cbl, .cob, .cpy), deduplicates by resolved absolute
    path (handles case-insensitive filesystem collisions), rejects any path that
    resolves outside repo_path (path traversal guard), and counts total LOC.

    Args:
        repo_path: String path to the root of the target repository. Must be an
                   existing directory that resolves within the filesystem safely.

    Returns:
        dict: {
            "success": bool,
            "data": {
                "files": list[str]  — absolute paths to discovered COBOL files,
                "file_count": int   — len(files),
                "total_lines": int  — sum of line counts across all files,
            },
            "error": None | str,
        }
    """
    try:
        base = pathlib.Path(repo_path).resolve()

        # Reject paths that do not exist
        if not base.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Repository path does not exist: {repo_path}",
            }

        # Reject paths that are not directories
        if not base.is_dir():
            return {
                "success": False,
                "data": None,
                "error": f"Repository path is not a directory: {repo_path}",
            }

        # Reject traversal: repo_path itself must not escape a reasonable root.
        # We check that the resolved base is an actual directory (done above).
        # We also detect ../  patterns in the raw string before resolution.
        raw = pathlib.Path(repo_path)
        if any(part == ".." for part in raw.parts):
            return {
                "success": False,
                "data": None,
                "error": (
                    f"Path traversal detected in repo_path: {repo_path}. "
                    "Use an absolute path to the repository."
                ),
            }

        valid_extensions = {ext.lower() for ext in COBOL_EXTENSIONS}

        seen_resolved: set = set()
        discovered: List[pathlib.Path] = []

        for candidate in base.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in valid_extensions:
                continue

            resolved = candidate.resolve()

            # Path safety: file must resolve within the repo root
            if not _is_safe_path(base, candidate):
                logger.warning(
                    "Skipping file outside repo root (traversal guard): %s", candidate
                )
                continue

            # Deduplication by resolved path (handles macOS case-insensitive collisions)
            resolved_str = str(resolved)
            if resolved_str in seen_resolved:
                logger.debug("Skipping duplicate (case-collision): %s", candidate)
                continue
            seen_resolved.add(resolved_str)
            discovered.append(resolved)

        total_lines = sum(_count_lines(f) for f in discovered)
        file_list = [str(f) for f in sorted(discovered)]

        logger.info(
            "File discovery complete: %d files, %d LOC in %s",
            len(file_list),
            total_lines,
            repo_path,
        )

        return {
            "success": True,
            "data": {
                "files": file_list,
                "file_count": len(file_list),
                "total_lines": total_lines,
            },
            "error": None,
        }

    except Exception as exc:
        logger.exception("Unexpected error during file discovery: %s", exc)
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(exc)}",
        }
