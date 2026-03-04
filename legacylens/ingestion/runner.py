"""
runner.py
---------
LegacyLens — RAG System for Legacy Enterprise Codebases — Ingestion pipeline orchestrator
------------------------------------------------------------------------------------------
Top-level entry point for the full ingestion pipeline. Wires together:

  1. file_discovery   → find all COBOL files under REPO_PATH
  2. preprocessor     → clean fixed-format COBOL (strip sequence/ID cols, handle col-7)
  3. chunker          → split into paragraph/section/fixed-size chunks
  4. reference_scraper → attach CALL/COPY/USING dependency metadata to each chunk
  5. embedder         → generate Voyage Code 2 embeddings in batches
  6. vector_store     → insert into ChromaDB with zero-tolerance verification

Produces two artefacts in tests/results/:
  - ingestion_coverage_<timestamp>.json  — file counts, LOC, chunk counts, skipped totals
  - ingestion_timing_<timestamp>.json    — wall-clock time per stage and total

Usage:
    python -m legacylens.ingestion.runner

Environment variables required:
    REPO_PATH        — absolute path to the cloned target codebase
    VOYAGE_API_KEY   — Voyage AI API key
    CHROMA_PERSIST_DIR (optional, defaults to ./chroma_db)

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import json
import logging
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from legacylens.config.constants import validate_required_env_vars
from legacylens.ingestion.chunker import chunk_file
from legacylens.ingestion.embedder import embed_chunks
from legacylens.ingestion.file_discovery import discover_files
from legacylens.ingestion.reference_scraper import attach_dependencies
from legacylens.retrieval.vector_store import insert_chunks

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

RESULTS_DIR = pathlib.Path(__file__).parent.parent.parent / "tests" / "results"


# ---------------------------------------------------------------------------
# Result writers
# ---------------------------------------------------------------------------

def _save_json(name: str, data: dict) -> pathlib.Path:
    """
    Write a dict as JSON to tests/results/<name>_<timestamp>.json.

    Args:
        name: Base filename prefix (e.g. "ingestion_coverage").
        data: Serialisable dict.

    Returns:
        pathlib.Path: Path of the written file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{name}_{ts}.json"
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved result artefact: %s", path)
    return path


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _stage_discover(repo_path: str) -> tuple:
    """Run file discovery and return (result_dict, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = discover_files(repo_path)
    return result, time.perf_counter() - t0


def _stage_preprocess_and_chunk(
    files: List[str],
    repo_path: str,
) -> tuple:
    """
    Chunk each file (chunk_file handles preprocessing internally). Returns
    (all_chunks, failed_count, elapsed_seconds).

    Files that fail chunking are logged and skipped — the pipeline continues
    with the remaining files (structured error handling).
    """
    t0 = time.perf_counter()
    all_chunks: List[Dict[str, Any]] = []
    failed_files = 0

    for file_path in files:
        chunk_result = chunk_file(file_path)
        if not chunk_result["success"]:
            logger.warning("Chunking failed for %s: %s", file_path, chunk_result.get("error"))
            failed_files += 1
            continue

        all_chunks.extend(chunk_result["data"]["chunks"])

    elapsed = time.perf_counter() - t0
    logger.info(
        "Preprocess+chunk: %d chunks from %d files (%d failed) in %.1fs",
        len(all_chunks),
        len(files),
        failed_files,
        elapsed,
    )
    return all_chunks, failed_files, elapsed


def _stage_attach_deps(chunks: List[Dict[str, Any]]) -> tuple:
    """Run reference scraper over all chunks. Returns (enriched_chunks, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = attach_dependencies(chunks)
    if not result["success"]:
        logger.error("Dependency attachment failed: %s", result.get("error"))
        return chunks, time.perf_counter() - t0   # fall back to unenriched

    return result["data"]["chunks"], time.perf_counter() - t0


def _stage_embed(chunks: List[Dict[str, Any]]) -> tuple:
    """Embed chunks via Voyage AI. Returns (embedded_chunks, skipped, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = embed_chunks(chunks)
    elapsed = time.perf_counter() - t0

    if not result["success"]:
        logger.error("Embedding failed: %s", result.get("error"))
        return [], 0, elapsed

    return (
        result["data"]["chunks"],
        result["data"]["skipped_count"],
        elapsed,
    )


def _stage_store(chunks: List[Dict[str, Any]], repo_path: str) -> tuple:
    """Insert embedded chunks into ChromaDB. Returns (insert_result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = insert_chunks(chunks, repo_root=repo_path)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ingestion(repo_path: str) -> dict:
    """
    Execute the full ingestion pipeline against the target COBOL codebase.

    Args:
        repo_path: Absolute path to the cloned target repository.

    Returns:
        dict: {
            "success": bool,
            "data": {
                "files_discovered": int,
                "total_loc":        int,
                "chunks_produced":  int,
                "chunks_embedded":  int,
                "chunks_skipped":   int,
                "verified":         bool,
                "coverage_path":    str,
                "timing_path":      str,
            } | None,
            "error": None | str,
        }
    """
    pipeline_start = time.perf_counter()
    timings: Dict[str, float] = {}

    logger.info("=== LegacyLens ingestion pipeline starting ===")
    logger.info("Repo path: %s", repo_path)

    # ── 1. File discovery ────────────────────────────────────────────────────
    disc_result, t_disc = _stage_discover(repo_path)
    timings["discovery_s"] = round(t_disc, 2)

    if not disc_result["success"]:
        return {
            "success": False,
            "data": None,
            "error": f"File discovery failed: {disc_result.get('error')}",
        }

    files = disc_result["data"]["files"]
    total_loc = disc_result["data"]["total_lines"]
    logger.info(
        "Discovered %d files, %d LOC in %.1fs",
        len(files),
        total_loc,
        t_disc,
    )

    # ── 2. Preprocess + chunk ────────────────────────────────────────────────
    all_chunks, failed_files, t_chunk = _stage_preprocess_and_chunk(files, repo_path)
    timings["preprocess_chunk_s"] = round(t_chunk, 2)

    if not all_chunks:
        return {
            "success": False,
            "data": None,
            "error": "Chunking produced zero chunks — cannot proceed.",
        }

    # ── 3. Dependency attachment ─────────────────────────────────────────────
    all_chunks, t_deps = _stage_attach_deps(all_chunks)
    timings["dependency_scrape_s"] = round(t_deps, 2)

    # ── 4. Embedding ─────────────────────────────────────────────────────────
    embedded, skipped, t_embed = _stage_embed(all_chunks)
    timings["embedding_s"] = round(t_embed, 2)

    if not embedded:
        return {
            "success": False,
            "data": None,
            "error": "Embedding produced zero vectors — cannot proceed.",
        }

    # ── 5. Vector store insertion ────────────────────────────────────────────
    store_result, t_store = _stage_store(embedded, repo_path)
    timings["vector_store_s"] = round(t_store, 2)

    total_elapsed = time.perf_counter() - pipeline_start
    timings["total_s"] = round(total_elapsed, 2)

    if not store_result["success"]:
        return {
            "success": False,
            "data": None,
            "error": f"Vector store insertion failed: {store_result.get('error')}",
        }

    # ── 6. Save artefacts ────────────────────────────────────────────────────
    coverage = {
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
        "repo_path": repo_path,
        "files_discovered": len(files),
        "files_failed": failed_files,
        "total_loc": total_loc,
        "chunks_produced": len(all_chunks),
        "chunks_embedded": len(embedded),
        "chunks_skipped_oversized": skipped,
        "verified": store_result["data"]["verified"],
    }
    timing_report = {
        "run_at": coverage["run_at"],
        "stages": timings,
    }

    cov_path = _save_json("ingestion_coverage", coverage)
    tim_path = _save_json("ingestion_timing", timing_report)

    logger.info(
        "=== Ingestion complete: %d chunks stored in %.1fs ===",
        len(embedded),
        total_elapsed,
    )

    return {
        "success": True,
        "data": {
            "files_discovered": len(files),
            "total_loc": total_loc,
            "chunks_produced": len(all_chunks),
            "chunks_embedded": len(embedded),
            "chunks_skipped": skipped,
            "verified": store_result["data"]["verified"],
            "coverage_path": str(cov_path),
            "timing_path": str(tim_path),
        },
        "error": None,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env_check = validate_required_env_vars()
    if not env_check["success"]:
        raise SystemExit(f"Missing environment variables: {env_check['error']}")

    repo_path = os.getenv("REPO_PATH", "")
    if not repo_path:
        raise SystemExit("REPO_PATH environment variable is not set")

    outcome = run_ingestion(repo_path)
    if outcome["success"]:
        logger.info("Pipeline succeeded. Summary: %s", outcome["data"])
    else:
        raise SystemExit(f"Pipeline failed: {outcome['error']}")
