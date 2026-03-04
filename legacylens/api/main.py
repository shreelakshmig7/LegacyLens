"""
main.py
-------
LegacyLens — RAG System for Legacy Enterprise Codebases — FastAPI application
-----------------------------------------------------------------------------
FastAPI app exposing POST /query, POST /query/stream, and POST /admin/ingest.
Pipeline: sanitize → out-of-scope check → search → rerank → assemble_context →
generate_answer (or stream).
Returns structured JSON: answer, chunks, file_paths, line_numbers, github_links, relevance_scores.
All API keys via environment variables.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List

# Load .env from project root so VOYAGE_API_KEY, OPENAI_API_KEY, etc. are set when running via uvicorn
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from legacylens.config.constants import TOP_K, validate_required_env_vars
from legacylens.generation.answer_generator import (
    _OUT_OF_SCOPE_TEMPLATE,
    _build_github_link,
    _is_out_of_scope,
    _parse_line_range,
    _sanitize_query,
    generate_answer,
    generate_answer_stream,
)
from legacylens.retrieval.context_assembler import assemble_context
from legacylens.retrieval.reranker import rerank
from legacylens.retrieval.searcher import search

logger = logging.getLogger(__name__)

# Default repo root for copybook resolution when REPO_PATH is not set.
_DEFAULT_REPO_ROOT = "data/gnucobol-contrib"

app = FastAPI(
    title="LegacyLens API",
    description="Query interface for legacy codebase RAG.",
)


@app.on_event("startup")
def _startup_validate_env() -> None:
    """Log missing required env vars at startup."""
    result = validate_required_env_vars()
    if not result["success"]:
        logger.warning(
            "Missing required env vars: %s",
            result["missing"],
        )


class QueryRequest(BaseModel):
    """Request body for POST /query and POST /query/stream."""

    query: str = ""


def _build_metadata_from_assembled(
    assembled_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build chunks, file_paths, line_numbers, github_links, relevance_scores from assembled results.

    Args:
        assembled_results: List of dicts from assemble_context(); each has text, metadata, score.

    Returns:
        dict: Keys chunks, file_paths, line_numbers, github_links, relevance_scores (lists).
    """
    chunks: List[str] = []
    file_paths: List[str] = []
    line_numbers: List[int] = []
    github_links: List[str] = []
    relevance_scores: List[float] = []

    for r in assembled_results:
        meta = r.get("metadata") or {}
        fp = (meta.get("file_path") or "").strip()
        lr = meta.get("line_range") or ""
        line_num = _parse_line_range(lr)
        chunks.append((r.get("text") or "").strip())
        file_paths.append(fp)
        line_numbers.append(line_num)
        github_links.append(_build_github_link(fp, str(lr)))
        relevance_scores.append(float(r.get("score", 0.0)))

    return {
        "chunks": chunks,
        "file_paths": file_paths,
        "line_numbers": line_numbers,
        "github_links": github_links,
        "relevance_scores": relevance_scores,
    }


def _stream_query_response(sanitized: str, assembled: List[Dict[str, Any]]):
    """Generator: one JSON line (metadata), then tokens from generate_answer_stream."""
    meta = _build_metadata_from_assembled(assembled)
    yield json.dumps(meta) + "\n"
    for token in generate_answer_stream(sanitized, assembled):
        yield token


@app.post("/query")
def query(request: QueryRequest):
    """
    Run full RAG pipeline and return structured JSON: answer, chunks, file_paths, etc.

    Sanitizes query, checks out-of-scope (returns without calling search), then
    search → rerank → assemble_context → generate_answer.

    Args:
        request: Body with query string.

    Returns:
        On success: 200 with answer, chunks, file_paths, line_numbers, github_links, relevance_scores.
        On error: 400 for empty query, 500 for pipeline errors (JSON with success=False, error).
    """
    try:
        sanitized = _sanitize_query(request.query or "")
        if not sanitized.strip():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Empty query after sanitization"},
            )

        if _is_out_of_scope(sanitized):
            meta = _build_metadata_from_assembled([])
            return {
                "answer": _OUT_OF_SCOPE_TEMPLATE,
                "chunks": meta["chunks"],
                "file_paths": meta["file_paths"],
                "line_numbers": meta["line_numbers"],
                "github_links": meta["github_links"],
                "relevance_scores": meta["relevance_scores"],
            }

        search_result = search(sanitized, top_k=TOP_K)
        if not search_result.get("success"):
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": search_result.get("error", "Search failed")},
            )

        results = (search_result.get("data") or {}).get("results") or []
        reranked = rerank(results, sanitized)
        repo_root = os.getenv("REPO_PATH", _DEFAULT_REPO_ROOT)
        assembled = assemble_context(reranked, repo_root=repo_root)

        gen_result = generate_answer(sanitized, assembled)
        if not gen_result.get("success"):
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": gen_result.get("error", "Answer generation failed"),
                },
            )

        meta = _build_metadata_from_assembled(assembled)
        return {
            "answer": gen_result.get("answer", ""),
            "chunks": meta["chunks"],
            "file_paths": meta["file_paths"],
            "line_numbers": meta["line_numbers"],
            "github_links": meta["github_links"],
            "relevance_scores": meta["relevance_scores"],
        }

    except Exception as exc:
        logger.exception("query endpoint failed")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Query failed: {exc}"},
        )


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """
    Run search → rerank → assemble_context, then stream metadata line + answer tokens.

    First line is JSON with chunks, file_paths, line_numbers, github_links, relevance_scores.
    Remaining content is raw text: __STATUS__ tokens and answer tokens (or __ERROR__ on failure).

    Args:
        request: Body with query string.

    Returns:
        StreamingResponse with text/plain; or 400/500 JSON on error before streaming starts.
    """
    try:
        sanitized = _sanitize_query(request.query or "")
        if not sanitized.strip():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Empty query after sanitization"},
            )

        if _is_out_of_scope(sanitized):
            meta = _build_metadata_from_assembled([])
            return StreamingResponse(
                iter([json.dumps(meta) + "\n", _OUT_OF_SCOPE_TEMPLATE]),
                media_type="text/plain; charset=utf-8",
            )

        search_result = search(sanitized, top_k=TOP_K)
        if not search_result.get("success"):
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": search_result.get("error", "Search failed")},
            )

        results = (search_result.get("data") or {}).get("results") or []
        reranked = rerank(results, sanitized)
        repo_root = os.getenv("REPO_PATH", _DEFAULT_REPO_ROOT)
        assembled = assemble_context(reranked, repo_root=repo_root)

        return StreamingResponse(
            _stream_query_response(sanitized, assembled),
            media_type="text/plain; charset=utf-8",
        )

    except Exception as exc:
        logger.exception("query/stream endpoint failed")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Stream failed: {exc}"},
        )


# ---------------------------------------------------------------------------
# Admin: ingestion endpoint (one-time use to populate Railway ChromaDB volume)
# ---------------------------------------------------------------------------

_ingest_state: Dict[str, Any] = {"running": False, "last_result": None}


def _run_ingestion_background(clone_dest: str) -> None:
    """
    Clone gnucobol-contrib and run the full ingestion pipeline in a background thread.
    Writes progress to Railway logs. Sets REPO_PATH so the pipeline finds the data.

    Args:
        clone_dest: Absolute path where the repo will be cloned.
    """
    _ingest_state["running"] = True
    _ingest_state["last_result"] = None
    try:
        repo_owner = os.getenv("REPO_OWNER", "shreelakshmig7")
        repo_name = os.getenv("REPO_NAME", "gnucobol-contrib")
        clone_url = f"https://github.com/{repo_owner}/{repo_name}.git"

        if not Path(clone_dest).exists():
            logger.info("[ingest] Cloning %s → %s", clone_url, clone_dest)
            subprocess.run(
                ["git", "clone", "--depth=1", clone_url, clone_dest],
                check=True,
                timeout=300,
            )
            logger.info("[ingest] Clone complete.")
        else:
            logger.info("[ingest] Repo already present at %s — skipping clone.", clone_dest)

        os.environ["REPO_PATH"] = clone_dest
        logger.info("[ingest] REPO_PATH set to %s. Starting ingestion pipeline...", clone_dest)

        from legacylens.ingestion.runner import run_ingestion
        result = run_ingestion(clone_dest)
        _ingest_state["last_result"] = result
        if result.get("success"):
            data = result.get("data") or {}
            logger.info(
                "[ingest] SUCCESS — %d chunks stored, %d files, %.1fs total.",
                data.get("chunks_embedded", 0),
                data.get("files_discovered", 0),
                (data.get("timing_path") and 0) or 0,
            )
        else:
            logger.error("[ingest] FAILED — %s", result.get("error"))
    except subprocess.CalledProcessError as exc:
        logger.exception("[ingest] git clone failed: %s", exc)
        _ingest_state["last_result"] = {"success": False, "error": f"Clone failed: {exc}"}
    except Exception as exc:
        logger.exception("[ingest] Ingestion failed: %s", exc)
        _ingest_state["last_result"] = {"success": False, "error": str(exc)}
    finally:
        _ingest_state["running"] = False


@app.post("/admin/ingest")
def admin_ingest(request: Request) -> JSONResponse:
    """
    Trigger ingestion pipeline on Railway to populate the ChromaDB volume.

    Protected by INGEST_SECRET env var. Pass token as: Authorization: Bearer <token>.
    Runs in background thread — returns immediately; check Railway logs for progress.
    Call GET /admin/ingest/status to poll completion.

    Args:
        request: FastAPI Request (reads Authorization header).

    Returns:
        202 {"status": "started"} — ingestion launched in background.
        200 {"status": "already_running"} — previous ingestion still in progress.
        401 — missing or wrong secret.
        500 — unexpected error.
    """
    secret = os.getenv("INGEST_SECRET", "")
    auth_header = request.headers.get("Authorization", "")
    if not secret or auth_header != f"Bearer {secret}":
        return JSONResponse(status_code=401, content={"error": "Unauthorized — set INGEST_SECRET in Railway Variables."})

    if _ingest_state["running"]:
        return JSONResponse(
            status_code=200,
            content={"status": "already_running", "message": "Ingestion is still in progress. Check Railway logs."},
        )

    clone_dest = "/app/data/gnucobol-contrib"
    thread = threading.Thread(target=_run_ingestion_background, args=(clone_dest,), daemon=True)
    thread.start()
    logger.info("[ingest] Background ingestion thread started. Clone dest: %s", clone_dest)

    return JSONResponse(
        status_code=202,
        content={
            "status": "started",
            "message": "Ingestion running in background. Check Railway deploy logs for progress (~5-10 min).",
            "clone_dest": clone_dest,
        },
    )


@app.get("/admin/ingest/status")
def admin_ingest_status(request: Request) -> JSONResponse:
    """
    Poll ingestion status. Same auth as POST /admin/ingest.

    Returns:
        200 with running=True/False and last_result summary.
    """
    secret = os.getenv("INGEST_SECRET", "")
    auth_header = request.headers.get("Authorization", "")
    if not secret or auth_header != f"Bearer {secret}":
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    last = _ingest_state.get("last_result")
    return JSONResponse(
        status_code=200,
        content={
            "running": _ingest_state["running"],
            "last_result_success": last.get("success") if last else None,
            "last_result_error": last.get("error") if last else None,
        },
    )
