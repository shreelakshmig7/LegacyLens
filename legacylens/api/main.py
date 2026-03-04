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

import io
import json
import logging
import os
import threading
import zipfile
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
    _normalize_file_path,
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
        file_paths.append(_normalize_file_path(fp))
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
            logger.warning(
                "Answer generation failed, returning retrieval-only fallback: %s",
                gen_result.get("error", "Answer generation failed"),
            )
            fallback_answer = (
                "Answer generation is currently unavailable. "
                "Returning retrieved code evidence with file path and line number references."
            )
            meta = _build_metadata_from_assembled(assembled)
            return {
                "answer": fallback_answer,
                "chunks": meta["chunks"],
                "file_paths": meta["file_paths"],
                "line_numbers": meta["line_numbers"],
                "github_links": meta["github_links"],
                "relevance_scores": meta["relevance_scores"],
            }

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


def _download_and_extract_repo(repo_owner: str, repo_name: str, ref: str, dest: str) -> None:
    """
    Download a GitHub repo ZIP archive (no git binary required) and extract it to dest.

    Args:
        repo_owner: GitHub username.
        repo_name: Repository name.
        ref: Branch name or commit SHA.
        dest: Destination directory (will be created).
    """
    import httpx

    zip_url = f"https://github.com/{repo_owner}/{repo_name}/archive/{ref}.zip"
    logger.info("[ingest] Downloading %s ...", zip_url)
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        response = client.get(zip_url)
        response.raise_for_status()

    logger.info("[ingest] Download complete (%d bytes). Extracting...", len(response.content))
    parent = str(Path(dest).parent)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(parent)

    # GitHub names the extracted folder <repo>-<ref>/  — rename it to dest
    extracted_name = f"{repo_name}-{ref}"
    extracted_path = Path(parent) / extracted_name
    if extracted_path.exists() and not Path(dest).exists():
        extracted_path.rename(dest)
        logger.info("[ingest] Extracted to %s", dest)
    elif Path(dest).exists():
        logger.info("[ingest] Destination %s already exists — skipping rename.", dest)
    else:
        raise FileNotFoundError(f"Extracted path not found: {extracted_path}")


_FILE_BATCH_SIZE = 80  # Files per batch — keeps memory well below 1 GB Railway limit


def _already_ingested_paths() -> set:
    """
    Query ChromaDB for file_path metadata values already ingested.
    Returns a set of file paths so resume logic can skip completed files.

    Returns:
        set[str]: File paths already present in ChromaDB.
    """
    try:
        from legacylens.retrieval.vector_store import _get_collection
        collection = _get_collection()
        total = collection.count()
        if total == 0:
            return set()
        raw = collection.get(include=["metadatas"], limit=min(total, 200000))
        paths = {(m or {}).get("file_path", "") for m in (raw.get("metadatas") or []) if m}
        logger.info("[ingest] ChromaDB: %d chunks, %d unique file paths already ingested.", total, len(paths))
        return paths
    except Exception as exc:
        logger.warning("[ingest] Could not query existing paths (treating as empty): %s", exc)
        return set()


def _run_ingestion_background(clone_dest: str) -> None:
    """
    Download gnucobol-contrib ZIP (no git required), then ingest in batches of _FILE_BATCH_SIZE files.
    Each batch is chunked → embedded → inserted to ChromaDB immediately.
    Resume-safe: already-ingested file paths are detected from ChromaDB on startup,
    so restarting picks up where it left off without re-processing completed files.

    Args:
        clone_dest: Absolute path where the repo will be extracted.
    """
    _ingest_state["running"] = True
    _ingest_state["last_result"] = None
    try:
        repo_owner = os.getenv("REPO_OWNER", "shreelakshmig7")
        repo_name = os.getenv("REPO_NAME", "gnucobol-contrib")
        ref = os.getenv("REPO_BRANCH", "master")

        if not Path(clone_dest).exists():
            _download_and_extract_repo(repo_owner, repo_name, ref, clone_dest)
        else:
            logger.info("[ingest] Repo already at %s — skipping download.", clone_dest)

        os.environ["REPO_PATH"] = clone_dest

        from legacylens.ingestion.file_discovery import discover_files
        from legacylens.ingestion.chunker import chunk_file
        from legacylens.ingestion.embedder import embed_chunks
        from legacylens.ingestion.reference_scraper import attach_dependencies
        from legacylens.retrieval.vector_store import insert_chunks

        disc = discover_files(clone_dest)
        if not disc.get("success"):
            raise RuntimeError(f"File discovery failed: {disc.get('error')}")
        all_files = disc["data"]["files"]
        logger.info("[ingest] Discovered %d files.", len(all_files))

        ingested = _already_ingested_paths()
        remaining = [f for f in all_files if f not in ingested]
        logger.info("[ingest] To process: %d files (%d already done).", len(remaining), len(ingested))

        total_stored = 0
        n_batches = max(1, (len(remaining) + _FILE_BATCH_SIZE - 1) // _FILE_BATCH_SIZE)

        for idx in range(n_batches):
            batch = remaining[idx * _FILE_BATCH_SIZE:(idx + 1) * _FILE_BATCH_SIZE]
            logger.info("[ingest] Batch %d/%d — %d files...", idx + 1, n_batches, len(batch))

            chunks: list = []
            for fp in batch:
                r = chunk_file(fp)
                if r.get("success"):
                    chunks.extend(r["data"]["chunks"])

            if not chunks:
                logger.info("[ingest] Batch %d: no chunks — skipping.", idx + 1)
                continue

            dep = attach_dependencies(chunks)
            if dep.get("success"):
                chunks = dep["data"]["chunks"]

            emb = embed_chunks(chunks)
            if not emb.get("success"):
                logger.error("[ingest] Batch %d: embed failed — %s", idx + 1, emb.get("error"))
                continue
            embedded = emb["data"]["chunks"]

            ins = insert_chunks(embedded, clone_dest)
            if ins.get("success"):
                n = ins["data"]["inserted_count"]
                total_stored += n
                logger.info("[ingest] Batch %d/%d: +%d chunks (total: %d).", idx + 1, n_batches, n, total_stored)
            else:
                logger.error("[ingest] Batch %d: insert failed — %s", idx + 1, ins.get("error"))

        logger.info("[ingest] === COMPLETE — %d total chunks stored ===", total_stored)
        _ingest_state["last_result"] = {
            "success": True,
            "total_stored": total_stored,
            "files_discovered": len(all_files),
            "files_processed": len(remaining),
        }
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
