"""
main.py
-------
LegacyLens — RAG System for Legacy Enterprise Codebases — FastAPI application
-----------------------------------------------------------------------------
FastAPI app exposing POST /query and POST /query/stream.
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
from pathlib import Path
from typing import Any, Dict, List

# Load .env from project root so VOYAGE_API_KEY, OPENAI_API_KEY, etc. are set when running via uvicorn
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from legacylens.config.constants import (
    FEATURE_TYPE_BUSINESS_LOGIC,
    FEATURE_TYPE_DEPENDENCY,
    FEATURE_TYPE_DOC_GENERATE,
    FEATURE_TYPE_EXPLAIN,
    FEATURE_TYPE_GENERAL,
    FILE_CONTENT_ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_VIEW_LINES,
    TOP_K,
    validate_required_env_vars,
)
from legacylens.features import detect_feature_type
from legacylens.features.business_logic_extractor import extract_business_logic as _extract_biz
from legacylens.features.code_explainer import explain as _explain_code
from legacylens.features.dependency_mapper import map_dependencies as _map_deps
from legacylens.features.doc_generator import generate_documentation as _gen_docs
from legacylens.generation.answer_generator import (
    _OUT_OF_SCOPE_TEMPLATE,
    _build_github_link,
    _is_fast_path,
    _is_out_of_scope,
    _normalize_file_path,
    _parse_line_range,
    _parse_line_range_tuple,
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

# Project root: parent of legacylens package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_and_validate_file_path(relative_path: str) -> Dict[str, Any]:
    """
    Resolve relative path to absolute file; validate security and constraints.

    Rejects: path traversal (..), disallowed extensions, paths outside project,
    non-existent files, files exceeding MAX_FILE_SIZE_BYTES.

    Args:
        relative_path: Path relative to project root (e.g. data/gnucobol-contrib/.../file.cbl).

    Returns:
        dict: {"success": bool, "path": str | None, "error": str | None}
    """
    if not relative_path or not isinstance(relative_path, str):
        return {"success": False, "path": None, "error": "Path is required"}
    path_str = relative_path.strip()
    if ".." in path_str or path_str.startswith("/"):
        return {"success": False, "path": None, "error": "Invalid path: traversal or absolute path not allowed"}
    ext = Path(path_str).suffix.lower()
    if ext not in FILE_CONTENT_ALLOWED_EXTENSIONS:
        return {
            "success": False,
            "path": None,
            "error": f"File type not allowed. Allowed: {', '.join(FILE_CONTENT_ALLOWED_EXTENSIONS)}",
        }
    resolved = (_PROJECT_ROOT / path_str).resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT.resolve())
    except ValueError:
        return {"success": False, "path": None, "error": "Path resolves outside project root"}
    if not resolved.exists():
        repo_path = os.getenv("REPO_PATH", "").strip()
        if repo_path:
            alt = (Path(repo_path) / path_str).resolve()
            try:
                alt.relative_to(_PROJECT_ROOT.resolve())
            except ValueError:
                return {"success": False, "path": None, "error": "Path resolves outside project root"}
            if alt.exists() and alt.is_file():
                resolved = alt
        if not resolved.exists():
            return {"success": False, "path": None, "error": "File not found"}
    if not resolved.is_file():
        return {"success": False, "path": None, "error": "Path is not a file"}
    try:
        size = resolved.stat().st_size
    except OSError as e:
        return {"success": False, "path": None, "error": f"Cannot read file: {e}"}
    if size > MAX_FILE_SIZE_BYTES:
        return {
            "success": False,
            "path": None,
            "error": f"File too large (max {MAX_FILE_SIZE_BYTES} bytes)",
        }
    return {"success": True, "path": str(resolved), "error": None}


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


@app.get("/file/content")
def file_content(path: str = ""):
    """
    Return full file content for PRD 9.3 drill-down (full file view with highlighted lines).

    Path must be relative to project root, use allowed extensions, and pass security checks.

    Args:
        path: Query param — relative path (e.g. data/gnucobol-contrib/samples/.../file.cbl).

    Returns:
        200: {"success": True, "content": str, "path": str}
        400/500: {"success": False, "error": str}
    """
    result = _resolve_and_validate_file_path(path)
    if not result["success"]:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": result["error"]},
        )
    file_path = result["path"]
    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Failed to read file %s: %s", file_path, e)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Cannot read file: {e}"},
        )
    lines = content.splitlines()
    if len(lines) > MAX_FILE_VIEW_LINES:
        lines = lines[:MAX_FILE_VIEW_LINES]
        content = "\n".join(lines) + "\n\n... (truncated)"
    return {
        "success": True,
        "content": content,
        "path": path,
    }


class QueryRequest(BaseModel):
    """Request body for POST /query and POST /query/stream."""

    query: str = ""


def _build_metadata_from_assembled(
    assembled_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build chunks, file_paths, line_numbers, line_ranges, github_links, relevance_scores.

    line_ranges: [(start, end), ...] per chunk for PRD 9.3 full-file highlighting.

    Args:
        assembled_results: List of dicts from assemble_context(); each has text, metadata, score.

    Returns:
        dict: Keys chunks, file_paths, line_numbers, line_ranges, github_links, relevance_scores.
    """
    chunks: List[str] = []
    file_paths: List[str] = []
    line_numbers: List[int] = []
    line_ranges: List[List[int]] = []
    github_links: List[str] = []
    relevance_scores: List[float] = []

    for r in assembled_results:
        meta = r.get("metadata") or {}
        fp = (meta.get("file_path") or "").strip()
        lr = meta.get("line_range") or ""
        start, end = _parse_line_range_tuple(lr)
        line_num = _parse_line_range(lr)
        chunks.append((r.get("text") or "").strip())
        file_paths.append(_normalize_file_path(fp))
        line_numbers.append(line_num)
        line_ranges.append([start, end])
        github_links.append(_build_github_link(fp, str(lr)))
        relevance_scores.append(float(r.get("score", 0.0)))

    return {
        "chunks": chunks,
        "file_paths": file_paths,
        "line_numbers": line_numbers,
        "line_ranges": line_ranges,
        "github_links": github_links,
        "relevance_scores": relevance_scores,
    }


def _generate_with_feature_routing(
    feature_type: str,
    sanitized: str,
    assembled: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Dispatch answer generation to the appropriate feature module based on query type.

    Falls through to the default generate_answer() for "general" type and for any
    feature type whose module is not yet implemented.

    Args:
        feature_type: One of the FEATURE_TYPE_* constants from detect_feature_type().
        sanitized: The sanitized query string.
        assembled: Assembled context results from the retrieval pipeline.

    Returns:
        dict: {"success": bool, "answer": str, ...} — same schema as generate_answer().
    """
    if feature_type == FEATURE_TYPE_GENERAL:
        return generate_answer(sanitized, assembled)

    if feature_type == FEATURE_TYPE_EXPLAIN:
        logger.info("Feature routing: explain for query: %.60s", sanitized)
        result = _explain_code(sanitized)
        return {
            "success": result.get("success", False),
            "answer": result.get("explanation", ""),
        }

    if feature_type == FEATURE_TYPE_DEPENDENCY:
        logger.info("Feature routing: dependency for query: %.60s", sanitized)
        result = _map_deps(sanitized)
        answer_parts = [result.get("summary", "")]
        deps = result.get("dependencies")
        if deps:
            if deps.get("calls"):
                answer_parts.append(
                    "\n\nCALL targets: "
                    + ", ".join(f"{c['target']} ({c['dep_type']})" for c in deps["calls"])
                )
            if deps.get("copies"):
                answer_parts.append(
                    "\nCOPY targets: "
                    + ", ".join(f"{c['target']} ({c['dep_type']})" for c in deps["copies"])
                )
            if deps.get("usings"):
                answer_parts.append(
                    "\nUSING targets: "
                    + ", ".join(f"{c['target']} ({c['dep_type']})" for c in deps["usings"])
                )
        return {
            "success": result.get("success", False),
            "answer": "".join(answer_parts),
        }

    if feature_type == FEATURE_TYPE_BUSINESS_LOGIC:
        logger.info("Feature routing: business_logic for query: %.60s", sanitized)
        result = _extract_biz(sanitized)
        return {
            "success": result.get("success", False),
            "answer": result.get("business_rules", ""),
        }

    if feature_type == FEATURE_TYPE_DOC_GENERATE:
        logger.info("Feature routing: doc_generate for query: %.60s", sanitized)
        result = _gen_docs(sanitized)
        return {
            "success": result.get("success", False),
            "answer": result.get("documentation", ""),
        }

    logger.info("Feature routing: type=%s for query: %.60s", feature_type, sanitized)
    return generate_answer(sanitized, assembled)


def _stream_query_response(sanitized: str, assembled: List[Dict[str, Any]]):
    """Generator: one JSON line (metadata), then tokens from generate_answer_stream.

    Chunks are suppressed in metadata when the fast-path applies (all relevance
    scores below NOT_FOUND_SCORE_THRESHOLD), because those chunks are irrelevant
    and should not be surfaced to the user as "Retrieved chunks".
    """
    chunks_for_meta = [] if _is_fast_path(assembled) else assembled
    meta = _build_metadata_from_assembled(chunks_for_meta)
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
                "line_ranges": meta["line_ranges"],
                "github_links": meta["github_links"],
                "relevance_scores": meta["relevance_scores"],
            }

        feature_type = detect_feature_type(sanitized)

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

        gen_result = _generate_with_feature_routing(feature_type, sanitized, assembled)
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
                "line_ranges": meta["line_ranges"],
                "github_links": meta["github_links"],
                "relevance_scores": meta["relevance_scores"],
            }

        chunks_for_meta = [] if _is_fast_path(assembled) else assembled
        meta = _build_metadata_from_assembled(chunks_for_meta)
        return {
            "answer": gen_result.get("answer", ""),
            "chunks": meta["chunks"],
            "file_paths": meta["file_paths"],
            "line_numbers": meta["line_numbers"],
            "line_ranges": meta["line_ranges"],
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

