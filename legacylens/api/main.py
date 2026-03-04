"""
main.py
-------
LegacyLens — RAG System for Legacy Enterprise Codebases — FastAPI application
-----------------------------------------------------------------------------
Skeleton FastAPI app exposing the /query POST endpoint. Returns a placeholder
response only; no retrieval or generation logic.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

from fastapi import FastAPI
from pydantic import BaseModel

from legacylens.generation.answer_generator import _sanitize_query

app = FastAPI(
    title="LegacyLens API",
    description="Query interface for legacy codebase RAG.",
)


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = ""


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """
    Stub endpoint for natural-language queries. Sanitizes the query and returns
    a placeholder response. Full retrieval + generation wired in PR 5.

    Args:
        request: Body containing at least a query string.

    Returns:
        dict: Placeholder response with status, message, and sanitized query.
    """
    sanitized = _sanitize_query(request.query)
    return {"status": "ok", "message": "placeholder", "query": sanitized}
