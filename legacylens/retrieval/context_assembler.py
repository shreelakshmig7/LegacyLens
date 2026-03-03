"""
context_assembler.py
--------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Assemble chunk + DATA xref + copybook
-----------------------------------------------------------------------------------------------
Builds full context per result: chunk text, parent section, variable definitions from same file's
DATA chunks in ChromaDB (only matching COBOL identifiers), and copybook content when COPY present.
Copybook resolution: same dir as file first, then recursive under repo_root. No magic numbers.

Key functions:
    assemble_context(results, repo_root) -> list

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from legacylens.config.constants import CHUNK_TYPE_DATA, DATA_XREF_MAX_CHUNKS
from legacylens.retrieval.vector_store import _get_collection

logger = logging.getLogger(__name__)

# COBOL identifier: ALL-CAPS with hyphens (variable/paragraph names)
_COBOL_IDENTIFIER_RE = re.compile(r"\b([A-Z][A-Z0-9\-]+)\b")


def _variable_names_in_text(text: str) -> Set[str]:
    """Extract COBOL-style identifiers from text."""
    if not text:
        return set()
    return set(m.group(1) for m in _COBOL_IDENTIFIER_RE.finditer(text))


def _get_data_chunks_for_file(file_path: str) -> List[Dict[str, Any]]:
    """Fetch DATA chunks for the given file from Chroma. Returns list of {document, metadata}."""
    if not file_path or not file_path.strip():
        return []
    try:
        collection = _get_collection()
        # Chroma where: equality filter (syntax may be field: value or field: {$eq: value})
        raw = collection.get(
            where={"$and": [{"file_path": {"$eq": file_path}}, {"type": {"$eq": CHUNK_TYPE_DATA}}]},
            include=["documents", "metadatas"],
            limit=DATA_XREF_MAX_CHUNKS,
        )
    except Exception as exc:
        logger.warning("Failed to get DATA chunks for %s: %s", file_path, exc)
        return []

    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    out = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        out.append({"document": doc, "metadata": meta})
    return out


def _data_xref_snippet(chunk_text: str, file_path: str, variables: Set[str]) -> str:
    """Return concatenated DATA chunk texts that define any of the given variables."""
    if not file_path or not variables:
        return ""
    data_chunks = _get_data_chunks_for_file(file_path)
    snippets = []
    for item in data_chunks:
        doc = item.get("document") or ""
        if not doc:
            continue
        doc_vars = _variable_names_in_text(doc)
        if doc_vars & variables:
            snippets.append(doc.strip())
    if not snippets:
        return ""
    return "\n\n--- DATA definitions ---\n" + "\n\n".join(snippets)


def _parse_dependencies(deps_str: str) -> List[str]:
    """Parse comma-joined dependencies into list of names (e.g. copybook names)."""
    if not deps_str or not isinstance(deps_str, str):
        return []
    return [n.strip() for n in deps_str.split(",") if n.strip()]


def _resolve_copybook(copybook_name: str, referencing_file_path: str, repo_root: str) -> Optional[Path]:
    """Resolve copybook path: same dir as referencing file first, then recursive under repo_root."""
    if not repo_root or not copybook_name:
        return None
    base = Path(repo_root)
    if not base.exists():
        return None
    # Clean copybook name (may have .cpy or not)
    name = copybook_name.strip()
    if not name.endswith(".cpy") and not name.endswith(".cob") and not name.endswith(".cbl"):
        name_cpy = name + ".cpy"
    else:
        name_cpy = name

    # Same directory as referencing file
    ref_path = Path(referencing_file_path)
    if ref_path.is_absolute() and str(ref_path).startswith(str(base)):
        try:
            same_dir = base / ref_path.relative_to(base).parent
        except ValueError:
            same_dir = base
    else:
        same_dir = base / ref_path.parent if ref_path.parent else base
    for candidate in (same_dir / name_cpy, same_dir / name):
        if candidate.exists() and candidate.is_file():
            return candidate

    # Recursive under repo_root
    for path in base.rglob(name_cpy):
        if path.is_file():
            return path
    for path in base.rglob(name):
        if path.is_file():
            return path
    return None


def _read_copybook(path: Path) -> str:
    """Read copybook file content with encoding fallback."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read copybook %s: %s", path, exc)
        return ""


def _copybook_snippet(dependencies: List[str], file_path: str, repo_root: str) -> str:
    """Resolve and read copybooks; return concatenated content or empty."""
    if not dependencies or not repo_root:
        return ""
    parts = []
    for dep in dependencies:
        resolved = _resolve_copybook(dep, file_path, repo_root)
        if resolved:
            content = _read_copybook(resolved)
            if content:
                parts.append(f"--- Copybook: {resolved.name} ---\n{content}")
        else:
            logger.warning("Copybook not found: %s (referencing file: %s)", dep, file_path)
    if not parts:
        return ""
    return "\n\n".join(parts)


def assemble_context(
    results: List[Dict[str, Any]],
    repo_root: str = "",
) -> List[Dict[str, Any]]:
    """
    Assemble full context for each result: chunk + parent section + DATA xref + copybook.

    Does not mutate input. Each returned dict has all original keys plus "assembled_context".

    Args:
        results: List of reranker output dicts (text, metadata, score).
        repo_root: Base path to repository for copybook resolution.

    Returns:
        list: New list of dicts with "assembled_context" added.
    """
    if not results:
        return []

    out = []
    for r in results:
        item = copy.deepcopy(r)
        text = item.get("text") or ""
        meta = item.get("metadata") or {}
        file_path = (meta.get("file_path") or "").strip()
        parent_section = (meta.get("parent_section") or "").strip()
        deps_str = meta.get("dependencies") or ""

        parts = []
        if parent_section:
            parts.append(f"[{parent_section}]\n")
        parts.append(text)

        variables = _variable_names_in_text(text)
        if variables and file_path:
            xref = _data_xref_snippet(text, file_path, variables)
            if xref:
                parts.append(xref)

        deps = _parse_dependencies(deps_str)
        if deps and repo_root:
            copybook = _copybook_snippet(deps, file_path, repo_root)
            if copybook:
                parts.append(copybook)

        item["assembled_context"] = "\n\n".join(parts)
        out.append(item)

    return out
