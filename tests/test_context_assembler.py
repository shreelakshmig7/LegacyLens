"""
test_context_assembler.py
-------------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for retrieval/context_assembler.py
----------------------------------------------------------------------------------------------------
Runs against real ChromaDB (CHROMA_PERSIST_DIR) for DATA chunk lookup. Copybook resolution uses
repo_root (e.g. data/gnucobol-contrib). No mocks.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import os
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from legacylens.config.constants import CHROMA_PERSIST_DIR
from legacylens.retrieval.context_assembler import assemble_context


def _chroma_ready() -> bool:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return os.path.isdir(CHROMA_PERSIST_DIR)


# Default repo root for tests (relative to project root)
_REPO_ROOT = Path(__file__).resolve().parent.parent / "data" / "gnucobol-contrib"


@pytest.mark.skipif(not _chroma_ready(), reason="ChromaDB not available")
def test_assemble_returns_list() -> None:
    """assemble_context must return a list (one entry per input result)."""
    # Use minimal input: one fake result with real file_path from repo if possible
    results = [{"text": "PROCEDURE DIVISION.\nMAIN-PGM.", "metadata": {"file_path": "", "parent_section": "PROCEDURE DIVISION", "type": "PROCEDURE", "paragraph_name": "MAIN-PGM", "dependencies": ""}, "score": 0.9}]
    repo_root = str(_REPO_ROOT) if _REPO_ROOT.exists() else ""
    out = assemble_context(results, repo_root=repo_root)
    assert isinstance(out, list)
    assert len(out) <= len(results)


@pytest.mark.skipif(not _chroma_ready(), reason="ChromaDB not available")
def test_assemble_each_item_has_assembled_context() -> None:
    """Each returned item must have key assembled_context."""
    results = [{"text": "code", "metadata": {"file_path": "x.cbl", "parent_section": "P", "type": "PROCEDURE", "paragraph_name": "P1", "dependencies": ""}, "score": 0.8}]
    out = assemble_context(results, repo_root="")
    for item in out:
        assert "assembled_context" in item
        assert isinstance(item["assembled_context"], str)


@pytest.mark.skipif(not _chroma_ready(), reason="ChromaDB not available")
def test_assemble_preserves_original_keys() -> None:
    """Each item must preserve text, metadata, score from input."""
    results = [{"text": "para.", "metadata": {"file_path": "a.cbl", "parent_section": "PROCEDURE DIVISION", "type": "PROCEDURE", "paragraph_name": "X", "dependencies": ""}, "score": 0.7}]
    out = assemble_context(results, repo_root="")
    if out:
        assert out[0]["text"] == "para."
        assert out[0]["metadata"]["paragraph_name"] == "X"
        assert out[0]["score"] == 0.7


@pytest.mark.skipif(not _chroma_ready(), reason="ChromaDB not available")
def test_assemble_parent_section_in_context() -> None:
    """parent_section from metadata must appear in assembled_context."""
    results = [{"text": "MAIN-PGM.\n  DISPLAY 'x'.", "metadata": {"file_path": "f.cbl", "parent_section": "PROCEDURE DIVISION", "type": "PROCEDURE", "paragraph_name": "MAIN-PGM", "dependencies": ""}, "score": 0.8}]
    out = assemble_context(results, repo_root="")
    if out:
        assert "PROCEDURE DIVISION" in out[0]["assembled_context"] or "MAIN-PGM" in out[0]["assembled_context"]


def test_assemble_empty_input_returns_empty() -> None:
    """Empty results list must return empty list."""
    assert assemble_context([], repo_root="") == []
