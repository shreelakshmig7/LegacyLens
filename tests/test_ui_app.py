"""
test_ui_app.py
--------------
LegacyLens — RAG System for Legacy Enterprise Codebases — UI helper tests
-------------------------------------------------------------------------
Unit tests for small pure helpers in legacylens/ui/app.py used to control
rendering behavior in Streamlit. These tests focus on preventing stale
retrieved snippets from appearing when the latest response has no chunks.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import unittest
from typing import Any, Dict, Optional

from legacylens.ui.app import _has_retrieved_chunks, _infer_code_language


class TestInferCodeLanguage(unittest.TestCase):
    """Tests for _infer_code_language (PRD 9.2: COBOL or Fortran syntax highlighting)."""

    def test_cobol_extensions_return_cobol(self) -> None:
        """COBOL file extensions must return 'cobol'."""
        self.assertEqual(_infer_code_language("data/repo/sample.cbl"), "cobol")
        self.assertEqual(_infer_code_language("x.cob"), "cobol")
        self.assertEqual(_infer_code_language("copybook.cpy"), "cobol")

    def test_fortran_extensions_return_fortran(self) -> None:
        """Fortran file extensions must return 'fortran'."""
        self.assertEqual(_infer_code_language("math.f"), "fortran")
        self.assertEqual(_infer_code_language("solver.f90"), "fortran")
        self.assertEqual(_infer_code_language("legacy.for"), "fortran")

    def test_empty_or_none_returns_cobol(self) -> None:
        """Empty or None path defaults to cobol."""
        self.assertEqual(_infer_code_language(""), "cobol")
        self.assertEqual(_infer_code_language(None), "cobol")  # type: ignore[arg-type]


class TestHasRetrievedChunks(unittest.TestCase):
    """Tests for _has_retrieved_chunks metadata helper."""

    def test_none_metadata_returns_false(self) -> None:
        """None metadata should be treated as no retrieved chunks."""
        metadata: Optional[Dict[str, Any]] = None
        self.assertFalse(_has_retrieved_chunks(metadata))

    def test_empty_dict_returns_false(self) -> None:
        """Empty metadata dict should be treated as no retrieved chunks."""
        metadata: Dict[str, Any] = {}
        self.assertFalse(_has_retrieved_chunks(metadata))

    def test_empty_chunks_list_returns_false(self) -> None:
        """An empty chunks list should return False."""
        metadata: Dict[str, Any] = {"chunks": []}
        self.assertFalse(_has_retrieved_chunks(metadata))

    def test_whitespace_only_chunks_returns_false(self) -> None:
        """Whitespace-only chunk text should not count as retrieved content."""
        metadata: Dict[str, Any] = {"chunks": ["   ", "\n\t"]}
        self.assertFalse(_has_retrieved_chunks(metadata))

    def test_non_empty_chunk_returns_true(self) -> None:
        """A non-empty chunk should return True."""
        metadata: Dict[str, Any] = {"chunks": ["MOVE A TO B."]}
        self.assertTrue(_has_retrieved_chunks(metadata))


if __name__ == "__main__":
    unittest.main()
