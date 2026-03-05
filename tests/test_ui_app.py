"""
test_ui_app.py
--------------
LegacyLens — RAG System for Legacy Enterprise Codebases — UI helper tests
-------------------------------------------------------------------------
Unit tests for small pure helpers in legacylens/ui/app.py used to control
rendering behavior in Streamlit. These tests focus on preventing stale
retrieved snippets from appearing when the latest response has no chunks.
Includes PRD 9.3 drill-down tests for _render_full_file_with_highlight and
_fetch_file_content.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from legacylens.ui.app import (
    _fetch_file_content,
    _has_retrieved_chunks,
    _infer_code_language,
    _render_full_file_with_highlight,
)


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


class TestRenderFullFileWithHighlight(unittest.TestCase):
    """Tests for _render_full_file_with_highlight (PRD 9.3: full file view with highlighted lines)."""

    def test_render_full_file_includes_line_numbers(self) -> None:
        """HTML must contain line number cells for each line."""
        content = "line1\nline2\nline3"
        html = _render_full_file_with_highlight(content, 1, 2)
        self.assertIn('<td class="ln">1</td>', html)
        self.assertIn('<td class="ln">2</td>', html)
        self.assertIn('<td class="ln">3</td>', html)

    def test_render_full_file_highlights_correct_lines(self) -> None:
        """Lines in [highlight_start, highlight_end] must have highlight class."""
        content = "a\nb\nc\nd"
        html = _render_full_file_with_highlight(content, 2, 3)
        self.assertIn('class="highlight"', html)
        self.assertIn("b", html)
        self.assertIn("c", html)
        # Line 2 and 3 should be highlighted
        self.assertIn('<td class="ln">2</td><td class="highlight">', html)
        self.assertIn('<td class="ln">3</td><td class="highlight">', html)

    def test_render_full_file_single_line_highlight(self) -> None:
        """highlight_start == highlight_end highlights exactly one line."""
        content = "x\ny\nz"
        html = _render_full_file_with_highlight(content, 2, 2)
        count = html.count('class="highlight"')
        self.assertEqual(count, 1, "Exactly one line should be highlighted")

    def test_render_full_file_empty_content(self) -> None:
        """Empty content produces valid HTML without crashing."""
        html = _render_full_file_with_highlight("", 1, 1)
        self.assertIn("file-view", html)
        self.assertIn("<table>", html)


class TestFetchFileContent(unittest.TestCase):
    """Tests for _fetch_file_content (PRD 9.3: fetch full file for drill-down)."""

    def test_fetch_file_content_empty_path(self) -> None:
        """Empty path returns (False, '', 'No file path')."""
        ok, content, err = _fetch_file_content("http://localhost:8000", "")
        self.assertFalse(ok)
        self.assertEqual(content, "")
        self.assertIn("No file path", err or "")

    def test_fetch_file_content_whitespace_path(self) -> None:
        """Whitespace-only path returns failure."""
        ok, content, err = _fetch_file_content("http://localhost:8000", "   ")
        self.assertFalse(ok)
        self.assertEqual(content, "")

    @patch("legacylens.ui.app.httpx")
    def test_fetch_file_content_success(self, mock_httpx: MagicMock) -> None:
        """Mock 200 with content returns (True, content, None)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "content": "IDENTIFICATION DIVISION.",
            "path": "data/gnucobol-contrib/sample.cbl",
        }
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_resp
        mock_httpx.Client.return_value.__enter__.return_value = mock_client_instance
        mock_httpx.Client.return_value.__exit__.return_value = False

        ok, content, err = _fetch_file_content("http://localhost:8000", "data/gnucobol-contrib/sample.cbl")
        self.assertTrue(ok)
        self.assertEqual(content, "IDENTIFICATION DIVISION.")
        self.assertIsNone(err)

    @patch("legacylens.ui.app.httpx")
    def test_fetch_file_content_http_error(self, mock_httpx: MagicMock) -> None:
        """Mock 404 returns (False, '', error message)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_resp
        mock_httpx.Client.return_value.__enter__.return_value = mock_client_instance
        mock_httpx.Client.return_value.__exit__.return_value = False

        ok, content, err = _fetch_file_content("http://localhost:8000", "data/missing.cbl")
        self.assertFalse(ok)
        self.assertEqual(content, "")
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()
