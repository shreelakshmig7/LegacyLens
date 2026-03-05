"""
test_api.py
-----------
LegacyLens — RAG System for Legacy Enterprise Codebases — Unit tests for FastAPI api/main.py
-------------------------------------------------------------------------------------------
TDD tests for legacylens/api/main.py. Covers POST /query, POST /query/stream, out-of-scope
short-circuit, streaming error token format, and REPO_PATH default.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Synthetic assembled result shape (matches context_assembler + reranker output)
# ---------------------------------------------------------------------------

def _make_assembled_result(
    file_path: str = "/repo/samples/file.cbl",
    line_range: str = "[10, 20]",
    score: float = 0.85,
    text: str = "       MOVE X TO Y.",
) -> Dict[str, Any]:
    """One assembled result dict as returned by assemble_context()."""
    return {
        "text": text,
        "metadata": {
            "file_path": file_path,
            "line_range": line_range,
            "type": "PROCEDURE",
            "parent_section": "PROCEDURE DIVISION",
            "paragraph_name": "MAIN-PARA",
            "dependencies": "",
        },
        "score": score,
        "assembled_context": text,
    }


def _make_search_result(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """search() return shape."""
    return {"success": True, "data": {"results": results}, "error": None}


# ---------------------------------------------------------------------------
# Import and app dependency
# ---------------------------------------------------------------------------

def _get_app():
    """Import app from api.main (used under patch so we test the real routes)."""
    from legacylens.api.main import app
    return app


# ---------------------------------------------------------------------------
# POST /query — success and structured body
# ---------------------------------------------------------------------------

class TestApiQueryEndpoint(unittest.TestCase):
    """POST /query returns 200 and structured JSON with answer, chunks, file_paths, etc."""

    @patch("legacylens.api.main.generate_answer")
    @patch("legacylens.api.main.assemble_context")
    @patch("legacylens.api.main.rerank")
    @patch("legacylens.api.main.search")
    @patch("legacylens.api.main._is_out_of_scope")
    @patch("legacylens.api.main._sanitize_query")
    @patch("legacylens.api.main.detect_feature_type", return_value="general")
    def test_api_query_returns_200_and_structured_body(
        self,
        mock_detect: MagicMock,
        mock_sanitize: MagicMock,
        mock_ooo: MagicMock,
        mock_search: MagicMock,
        mock_rerank: MagicMock,
        mock_assemble: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """POST /query returns 200 and body has answer, chunks, file_paths, line_numbers, github_links, relevance_scores."""
        mock_sanitize.return_value = "what does MAIN do"
        mock_ooo.return_value = False
        one = _make_assembled_result(file_path="/repo/a.cbl", line_range="[1, 5]", score=0.9)
        mock_search.return_value = _make_search_result([one])
        mock_rerank.return_value = [one]
        mock_assemble.return_value = [one]
        mock_generate.return_value = {"success": True, "answer": "It does X.", "error": None}

        client = TestClient(_get_app())
        response = client.post("/query", json={"query": "what does MAIN do"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("chunks", data)
        self.assertIn("file_paths", data)
        self.assertIn("line_numbers", data)
        self.assertIn("github_links", data)
        self.assertIn("relevance_scores", data)
        self.assertEqual(data["answer"], "It does X.")
        self.assertIsInstance(data["chunks"], list)
        self.assertIsInstance(data["file_paths"], list)
        self.assertIsInstance(data["line_numbers"], list)
        self.assertIsInstance(data["github_links"], list)
        self.assertIsInstance(data["relevance_scores"], list)
        self.assertEqual(len(data["chunks"]), 1)
        self.assertEqual(len(data["file_paths"]), 1)
        self.assertEqual(len(data["line_numbers"]), 1)
        self.assertEqual(len(data["github_links"]), 1)
        self.assertEqual(len(data["relevance_scores"]), 1)

    @patch("legacylens.api.main._sanitize_query")
    def test_api_query_empty_after_sanitize_returns_400(self, mock_sanitize: MagicMock) -> None:
        """POST /query with empty or sanitized-to-empty query returns 400."""
        mock_sanitize.return_value = ""

        client = TestClient(_get_app())
        response = client.post("/query", json={"query": "  "})

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    @patch("legacylens.api.main.generate_answer")
    @patch("legacylens.api.main.assemble_context")
    @patch("legacylens.api.main.rerank")
    @patch("legacylens.api.main.search")
    @patch("legacylens.api.main._is_out_of_scope")
    @patch("legacylens.api.main._sanitize_query")
    def test_api_query_search_failure_returns_500(
        self,
        mock_sanitize: MagicMock,
        mock_ooo: MagicMock,
        mock_search: MagicMock,
        mock_rerank: MagicMock,
        mock_assemble: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """POST /query when search fails returns 500 and error body."""
        mock_sanitize.return_value = "some query"
        mock_ooo.return_value = False
        mock_search.return_value = {"success": False, "data": None, "error": "ChromaDB unavailable"}

        client = TestClient(_get_app())
        response = client.post("/query", json={"query": "some query"})

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("error", data)
        mock_generate.assert_not_called()

    @patch("legacylens.api.main.search")
    @patch("legacylens.api.main._is_out_of_scope")
    @patch("legacylens.api.main._sanitize_query")
    def test_api_out_of_scope_query_returns_without_search(
        self,
        mock_sanitize: MagicMock,
        mock_ooo: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """When query is out-of-scope, API returns structured response without calling search."""
        mock_sanitize.return_value = "tell me a joke"
        mock_ooo.return_value = True

        client = TestClient(_get_app())
        response = client.post("/query", json={"query": "tell me a joke"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        mock_search.assert_not_called()


# ---------------------------------------------------------------------------
# POST /query/stream — metadata line then tokens; error token format
# ---------------------------------------------------------------------------

class TestApiQueryStreamEndpoint(unittest.TestCase):
    """POST /query/stream returns streaming response; first line JSON metadata; errors use __ERROR__."""

    @patch("legacylens.api.main.generate_answer_stream")
    @patch("legacylens.api.main.assemble_context")
    @patch("legacylens.api.main.rerank")
    @patch("legacylens.api.main.search")
    @patch("legacylens.api.main._is_out_of_scope")
    @patch("legacylens.api.main._sanitize_query")
    def test_api_stream_first_line_is_json_metadata(
        self,
        mock_sanitize: MagicMock,
        mock_ooo: MagicMock,
        mock_search: MagicMock,
        mock_rerank: MagicMock,
        mock_assemble: MagicMock,
        mock_stream: MagicMock,
    ) -> None:
        """POST /query/stream returns 200; first line is JSON with chunks, file_paths, line_numbers, github_links, relevance_scores."""
        mock_sanitize.return_value = "query"
        mock_ooo.return_value = False
        one = _make_assembled_result()
        mock_search.return_value = _make_search_result([one])
        mock_rerank.return_value = [one]
        mock_assemble.return_value = [one]
        mock_stream.return_value = iter(["__STATUS__Generating...", "Answer ", "text."])

        client = TestClient(_get_app())
        response = client.post("/query/stream", json={"query": "query"})

        self.assertEqual(response.status_code, 200)
        lines = response.text.strip().split("\n")
        self.assertGreaterEqual(len(lines), 1)
        first = json.loads(lines[0])
        self.assertIn("chunks", first)
        self.assertIn("file_paths", first)
        self.assertIn("line_numbers", first)
        self.assertIn("github_links", first)
        self.assertIn("relevance_scores", first)

    @patch("legacylens.api.main.generate_answer_stream")
    @patch("legacylens.api.main.assemble_context")
    @patch("legacylens.api.main.rerank")
    @patch("legacylens.api.main.search")
    @patch("legacylens.api.main._is_out_of_scope")
    @patch("legacylens.api.main._sanitize_query")
    def test_api_stream_error_token_format(
        self,
        mock_sanitize: MagicMock,
        mock_ooo: MagicMock,
        mock_search: MagicMock,
        mock_rerank: MagicMock,
        mock_assemble: MagicMock,
        mock_stream: MagicMock,
    ) -> None:
        """Streamed errors use __ERROR__<message> prefix so UI can detect error state."""
        mock_sanitize.return_value = "query"
        mock_ooo.return_value = False
        one = _make_assembled_result()
        mock_search.return_value = _make_search_result([one])
        mock_rerank.return_value = [one]
        mock_assemble.return_value = [one]
        mock_stream.return_value = iter(["__ERROR__Answer generation failed: timeout"])

        client = TestClient(_get_app())
        response = client.post("/query/stream", json={"query": "query"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("__ERROR__", response.text)

    @patch("legacylens.api.main._sanitize_query")
    def test_api_stream_empty_query_returns_400(self, mock_sanitize: MagicMock) -> None:
        """POST /query/stream with empty query after sanitize returns 400."""
        mock_sanitize.return_value = ""

        client = TestClient(_get_app())
        response = client.post("/query/stream", json={"query": ""})

        self.assertEqual(response.status_code, 400)


# ---------------------------------------------------------------------------
# REPO_PATH default
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GET /file/content — PRD 9.3 drill-down (full file view)
# ---------------------------------------------------------------------------

class TestApiFileContentEndpoint(unittest.TestCase):
    """GET /file/content validates path and returns file content or error."""

    def test_file_content_rejects_empty_path(self) -> None:
        """GET /file/content with empty path returns 400."""
        client = TestClient(_get_app())
        response = client.get("/file/content")
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_file_content_rejects_path_traversal(self) -> None:
        """GET /file/content rejects path with .. (directory traversal)."""
        client = TestClient(_get_app())
        response = client.get("/file/content?path=../../../etc/passwd")
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("traversal", data["error"].lower() or "invalid" in data["error"].lower())

    def test_file_content_rejects_disallowed_extension(self) -> None:
        """GET /file/content rejects file types not in allowlist."""
        client = TestClient(_get_app())
        response = client.get("/file/content?path=data/gnucobol-contrib/README.md")
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_file_content_returns_content_for_valid_path(self) -> None:
        """GET /file/content returns success and content for valid COBOL file."""
        import pathlib
        project_root = pathlib.Path(__file__).resolve().parent.parent
        # Use a known file from the repo
        sample = project_root / "data" / "gnucobol-contrib" / "samples" / "games" / "star_trek" / "ctrek.cob"
        if not sample.exists():
            self.skipTest("Sample file ctrek.cob not found (data not cloned)")
        rel_path = str(sample.relative_to(project_root))
        client = TestClient(_get_app())
        response = client.get(f"/file/content?path={rel_path}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("content", data)
        self.assertIn("path", data)
        self.assertIsInstance(data["content"], str)
        self.assertGreater(len(data["content"]), 0)

    def test_file_content_resolves_via_repo_path_outside_project_root(self) -> None:
        """GET /file/content succeeds when REPO_PATH is outside project root and path is under repo."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            subdir = repo_root / "samples" / "example"
            subdir.mkdir(parents=True)
            sample_file = subdir / "test.cbl"
            sample_file.write_text("       IDENTIFICATION DIVISION.\n       PROGRAM-ID. TEST.\n", encoding="utf-8")
            rel_path = "samples/example/test.cbl"
            with patch.dict(os.environ, {"REPO_PATH": str(repo_root)}, clear=False):
                client = TestClient(_get_app())
                response = client.get(f"/file/content?path={rel_path}")
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertTrue(data.get("success"), data)
            self.assertIn("content", data)
            self.assertIn("IDENTIFICATION DIVISION", data["content"])


class TestApiRepoPathDefault(unittest.TestCase):
    """assemble_context is called with repo_root default data/gnucobol-contrib when REPO_PATH unset."""

    @patch("legacylens.api.main.generate_answer")
    @patch("legacylens.api.main.assemble_context")
    @patch("legacylens.api.main.rerank")
    @patch("legacylens.api.main.search")
    @patch("legacylens.api.main._is_out_of_scope")
    @patch("legacylens.api.main._sanitize_query")
    def test_api_repo_path_default(
        self,
        mock_sanitize: MagicMock,
        mock_ooo: MagicMock,
        mock_search: MagicMock,
        mock_rerank: MagicMock,
        mock_assemble: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """When REPO_PATH is not set, assemble_context is called with repo_root='data/gnucobol-contrib'."""
        mock_sanitize.return_value = "query"
        mock_ooo.return_value = False
        one = _make_assembled_result()
        mock_search.return_value = _make_search_result([one])
        mock_rerank.return_value = [one]
        mock_assemble.return_value = [one]
        mock_generate.return_value = {"success": True, "answer": "Ok.", "error": None}

        with patch.dict(os.environ, {}, clear=False):
            if "REPO_PATH" in os.environ:
                del os.environ["REPO_PATH"]
            client = TestClient(_get_app())
            client.post("/query", json={"query": "query"})

        mock_assemble.assert_called_once()
        call_kwargs = mock_assemble.call_args
        self.assertEqual(call_kwargs[1]["repo_root"], "data/gnucobol-contrib")


if __name__ == "__main__":
    unittest.main()
