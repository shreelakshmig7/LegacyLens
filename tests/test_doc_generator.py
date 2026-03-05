"""
test_doc_generator.py
---------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for Documentation Generator
---------------------------------------------------------------------------------------------
TDD tests for the Documentation Generation feature (PRD 7.4). Mocks the retrieval
pipeline and LLM to verify structured output, section completeness, and edge cases.

PRD 7.4 key requirements:
  - Must generate structured documentation for any COBOL module/paragraph
  - Output must include: summary, parameters, dependencies, side effects sections
  - Must cite file path and line number
  - Must reference only variables and logic present in context (no hallucination)

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

from unittest.mock import MagicMock, patch

import pytest

from legacylens.features.doc_generator import generate_documentation


_MOCK_SEARCH_RESULTS = [
    {
        "text": (
            "       PROCEDURE DIVISION.\n"
            "       MAIN-LOGIC.\n"
            "           PERFORM INIT-DB\n"
            "           PERFORM PROCESS-RECORDS\n"
            "           PERFORM CLEANUP\n"
            "           STOP RUN.\n"
        ),
        "metadata": {
            "file_path": "/repo/data/gnucobol-contrib/samples/DBsample/PostgreSQL/example1/PGMOD1.cbl",
            "file_name": "PGMOD1",
            "line_range": "[10, 30]",
            "type": "PROCEDURE",
            "parent_section": "MAIN-LOGIC",
            "paragraph_name": "MAIN-LOGIC",
            "dependencies": "PGMOD2,STATETXT",
        },
        "score": 0.88,
    },
    {
        "text": (
            "       INIT-DB.\n"
            "           CALL 'PGMOD2' USING WS-DB-CONN\n"
            "           IF WS-DB-STATUS = 'OK'\n"
            "               DISPLAY 'Database connected'\n"
            "           END-IF.\n"
        ),
        "metadata": {
            "file_path": "/repo/data/gnucobol-contrib/samples/DBsample/PostgreSQL/example1/PGMOD1.cbl",
            "file_name": "PGMOD1",
            "line_range": "[35, 50]",
            "type": "PROCEDURE",
            "parent_section": "MAIN-LOGIC",
            "paragraph_name": "INIT-DB",
            "dependencies": "PGMOD2",
        },
        "score": 0.82,
    },
]

_MOCK_LLM_RESPONSE = (
    "## PGMOD1 — Module Documentation\n\n"
    "### Summary\n"
    "PGMOD1 is the main entry point for PostgreSQL database operations. "
    "It initializes the database connection, processes records, and performs cleanup.\n\n"
    "### Parameters\n"
    "- WS-DB-CONN: Database connection handle passed via USING clause to PGMOD2.\n"
    "- WS-DB-STATUS: Status flag checked after database initialization.\n\n"
    "### Dependencies\n"
    "- PGMOD2: Called via CALL statement for database initialization.\n"
    "- STATETXT: Referenced as a dependency in module metadata.\n\n"
    "### Side Effects\n"
    "- Establishes a database connection via PGMOD2.\n"
    "- Displays 'Database connected' to the console on success.\n\n"
    "The file path is data/gnucobol-contrib/samples/DBsample/PostgreSQL/example1/PGMOD1.cbl "
    "and the line number is 10."
)


class TestGenerateDocumentationStructure:
    """generate_documentation() returns a properly structured dict."""

    @patch("legacylens.features.doc_generator._call_doc_generator_llm")
    @patch("legacylens.features.doc_generator.assemble_context")
    @patch("legacylens.features.doc_generator.rerank")
    @patch("legacylens.features.doc_generator.search")
    def test_returns_structured_dict(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """Verify return has keys success, documentation, sources."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_SEARCH_RESULTS},
        }
        mock_rerank.return_value = _MOCK_SEARCH_RESULTS
        mock_assemble.return_value = _MOCK_SEARCH_RESULTS
        mock_llm.return_value = _MOCK_LLM_RESPONSE

        result = generate_documentation("Generate documentation for PGMOD1")

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "documentation" in result
        assert "sources" in result

    @patch("legacylens.features.doc_generator._call_doc_generator_llm")
    @patch("legacylens.features.doc_generator.assemble_context")
    @patch("legacylens.features.doc_generator.rerank")
    @patch("legacylens.features.doc_generator.search")
    def test_documentation_is_non_empty_string(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """documentation should be a non-empty string from the LLM."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_SEARCH_RESULTS},
        }
        mock_rerank.return_value = _MOCK_SEARCH_RESULTS
        mock_assemble.return_value = _MOCK_SEARCH_RESULTS
        mock_llm.return_value = _MOCK_LLM_RESPONSE

        result = generate_documentation("Document the INIT-DB paragraph")

        assert isinstance(result["documentation"], str)
        assert len(result["documentation"]) > 0

    @patch("legacylens.features.doc_generator._call_doc_generator_llm")
    @patch("legacylens.features.doc_generator.assemble_context")
    @patch("legacylens.features.doc_generator.rerank")
    @patch("legacylens.features.doc_generator.search")
    def test_sources_contain_file_path_and_line(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """Each source entry must have file_path, line_number, paragraph_name."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_SEARCH_RESULTS},
        }
        mock_rerank.return_value = _MOCK_SEARCH_RESULTS
        mock_assemble.return_value = _MOCK_SEARCH_RESULTS
        mock_llm.return_value = _MOCK_LLM_RESPONSE

        result = generate_documentation("Create documentation for PGMOD1")

        assert len(result["sources"]) > 0
        for src in result["sources"]:
            assert "file_path" in src
            assert "line_number" in src
            assert "paragraph_name" in src


class TestGenerateDocumentationContent:
    """Verify the LLM receives proper context and is prompted for structured docs."""

    @patch("legacylens.features.doc_generator._call_doc_generator_llm")
    @patch("legacylens.features.doc_generator.assemble_context")
    @patch("legacylens.features.doc_generator.rerank")
    @patch("legacylens.features.doc_generator.search")
    def test_llm_receives_system_prompt_and_context(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """The LLM call should include a system prompt and user context."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_SEARCH_RESULTS},
        }
        mock_rerank.return_value = _MOCK_SEARCH_RESULTS
        mock_assemble.return_value = _MOCK_SEARCH_RESULTS
        mock_llm.return_value = _MOCK_LLM_RESPONSE

        generate_documentation("Auto-document PGMOD1")

        mock_llm.assert_called_once()
        messages = mock_llm.call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "PGMOD1" in messages[1]["content"] or "CONTEXT" in messages[1]["content"]


class TestGenerateDocumentationEdgeCases:
    """Edge cases: empty query, no results, LLM failure, search failure."""

    def test_empty_query_returns_error(self) -> None:
        """Empty query returns success=False."""
        result = generate_documentation("")
        assert result["success"] is False
        assert "error" in result

    def test_none_query_returns_error(self) -> None:
        """None query returns success=False."""
        result = generate_documentation(None)
        assert result["success"] is False
        assert "error" in result

    @patch("legacylens.features.doc_generator.assemble_context")
    @patch("legacylens.features.doc_generator.rerank")
    @patch("legacylens.features.doc_generator.search")
    def test_no_results_returns_not_found(
        self, mock_search, mock_rerank, mock_assemble
    ) -> None:
        """When search returns no results, return a not-found message."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": []},
        }

        result = generate_documentation("Generate documentation for NONEXISTENT")

        assert result["success"] is True
        assert "not found" in result["documentation"].lower()

    @patch("legacylens.features.doc_generator._call_doc_generator_llm")
    @patch("legacylens.features.doc_generator.assemble_context")
    @patch("legacylens.features.doc_generator.rerank")
    @patch("legacylens.features.doc_generator.search")
    def test_llm_failure_returns_fallback(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """When LLM fails, a fallback message is returned (not an exception)."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_SEARCH_RESULTS},
        }
        mock_rerank.return_value = _MOCK_SEARCH_RESULTS
        mock_assemble.return_value = _MOCK_SEARCH_RESULTS
        mock_llm.side_effect = Exception("LLM unavailable")

        result = generate_documentation("Generate documentation for PGMOD1")

        assert result["success"] is True
        assert "unavailable" in result["documentation"].lower()

    @patch("legacylens.features.doc_generator.search")
    def test_search_failure_returns_error(self, mock_search) -> None:
        """When search itself fails, return success=False with error."""
        mock_search.return_value = {
            "success": False,
            "data": None,
            "error": "Search service down",
        }

        result = generate_documentation("Document the PGMOD1 module")

        assert result["success"] is False
        assert "error" in result
