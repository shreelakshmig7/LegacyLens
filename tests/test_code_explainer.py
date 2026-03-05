"""
test_code_explainer.py
----------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for Code Explainer
------------------------------------------------------------------------------------
TDD tests for the Code Explanation feature (PRD 7.1). Mocks the retrieval pipeline
and LLM calls to verify structured output, citation requirements, and edge cases.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

from unittest.mock import MagicMock, patch

import pytest

from legacylens.features.code_explainer import explain


_MOCK_ASSEMBLED = [
    {
        "text": "       PERFORM UPDATE-RECORD\n       MOVE WS-CUST-NAME TO CUST-NAME\n       REWRITE CUSTOMER-RECORD",
        "metadata": {
            "file_path": "/repo/data/gnucobol-contrib/samples/cust01.cbl",
            "file_name": "CUST01",
            "line_range": "[42, 55]",
            "type": "PROCEDURE",
            "parent_section": "MAIN-LOGIC",
            "paragraph_name": "UPDATE-RECORD",
            "dependencies": "CUSTOMER-FILE",
        },
        "score": 0.85,
        "assembled_context": (
            "[MAIN-LOGIC]\n"
            "       PERFORM UPDATE-RECORD\n"
            "       MOVE WS-CUST-NAME TO CUST-NAME\n"
            "       REWRITE CUSTOMER-RECORD"
        ),
    }
]


class TestExplainReturnsStructuredDict:
    """explain() must return a dict with success, explanation, sources."""

    @patch("legacylens.features.code_explainer._call_explainer_llm")
    @patch("legacylens.features.code_explainer.assemble_context")
    @patch("legacylens.features.code_explainer.rerank")
    @patch("legacylens.features.code_explainer.search")
    def test_explain_returns_structured_dict(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """Verify return has keys success, explanation, sources."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_ASSEMBLED},
        }
        mock_rerank.return_value = _MOCK_ASSEMBLED
        mock_assemble.return_value = _MOCK_ASSEMBLED
        mock_llm.return_value = (
            "The UPDATE-RECORD paragraph updates the customer record "
            "by moving WS-CUST-NAME to CUST-NAME and rewriting CUSTOMER-RECORD. "
            "The file path is data/gnucobol-contrib/samples/cust01.cbl and "
            "the line number is 42. This paragraph is in the MAIN-LOGIC section."
        )

        result = explain("Explain what the UPDATE-RECORD paragraph does")

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "explanation" in result
        assert "sources" in result
        assert len(result["explanation"]) > 0

    @patch("legacylens.features.code_explainer._call_explainer_llm")
    @patch("legacylens.features.code_explainer.assemble_context")
    @patch("legacylens.features.code_explainer.rerank")
    @patch("legacylens.features.code_explainer.search")
    def test_explain_includes_paragraph_name_and_line_ref(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """Verify output references paragraph name and file:line from metadata."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_ASSEMBLED},
        }
        mock_rerank.return_value = _MOCK_ASSEMBLED
        mock_assemble.return_value = _MOCK_ASSEMBLED
        mock_llm.return_value = (
            "The UPDATE-RECORD paragraph moves WS-CUST-NAME to CUST-NAME. "
            "File path: data/gnucobol-contrib/samples/cust01.cbl, line number 42."
        )

        result = explain("Explain what the UPDATE-RECORD paragraph does")

        assert result["success"] is True
        assert len(result["sources"]) > 0
        source = result["sources"][0]
        assert "file_path" in source
        assert "line_number" in source
        assert "paragraph_name" in source

    @patch("legacylens.features.code_explainer._call_explainer_llm")
    @patch("legacylens.features.code_explainer.assemble_context")
    @patch("legacylens.features.code_explainer.rerank")
    @patch("legacylens.features.code_explainer.search")
    def test_explain_references_variables_from_context(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """Verify explanation references specific variable names from context."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_ASSEMBLED},
        }
        mock_rerank.return_value = _MOCK_ASSEMBLED
        mock_assemble.return_value = _MOCK_ASSEMBLED
        mock_llm.return_value = (
            "The UPDATE-RECORD paragraph performs a record update by moving "
            "WS-CUST-NAME to CUST-NAME and then issuing a REWRITE on CUSTOMER-RECORD. "
            "File path: data/gnucobol-contrib/samples/cust01.cbl, line number 42."
        )

        result = explain("Explain what the UPDATE-RECORD paragraph does")

        assert result["success"] is True
        explanation = result["explanation"]
        assert "WS-CUST-NAME" in explanation or "CUSTOMER-RECORD" in explanation


class TestExplainEdgeCases:
    """Edge cases: empty query, no results, LLM failure."""

    def test_explain_empty_query_returns_error(self) -> None:
        """Empty/whitespace query returns success=False."""
        result = explain("")
        assert result["success"] is False
        assert "error" in result

    def test_explain_none_query_returns_error(self) -> None:
        """None query returns success=False."""
        result = explain(None)
        assert result["success"] is False
        assert "error" in result

    @patch("legacylens.features.code_explainer.search")
    def test_explain_no_results_returns_not_found(self, mock_search) -> None:
        """When search returns empty results, return structured not-found."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": []},
        }

        result = explain("Explain what the NONEXISTENT paragraph does")

        assert result["success"] is True
        assert "not found" in result["explanation"].lower()

    @patch("legacylens.features.code_explainer._call_explainer_llm")
    @patch("legacylens.features.code_explainer.assemble_context")
    @patch("legacylens.features.code_explainer.rerank")
    @patch("legacylens.features.code_explainer.search")
    def test_explain_llm_failure_returns_fallback(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """When LLM call raises, return fallback with retrieved chunks."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_ASSEMBLED},
        }
        mock_rerank.return_value = _MOCK_ASSEMBLED
        mock_assemble.return_value = _MOCK_ASSEMBLED
        mock_llm.side_effect = Exception("LLM API unavailable")

        result = explain("Explain what the UPDATE-RECORD paragraph does")

        assert result["success"] is True
        assert len(result["sources"]) > 0
        assert "unavailable" in result["explanation"].lower() or len(result["explanation"]) > 0

    @patch("legacylens.features.code_explainer.search")
    def test_explain_search_failure_returns_error(self, mock_search) -> None:
        """When search itself fails, return success=False."""
        mock_search.return_value = {
            "success": False,
            "data": None,
            "error": "ChromaDB connection failed",
        }

        result = explain("Explain what UPDATE-RECORD does")

        assert result["success"] is False
        assert "error" in result
