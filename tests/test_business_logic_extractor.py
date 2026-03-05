"""
test_business_logic_extractor.py
---------------------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for Business Logic Extractor
----------------------------------------------------------------------------------------------
TDD tests for the Business Logic Extraction feature (PRD 7.3). Mocks the retrieval
pipeline and LLM to verify structured output, rule extraction format, condition
surfacing, and edge case handling.

PRD 7.3 key requirements:
  - Must identify and extract business rules embedded in COBOL code
  - Must surface IF/EVALUATE conditions, thresholds, validation checks
  - Must explain business intent behind conditional logic
  - Must reference specific variables and conditions from context

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

from unittest.mock import MagicMock, patch

import pytest

from legacylens.features.business_logic_extractor import extract_business_logic


_MOCK_SEARCH_RESULTS = [
    {
        "text": (
            "       IF WS-CUST-STATUS = 'ACTIVE'\n"
            "           IF WS-BALANCE > 10000\n"
            "               PERFORM PREMIUM-DISCOUNT\n"
            "           ELSE\n"
            "               PERFORM STANDARD-RATE\n"
            "           END-IF\n"
            "       END-IF\n"
        ),
        "metadata": {
            "file_path": "/repo/data/gnucobol-contrib/samples/cust01.cbl",
            "file_name": "CUST01",
            "line_range": "[50, 65]",
            "type": "PROCEDURE",
            "parent_section": "PRICING-LOGIC",
            "paragraph_name": "DETERMINE-RATE",
            "dependencies": "",
        },
        "score": 0.85,
    },
    {
        "text": (
            "       EVALUATE TRUE\n"
            "           WHEN WS-ORDER-TYPE = 'RUSH'\n"
            "               MOVE 25.00 TO WS-SURCHARGE\n"
            "           WHEN WS-ORDER-TYPE = 'STANDARD'\n"
            "               MOVE 0.00 TO WS-SURCHARGE\n"
            "           WHEN OTHER\n"
            "               MOVE 10.00 TO WS-SURCHARGE\n"
            "       END-EVALUATE\n"
        ),
        "metadata": {
            "file_path": "/repo/data/gnucobol-contrib/samples/cust01.cbl",
            "file_name": "CUST01",
            "line_range": "[70, 85]",
            "type": "PROCEDURE",
            "parent_section": "ORDER-PROCESSING",
            "paragraph_name": "CALC-SURCHARGE",
            "dependencies": "",
        },
        "score": 0.80,
    },
]

_MOCK_LLM_RESPONSE = (
    "Business Rule 1: Premium discount eligibility — "
    "If WS-CUST-STATUS is 'ACTIVE' and WS-BALANCE exceeds 10000, the customer "
    "qualifies for the PREMIUM-DISCOUNT paragraph. Otherwise, STANDARD-RATE applies.\n\n"
    "Business Rule 2: Order surcharge calculation — "
    "RUSH orders incur a $25.00 surcharge (WS-SURCHARGE), STANDARD orders have no "
    "surcharge, and all other order types default to $10.00.\n\n"
    "The file path is data/gnucobol-contrib/samples/cust01.cbl and the line number is 50."
)


class TestExtractBusinessLogicStructure:
    """extract_business_logic() returns a properly structured dict."""

    @patch("legacylens.features.business_logic_extractor._call_business_logic_llm")
    @patch("legacylens.features.business_logic_extractor.assemble_context")
    @patch("legacylens.features.business_logic_extractor.rerank")
    @patch("legacylens.features.business_logic_extractor.search")
    def test_returns_structured_dict(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """Verify return has keys success, business_rules, sources."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_SEARCH_RESULTS},
        }
        mock_rerank.return_value = _MOCK_SEARCH_RESULTS
        mock_assemble.return_value = _MOCK_SEARCH_RESULTS
        mock_llm.return_value = _MOCK_LLM_RESPONSE

        result = extract_business_logic(
            "What are the business rules in the pricing logic?"
        )

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "business_rules" in result
        assert "sources" in result

    @patch("legacylens.features.business_logic_extractor._call_business_logic_llm")
    @patch("legacylens.features.business_logic_extractor.assemble_context")
    @patch("legacylens.features.business_logic_extractor.rerank")
    @patch("legacylens.features.business_logic_extractor.search")
    def test_business_rules_is_non_empty_string(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """business_rules should be a non-empty string from the LLM."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_SEARCH_RESULTS},
        }
        mock_rerank.return_value = _MOCK_SEARCH_RESULTS
        mock_assemble.return_value = _MOCK_SEARCH_RESULTS
        mock_llm.return_value = _MOCK_LLM_RESPONSE

        result = extract_business_logic(
            "What validation logic is applied to customer input data?"
        )

        assert isinstance(result["business_rules"], str)
        assert len(result["business_rules"]) > 0

    @patch("legacylens.features.business_logic_extractor._call_business_logic_llm")
    @patch("legacylens.features.business_logic_extractor.assemble_context")
    @patch("legacylens.features.business_logic_extractor.rerank")
    @patch("legacylens.features.business_logic_extractor.search")
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

        result = extract_business_logic(
            "What are the conditions and thresholds in CUST01?"
        )

        assert len(result["sources"]) > 0
        for src in result["sources"]:
            assert "file_path" in src
            assert "line_number" in src
            assert "paragraph_name" in src


class TestExtractBusinessLogicContent:
    """Verify the LLM is prompted to surface conditions, thresholds, variables."""

    @patch("legacylens.features.business_logic_extractor._call_business_logic_llm")
    @patch("legacylens.features.business_logic_extractor.assemble_context")
    @patch("legacylens.features.business_logic_extractor.rerank")
    @patch("legacylens.features.business_logic_extractor.search")
    def test_llm_receives_context_with_conditions(
        self, mock_search, mock_rerank, mock_assemble, mock_llm
    ) -> None:
        """The LLM call should include context containing IF/EVALUATE statements."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": _MOCK_SEARCH_RESULTS},
        }
        mock_rerank.return_value = _MOCK_SEARCH_RESULTS
        mock_assemble.return_value = _MOCK_SEARCH_RESULTS
        mock_llm.return_value = _MOCK_LLM_RESPONSE

        extract_business_logic("What business rules exist in CUST01?")

        mock_llm.assert_called_once()
        messages = mock_llm.call_args[0][0]
        user_content = messages[1]["content"]
        assert "IF" in user_content or "EVALUATE" in user_content


class TestExtractBusinessLogicEdgeCases:
    """Edge cases: empty query, no results, LLM failure, search failure."""

    def test_empty_query_returns_error(self) -> None:
        """Empty query returns success=False."""
        result = extract_business_logic("")
        assert result["success"] is False
        assert "error" in result

    def test_none_query_returns_error(self) -> None:
        """None query returns success=False."""
        result = extract_business_logic(None)
        assert result["success"] is False
        assert "error" in result

    @patch("legacylens.features.business_logic_extractor.assemble_context")
    @patch("legacylens.features.business_logic_extractor.rerank")
    @patch("legacylens.features.business_logic_extractor.search")
    def test_no_results_returns_not_found(
        self, mock_search, mock_rerank, mock_assemble
    ) -> None:
        """When search returns no results, return a not-found message."""
        mock_search.return_value = {
            "success": True,
            "data": {"results": []},
        }

        result = extract_business_logic("What business rules exist in NONEXISTENT?")

        assert result["success"] is True
        assert "not found" in result["business_rules"].lower()

    @patch("legacylens.features.business_logic_extractor._call_business_logic_llm")
    @patch("legacylens.features.business_logic_extractor.assemble_context")
    @patch("legacylens.features.business_logic_extractor.rerank")
    @patch("legacylens.features.business_logic_extractor.search")
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

        result = extract_business_logic("What are the business rules in CUST01?")

        assert result["success"] is True
        assert "unavailable" in result["business_rules"].lower()

    @patch("legacylens.features.business_logic_extractor.search")
    def test_search_failure_returns_error(self, mock_search) -> None:
        """When search itself fails, return success=False with error."""
        mock_search.return_value = {
            "success": False,
            "data": None,
            "error": "Search service down",
        }

        result = extract_business_logic("What business rules exist in CUST01?")

        assert result["success"] is False
        assert "error" in result
