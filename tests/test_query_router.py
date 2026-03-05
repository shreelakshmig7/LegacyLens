"""
test_query_router.py
--------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for query-type detection
-----------------------------------------------------------------------------------------
TDD tests for detect_feature_type() which routes incoming queries to the
appropriate code-understanding feature module (explain, dependency, business_logic,
doc_generate) or falls through to the general pipeline.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import pytest

from legacylens.config.constants import (
    FEATURE_TYPE_BUSINESS_LOGIC,
    FEATURE_TYPE_DEPENDENCY,
    FEATURE_TYPE_DOC_GENERATE,
    FEATURE_TYPE_EXPLAIN,
    FEATURE_TYPE_GENERAL,
)
from legacylens.features import detect_feature_type


class TestDetectFeatureType:
    """Tests for detect_feature_type() query routing."""

    # ── Dependency queries (highest priority) ─────────────────────────────

    def test_dependency_query_explicit(self) -> None:
        """'dependencies of MODULE-X' routes to dependency feature."""
        result = detect_feature_type("What are the dependencies of MODULE-X?")
        assert result == FEATURE_TYPE_DEPENDENCY

    def test_dependency_query_call_chain(self) -> None:
        """'call chain' routes to dependency feature."""
        result = detect_feature_type("Trace the full call chain starting from the main entry point")
        assert result == FEATURE_TYPE_DEPENDENCY

    def test_dependency_query_copybooks(self) -> None:
        """'copybooks included' routes to dependency feature."""
        result = detect_feature_type("What copybooks are included via COPY statements?")
        assert result == FEATURE_TYPE_DEPENDENCY

    def test_dependency_query_using_clause(self) -> None:
        """'using clause' routes to dependency feature (ll-013)."""
        result = detect_feature_type("Which paragraphs pass parameters using USING clauses?")
        assert result == FEATURE_TYPE_DEPENDENCY

    # ── Explain queries ───────────────────────────────────────────────────

    def test_explain_query_paragraph(self) -> None:
        """'Explain what CALCULATE-INTEREST does' routes to explain (ll-003)."""
        result = detect_feature_type("Explain what the CALCULATE-INTEREST paragraph does")
        assert result == FEATURE_TYPE_EXPLAIN

    def test_explain_query_purpose(self) -> None:
        """'purpose of' routes to explain feature."""
        result = detect_feature_type("What is the purpose of the MAIN-ENTRY paragraph?")
        assert result == FEATURE_TYPE_EXPLAIN

    def test_explain_query_describe(self) -> None:
        """'describe the' routes to explain feature."""
        result = detect_feature_type("Describe the UPDATE-RECORD paragraph")
        assert result == FEATURE_TYPE_EXPLAIN

    # ── Business logic queries ────────────────────────────────────────────

    def test_business_logic_query_rules(self) -> None:
        """'business rules' routes to business_logic (ll-019)."""
        result = detect_feature_type("What business rules govern the interest calculation logic?")
        assert result == FEATURE_TYPE_BUSINESS_LOGIC

    def test_business_logic_query_validation(self) -> None:
        """'validation logic' routes to business_logic when not prefixed by 'explain'."""
        result = detect_feature_type("What validation logic is applied to customer input data?")
        assert result == FEATURE_TYPE_BUSINESS_LOGIC

    def test_explain_prefix_overrides_business_logic(self) -> None:
        """'Explain the validation rules' routes to explain (explain has higher priority)."""
        result = detect_feature_type("Explain the validation rules applied to customer input data")
        assert result == FEATURE_TYPE_EXPLAIN

    # ── Doc generate queries ──────────────────────────────────────────────

    def test_doc_generate_query(self) -> None:
        """'generate documentation' routes to doc_generate."""
        result = detect_feature_type("Generate documentation for the CUST01 program")
        assert result == FEATURE_TYPE_DOC_GENERATE

    def test_doc_generate_auto_document(self) -> None:
        """'auto-document' routes to doc_generate."""
        result = detect_feature_type("Auto-document the PGMOD1 module")
        assert result == FEATURE_TYPE_DOC_GENERATE

    # ── General queries (no feature match) ────────────────────────────────

    def test_general_query_entry_point(self) -> None:
        """'Where is the main entry point' is a general retrieval query (ll-001)."""
        result = detect_feature_type("Where is the main entry point of this program?")
        assert result == FEATURE_TYPE_GENERAL

    def test_general_query_file_io(self) -> None:
        """'Find all file I/O operations' is general retrieval (ll-004)."""
        result = detect_feature_type("Find all file I/O operations")
        assert result == FEATURE_TYPE_GENERAL

    def test_general_query_modify(self) -> None:
        """'What functions modify' is general retrieval (ll-002)."""
        result = detect_feature_type("What functions modify the CUSTOMER-RECORD?")
        assert result == FEATURE_TYPE_GENERAL

    def test_general_query_error_handling(self) -> None:
        """'error handling patterns' is general retrieval (ll-006)."""
        result = detect_feature_type("Show me error handling patterns in this codebase")
        assert result == FEATURE_TYPE_GENERAL

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_empty_query_returns_general(self) -> None:
        """Empty string routes to general."""
        assert detect_feature_type("") == FEATURE_TYPE_GENERAL

    def test_none_query_returns_general(self) -> None:
        """None input routes to general."""
        assert detect_feature_type(None) == FEATURE_TYPE_GENERAL

    def test_whitespace_query_returns_general(self) -> None:
        """Whitespace-only input routes to general."""
        assert detect_feature_type("   ") == FEATURE_TYPE_GENERAL

    def test_case_insensitive(self) -> None:
        """Routing is case-insensitive."""
        assert detect_feature_type("EXPLAIN WHAT the paragraph does") == FEATURE_TYPE_EXPLAIN
        assert detect_feature_type("DEPENDENCIES OF module-x") == FEATURE_TYPE_DEPENDENCY

    def test_priority_dependency_over_explain(self) -> None:
        """'explain the dependencies' routes to dependency (higher priority)."""
        result = detect_feature_type("Explain the dependencies of PGMOD1")
        assert result == FEATURE_TYPE_DEPENDENCY
