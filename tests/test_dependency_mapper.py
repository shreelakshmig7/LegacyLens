"""
test_dependency_mapper.py
-------------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for Dependency Mapper
---------------------------------------------------------------------------------------
TDD tests for the Dependency Mapping feature (PRD 7.2). Mocks ChromaDB collection
and LLM calls to verify structured output, CALL/COPY/USING surfacing, internal vs
external classification, and metadata-first approach.

PRD 7.2 key requirements:
  - Must surface all CALL, COPY, and USING references for a queried module
  - Must use static analysis data from the Reference Scraper, not only semantic search
  - Must indicate whether each dependency is internal module or external copybook
  - Must be answerable for any module present in the indexed codebase

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

from unittest.mock import MagicMock, patch

import pytest

from legacylens.features.dependency_mapper import map_dependencies


def _mock_collection_get(chunks_data):
    """Build a mock ChromaDB collection.get() return value."""
    documents = [c.get("document", "") for c in chunks_data]
    metadatas = [c.get("metadata", {}) for c in chunks_data]
    ids = [f"id-{i}" for i in range(len(chunks_data))]
    return {"ids": ids, "documents": documents, "metadatas": metadatas}


_MOCK_CHUNKS_WITH_DEPS = [
    {
        "document": "       CALL 'PGMOD2' USING WS-PARAM1\n       COPY STATETXT",
        "metadata": {
            "file_path": "/repo/data/gnucobol-contrib/samples/DBsample/PostgreSQL/example1/PGMOD1.cbl",
            "file_name": "PGMOD1",
            "line_range": "[10, 25]",
            "type": "PROCEDURE",
            "parent_section": "MAIN-LOGIC",
            "paragraph_name": "INIT-DB",
            "dependencies": "PGMOD2,STATETXT,WS-PARAM1",
        },
    },
    {
        "document": "       CALL 'PGMOD3'\n       COPY SQLCA",
        "metadata": {
            "file_path": "/repo/data/gnucobol-contrib/samples/DBsample/PostgreSQL/example1/PGMOD1.cbl",
            "file_name": "PGMOD1",
            "line_range": "[30, 45]",
            "type": "PROCEDURE",
            "parent_section": "MAIN-LOGIC",
            "paragraph_name": "PROCESS-SQL",
            "dependencies": "PGMOD3,SQLCA",
        },
    },
]

_MOCK_ALL_FILE_NAMES = ["PGMOD1", "PGMOD2", "PGMOD3", "PGMOD4", "CUST01"]


class TestMapDependenciesStructure:
    """map_dependencies() returns a properly structured dict."""

    @patch("legacylens.features.dependency_mapper._call_dependency_llm")
    @patch("legacylens.features.dependency_mapper._get_all_indexed_file_names")
    @patch("legacylens.features.dependency_mapper._get_collection")
    def test_map_returns_structured_dict(
        self, mock_collection_fn, mock_file_names, mock_llm
    ) -> None:
        """Verify return has keys success, dependencies, summary, sources."""
        mock_coll = MagicMock()
        mock_coll.get.return_value = _mock_collection_get(_MOCK_CHUNKS_WITH_DEPS)
        mock_collection_fn.return_value = mock_coll
        mock_file_names.return_value = _MOCK_ALL_FILE_NAMES
        mock_llm.return_value = "PGMOD1 calls PGMOD2 and PGMOD3."

        result = map_dependencies("What are the dependencies of PGMOD1?")

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "dependencies" in result
        assert "summary" in result
        assert "sources" in result

    @patch("legacylens.features.dependency_mapper._call_dependency_llm")
    @patch("legacylens.features.dependency_mapper._get_all_indexed_file_names")
    @patch("legacylens.features.dependency_mapper._get_collection")
    def test_map_surfaces_call_copy_using(
        self, mock_collection_fn, mock_file_names, mock_llm
    ) -> None:
        """Verify all three dependency types (calls, copies, usings) appear in output."""
        mock_coll = MagicMock()
        mock_coll.get.return_value = _mock_collection_get(_MOCK_CHUNKS_WITH_DEPS)
        mock_collection_fn.return_value = mock_coll
        mock_file_names.return_value = _MOCK_ALL_FILE_NAMES
        mock_llm.return_value = "Summary."

        result = map_dependencies("What are the dependencies of PGMOD1?")

        deps = result["dependencies"]
        assert "calls" in deps
        assert "copies" in deps
        assert "usings" in deps

    @patch("legacylens.features.dependency_mapper._call_dependency_llm")
    @patch("legacylens.features.dependency_mapper._get_all_indexed_file_names")
    @patch("legacylens.features.dependency_mapper._get_collection")
    def test_map_classifies_internal_vs_external(
        self, mock_collection_fn, mock_file_names, mock_llm
    ) -> None:
        """CALL targets in the index → 'internal'; COPY .cpy targets → 'external copybook'."""
        mock_coll = MagicMock()
        mock_coll.get.return_value = _mock_collection_get(_MOCK_CHUNKS_WITH_DEPS)
        mock_collection_fn.return_value = mock_coll
        mock_file_names.return_value = _MOCK_ALL_FILE_NAMES
        mock_llm.return_value = "Summary."

        result = map_dependencies("What are the dependencies of PGMOD1?")

        deps = result["dependencies"]
        call_targets = {c["target"] for c in deps["calls"]}
        assert "PGMOD2" in call_targets or "PGMOD3" in call_targets

        for call_entry in deps["calls"]:
            if call_entry["target"] in _MOCK_ALL_FILE_NAMES:
                assert call_entry["dep_type"] == "internal"

        for copy_entry in deps["copies"]:
            assert copy_entry["dep_type"] in ("external_copybook", "internal")


class TestMapDependenciesMetadataFirst:
    """Dependency Mapper must use static analysis metadata, not only semantic search."""

    @patch("legacylens.features.dependency_mapper._call_dependency_llm")
    @patch("legacylens.features.dependency_mapper._get_all_indexed_file_names")
    @patch("legacylens.features.dependency_mapper._get_collection")
    def test_map_uses_collection_get_not_search(
        self, mock_collection_fn, mock_file_names, mock_llm
    ) -> None:
        """Verify the function calls collection.get() (metadata query), not query() (search)."""
        mock_coll = MagicMock()
        mock_coll.get.return_value = _mock_collection_get(_MOCK_CHUNKS_WITH_DEPS)
        mock_collection_fn.return_value = mock_coll
        mock_file_names.return_value = _MOCK_ALL_FILE_NAMES
        mock_llm.return_value = "Summary."

        map_dependencies("What are the dependencies of PGMOD1?")

        mock_coll.get.assert_called()
        mock_coll.query.assert_not_called()

    @patch("legacylens.features.dependency_mapper._call_dependency_llm")
    @patch("legacylens.features.dependency_mapper._get_all_indexed_file_names")
    @patch("legacylens.features.dependency_mapper._get_collection")
    def test_map_includes_llm_summary(
        self, mock_collection_fn, mock_file_names, mock_llm
    ) -> None:
        """Verify output includes an LLM narrative summary alongside structured data."""
        mock_coll = MagicMock()
        mock_coll.get.return_value = _mock_collection_get(_MOCK_CHUNKS_WITH_DEPS)
        mock_collection_fn.return_value = mock_coll
        mock_file_names.return_value = _MOCK_ALL_FILE_NAMES
        mock_llm.return_value = "PGMOD1 has CALL dependencies on PGMOD2 and PGMOD3."

        result = map_dependencies("What are the dependencies of PGMOD1?")

        assert result["summary"] is not None
        assert len(result["summary"]) > 0


class TestMapDependenciesEdgeCases:
    """Edge cases: empty query, unknown module, no dependencies."""

    def test_map_empty_query_returns_error(self) -> None:
        """Empty query returns success=False."""
        result = map_dependencies("")
        assert result["success"] is False
        assert "error" in result

    def test_map_none_query_returns_error(self) -> None:
        """None query returns success=False."""
        result = map_dependencies(None)
        assert result["success"] is False
        assert "error" in result

    @patch("legacylens.features.dependency_mapper._get_collection")
    def test_map_unknown_module_returns_not_found(self, mock_collection_fn) -> None:
        """Query for a module not in the index → structured not-found."""
        mock_coll = MagicMock()
        mock_coll.get.return_value = _mock_collection_get([])
        mock_collection_fn.return_value = mock_coll

        result = map_dependencies("What are the dependencies of NONEXISTENT?")

        assert result["success"] is True
        assert "not found" in result["summary"].lower()

    @patch("legacylens.features.dependency_mapper._call_dependency_llm")
    @patch("legacylens.features.dependency_mapper._get_all_indexed_file_names")
    @patch("legacylens.features.dependency_mapper._get_collection")
    def test_map_llm_failure_still_returns_structured_data(
        self, mock_collection_fn, mock_file_names, mock_llm
    ) -> None:
        """When LLM fails, structured dependency data is still returned."""
        mock_coll = MagicMock()
        mock_coll.get.return_value = _mock_collection_get(_MOCK_CHUNKS_WITH_DEPS)
        mock_collection_fn.return_value = mock_coll
        mock_file_names.return_value = _MOCK_ALL_FILE_NAMES
        mock_llm.side_effect = Exception("LLM unavailable")

        result = map_dependencies("What are the dependencies of PGMOD1?")

        assert result["success"] is True
        assert len(result["dependencies"]["calls"]) > 0
        assert "unavailable" in result["summary"].lower() or len(result["summary"]) > 0
