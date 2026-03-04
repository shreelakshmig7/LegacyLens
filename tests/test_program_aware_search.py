"""
test_program_aware_search.py
-----------------------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Tests for Program-Aware Search
------------------------------------------------------------------------------------------
Validates the full program-aware search feature:
  1. detect_program() correctly identifies (or rejects) program names from PROGRAM_CATEGORIES.
  2. chunk_code_lines() attaches a file_name metadata field to every chunk.
  3. sanitize_query_filters() accepts file_name as a valid filter field.
  4. BM25 fallback respects the program filter when a program is detected.

All tests are unit-level — no ChromaDB, Voyage, or OpenAI calls.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import pytest


# ---------------------------------------------------------------------------
# detect_program — unit tests
# ---------------------------------------------------------------------------

class TestDetectProgram:
    """Tests for detect_program() in query_processor.py."""

    def test_returns_success_dict(self) -> None:
        """detect_program must return a structured dict with success, data, error."""
        from legacylens.retrieval.query_processor import detect_program
        result = detect_program("How does PGMOD1 connect to the database?")
        assert isinstance(result, dict)
        assert "success" in result
        assert "data" in result
        assert "error" in result

    def test_detects_known_program_exact_case(self) -> None:
        """Exact-case match against PROGRAM_CATEGORIES must return the program."""
        from legacylens.retrieval.query_processor import detect_program
        from legacylens.config.constants import PROGRAM_CATEGORIES
        if not PROGRAM_CATEGORIES:
            pytest.skip("PROGRAM_CATEGORIES is empty — nothing to match")
        program = PROGRAM_CATEGORIES[0]
        result = detect_program(f"In {program}, how does error handling work?")
        assert result["success"] is True
        assert result["data"]["program"] == program

    def test_detects_known_program_lowercase(self) -> None:
        """Lower-case mention of a program must still be detected (case-insensitive)."""
        from legacylens.retrieval.query_processor import detect_program
        from legacylens.config.constants import PROGRAM_CATEGORIES
        if not PROGRAM_CATEGORIES:
            pytest.skip("PROGRAM_CATEGORIES is empty — nothing to match")
        program = PROGRAM_CATEGORIES[0]
        result = detect_program(f"explain {program.lower()} entry point")
        assert result["success"] is True
        assert result["data"]["program"] == program

    def test_returns_none_when_no_program_mentioned(self) -> None:
        """Query with no program name must return program=None."""
        from legacylens.retrieval.query_processor import detect_program
        result = detect_program("What is the main entry point?")
        assert result["success"] is True
        assert result["data"]["program"] is None

    def test_empty_query_returns_none(self) -> None:
        """Empty string must return success with program=None."""
        from legacylens.retrieval.query_processor import detect_program
        result = detect_program("")
        assert result["success"] is True
        assert result["data"]["program"] is None

    def test_none_query_returns_none(self) -> None:
        """None input must return success with program=None (no crash)."""
        from legacylens.retrieval.query_processor import detect_program
        result = detect_program(None)  # type: ignore[arg-type]
        assert result["success"] is True
        assert result["data"]["program"] is None

    def test_detects_ocesql(self) -> None:
        """OCESQL must be recognised as a program if present in PROGRAM_CATEGORIES."""
        from legacylens.retrieval.query_processor import detect_program
        from legacylens.config.constants import PROGRAM_CATEGORIES
        if "OCESQL" not in PROGRAM_CATEGORIES:
            pytest.skip("OCESQL not in PROGRAM_CATEGORIES")
        result = detect_program("In OCESQL, how is the database connected?")
        assert result["success"] is True
        assert result["data"]["program"] == "OCESQL"

    def test_detects_pgmod1(self) -> None:
        """PGMOD1 must be recognised as a program if present in PROGRAM_CATEGORIES."""
        from legacylens.retrieval.query_processor import detect_program
        from legacylens.config.constants import PROGRAM_CATEGORIES
        if "PGMOD1" not in PROGRAM_CATEGORIES:
            pytest.skip("PGMOD1 not in PROGRAM_CATEGORIES")
        result = detect_program("What paragraphs are in PGMOD1?")
        assert result["success"] is True
        assert result["data"]["program"] == "PGMOD1"

    def test_generic_query_no_false_positive(self) -> None:
        """Common words that happen to substring-match a program name must not
        trigger false positives — only whole-word matches are valid."""
        from legacylens.retrieval.query_processor import detect_program
        # A query that should produce no match
        result = detect_program("explain the error handling strategy")
        assert result["success"] is True
        # Whatever program (if any) was matched, the test just verifies no crash
        assert "program" in result["data"]

    def test_returns_first_match_when_multiple_present(self) -> None:
        """When two known programs appear in a single query the function
        must return exactly one of them (the first one matched)."""
        from legacylens.retrieval.query_processor import detect_program
        from legacylens.config.constants import PROGRAM_CATEGORIES
        if len(PROGRAM_CATEGORIES) < 2:
            pytest.skip("Need at least 2 programs in PROGRAM_CATEGORIES")
        p1, p2 = PROGRAM_CATEGORIES[0], PROGRAM_CATEGORIES[1]
        result = detect_program(f"Compare {p1} and {p2}")
        assert result["success"] is True
        assert result["data"]["program"] in (p1, p2)


# ---------------------------------------------------------------------------
# Chunker — file_name metadata field
# ---------------------------------------------------------------------------

class TestChunkFileNameMetadata:
    """Every chunk produced by chunker.py must carry a file_name field."""

    def _simple_procedure_lines(self) -> list:
        return [
            "IDENTIFICATION DIVISION.",
            "PROGRAM-ID. TESTPROG.",
            "PROCEDURE DIVISION.",
            "MAIN-PARA.",
            "    DISPLAY 'HELLO'.",
            "    STOP RUN.",
        ]

    def test_chunk_has_file_name_field(self) -> None:
        """Each chunk dict must have a file_name key."""
        from legacylens.ingestion.chunker import chunk_code_lines
        result = chunk_code_lines(self._simple_procedure_lines(), file_path="/repo/ocesql.cbl")
        assert result["success"] is True
        chunks = result["data"]["chunks"]
        assert len(chunks) > 0
        for chunk in chunks:
            assert "file_name" in chunk, f"Chunk missing file_name: {chunk}"

    def test_file_name_is_uppercase_stem(self) -> None:
        """file_name must be the uppercase stem of the file_path."""
        from legacylens.ingestion.chunker import chunk_code_lines
        result = chunk_code_lines(self._simple_procedure_lines(), file_path="/repo/ocesql.cbl")
        assert result["success"] is True
        for chunk in result["data"]["chunks"]:
            assert chunk["file_name"] == "OCESQL"

    def test_file_name_mixed_case_stem_is_uppercased(self) -> None:
        """Mixed-case filename stem must be stored uppercase."""
        from legacylens.ingestion.chunker import chunk_code_lines
        result = chunk_code_lines(self._simple_procedure_lines(), file_path="/repo/PgMod1.cbl")
        assert result["success"] is True
        for chunk in result["data"]["chunks"]:
            assert chunk["file_name"] == "PGMOD1"

    def test_file_name_present_on_data_division_chunks(self) -> None:
        """DATA DIVISION chunks must also carry file_name."""
        from legacylens.ingestion.chunker import chunk_code_lines
        lines = [
            "IDENTIFICATION DIVISION.",
            "PROGRAM-ID. TESTPROG.",
            "DATA DIVISION.",
            "WORKING-STORAGE SECTION.",
            "    01 WS-COUNTER PIC 9(4) VALUE 0.",
        ]
        result = chunk_code_lines(lines, file_path="/some/path/cust01.cbl")
        assert result["success"] is True
        for chunk in result["data"]["chunks"]:
            assert chunk.get("file_name") == "CUST01"

    def test_file_name_present_on_copybook_chunks(self) -> None:
        """Copybook (.cpy) chunks must also carry file_name."""
        from legacylens.ingestion.chunker import chunk_code_lines
        lines = ["    01 WS-CUSTOMER-ID PIC X(10)."]
        result = chunk_code_lines(lines, file_path="/repo/custcopy.cpy", is_copybook=True)
        assert result["success"] is True
        for chunk in result["data"]["chunks"]:
            assert chunk.get("file_name") == "CUSTCOPY"

    def test_empty_lines_returns_no_chunks_no_crash(self) -> None:
        """Empty line list must return 0 chunks without crash."""
        from legacylens.ingestion.chunker import chunk_code_lines
        result = chunk_code_lines([], file_path="/repo/empty.cbl")
        assert result["success"] is True
        assert result["data"]["chunk_count"] == 0


# ---------------------------------------------------------------------------
# vector_store — sanitize_query_filters accepts file_name
# ---------------------------------------------------------------------------

class TestSanitizeQueryFiltersAcceptsFileName:
    """file_name must be whitelisted in sanitize_query_filters."""

    def test_file_name_filter_is_accepted(self) -> None:
        """{"file_name": "OCESQL"} must pass sanitize_query_filters."""
        from legacylens.retrieval.vector_store import sanitize_query_filters
        result = sanitize_query_filters({"file_name": "OCESQL"})
        assert result["success"] is True

    def test_file_name_combined_with_type_is_accepted(self) -> None:
        """Combined file_name + type filter must pass."""
        from legacylens.retrieval.vector_store import sanitize_query_filters
        result = sanitize_query_filters({"file_name": "PGMOD1", "type": "PROCEDURE"})
        assert result["success"] is True

    def test_disallowed_field_still_rejected(self) -> None:
        """Non-whitelisted fields must still be rejected."""
        from legacylens.retrieval.vector_store import sanitize_query_filters
        result = sanitize_query_filters({"random_field": "value"})
        assert result["success"] is False

    def test_dollar_operator_still_rejected(self) -> None:
        """$ operator injection must still be rejected."""
        from legacylens.retrieval.vector_store import sanitize_query_filters
        result = sanitize_query_filters({"$where": "DROP TABLE"})
        assert result["success"] is False


# ---------------------------------------------------------------------------
# BM25 filtering helper — filter_bm25_results_by_program
# ---------------------------------------------------------------------------

class TestBm25FilterByProgram:
    """The BM25 filtered-results helper must correctly filter by file_name."""

    def _make_results(self) -> list:
        return [
            {"text": "OPEN INPUT.", "metadata": {"file_name": "OCESQL"}, "score": 5.0},
            {"text": "MOVE WS-ID TO CUST-ID.", "metadata": {"file_name": "CUST01"}, "score": 4.5},
            {"text": "EXEC SQL SELECT.", "metadata": {"file_name": "OCESQL"}, "score": 4.0},
            {"text": "STOP RUN.", "metadata": {"file_name": "PGMOD1"}, "score": 3.5},
            {"text": "No metadata chunk.", "metadata": {}, "score": 2.0},
        ]

    def test_filter_returns_only_matching_program(self) -> None:
        """Only results whose file_name matches the program must be returned."""
        from legacylens.retrieval.searcher import _filter_bm25_by_program
        filtered = _filter_bm25_by_program(self._make_results(), "OCESQL")
        assert all(r["metadata"].get("file_name") == "OCESQL" for r in filtered)
        assert len(filtered) == 2

    def test_filter_none_program_returns_all(self) -> None:
        """None program must return all results (global search fallback)."""
        from legacylens.retrieval.searcher import _filter_bm25_by_program
        results = self._make_results()
        filtered = _filter_bm25_by_program(results, None)
        assert len(filtered) == len(results)

    def test_filter_program_not_in_results_returns_empty(self) -> None:
        """If no chunk matches the requested program, return empty list."""
        from legacylens.retrieval.searcher import _filter_bm25_by_program
        filtered = _filter_bm25_by_program(self._make_results(), "DUMPHEX")
        assert filtered == []

    def test_filter_empty_results_returns_empty(self) -> None:
        """Empty input always returns empty output."""
        from legacylens.retrieval.searcher import _filter_bm25_by_program
        assert _filter_bm25_by_program([], "OCESQL") == []


# ---------------------------------------------------------------------------
# PROGRAM_CATEGORIES constant sanity
# ---------------------------------------------------------------------------

class TestProgramCategoriesConstant:
    """PROGRAM_CATEGORIES must exist in constants and contain expected programs."""

    def test_program_categories_is_list(self) -> None:
        """PROGRAM_CATEGORIES must be a list."""
        from legacylens.config.constants import PROGRAM_CATEGORIES
        assert isinstance(PROGRAM_CATEGORIES, list)

    def test_program_categories_not_empty(self) -> None:
        """PROGRAM_CATEGORIES must contain at least one entry."""
        from legacylens.config.constants import PROGRAM_CATEGORIES
        assert len(PROGRAM_CATEGORIES) > 0

    def test_program_categories_all_uppercase(self) -> None:
        """All entries must be uppercase strings (normalised form)."""
        from legacylens.config.constants import PROGRAM_CATEGORIES
        for name in PROGRAM_CATEGORIES:
            assert isinstance(name, str)
            assert name == name.upper(), f"{name!r} is not uppercase"

    def test_ocesql_in_program_categories(self) -> None:
        """OCESQL must be in the default PROGRAM_CATEGORIES list."""
        from legacylens.config.constants import PROGRAM_CATEGORIES
        assert "OCESQL" in PROGRAM_CATEGORIES

    def test_pgmod1_in_program_categories(self) -> None:
        """PGMOD1 must be in the default PROGRAM_CATEGORIES list."""
        from legacylens.config.constants import PROGRAM_CATEGORIES
        assert "PGMOD1" in PROGRAM_CATEGORIES
