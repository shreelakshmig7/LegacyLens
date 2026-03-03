"""
run_eval.py
-----------
LegacyLens — RAG System for Legacy Enterprise Codebases — Evaluation runner
----------------------------------------------------------------------------
Loads the 20-query golden benchmark from eval/golden_data.yaml and runs each
test case through the full LegacyLens pipeline (retrieval + answer generation),
scoring retrieval precision and answer quality against the ground-truth criteria.

Pipeline:
  1. Load golden_data.yaml
  2. For each test case: submit query → retrieve top-k chunks → generate answer
  3. Score: check must_contain / must_not_contain in the answer
  4. Compute precision@5 over expected_chunks
  5. Write a timestamped results file to tests/results/

Invoke after every new feature to catch regressions before moving on.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import os
import pathlib
from datetime import datetime
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_GOLDEN_DATA_PATH = _REPO_ROOT / "eval" / "golden_data.yaml"
_RESULTS_DIR = _REPO_ROOT / "tests" / "results"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_golden_data(path: pathlib.Path = _GOLDEN_DATA_PATH) -> List[Dict[str, Any]]:
    """
    Load and return the list of golden test cases from the YAML benchmark file.

    Args:
        path: Filesystem path to golden_data.yaml.

    Returns:
        list[dict]: Each dict is one test case as defined in the benchmark format.

    Raises:
        FileNotFoundError: If the YAML file does not exist at the given path.
        yaml.YAMLError: If the file cannot be parsed as valid YAML.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        test_cases: List[Dict[str, Any]] = data.get("test_cases", [])
        logger.info("Loaded %d golden test cases from %s", len(test_cases), path)
        return test_cases
    except FileNotFoundError:
        logger.error("Golden data file not found: %s", path)
        raise
    except yaml.YAMLError as exc:
        logger.error("Failed to parse golden data YAML: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def score_answer(test_case: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """
    Score a generated answer against the must_contain / must_not_contain criteria.

    Args:
        test_case: A single golden test-case dict with must_contain and
                   must_not_contain lists.
        answer:    The raw string answer produced by the generation pipeline.

    Returns:
        dict: {
            "id": str,
            "passed": bool,
            "missing_terms": list[str]  — must_contain terms absent from answer,
            "forbidden_terms": list[str] — must_not_contain terms found in answer,
        }
    """
    answer_lower = answer.lower()
    missing = [t for t in test_case.get("must_contain", []) if t.lower() not in answer_lower]
    forbidden = [t for t in test_case.get("must_not_contain", []) if t.lower() in answer_lower]
    return {
        "id": test_case["id"],
        "passed": len(missing) == 0 and len(forbidden) == 0,
        "missing_terms": missing,
        "forbidden_terms": forbidden,
    }


# ---------------------------------------------------------------------------
# Main runner (stub — pipeline integration added in PR 3+)
# ---------------------------------------------------------------------------

def run_eval() -> Dict[str, Any]:
    """
    Execute the full evaluation benchmark and return a summary result dict.

    The retrieval and generation pipeline calls are stubbed here. They will be
    wired to the real implementations as each PR lands. The runner is intentionally
    executable now so CI can invoke it from PR 2 onward.

    Args:
        None

    Returns:
        dict: {
            "success": bool,
            "total": int,
            "passed": int,
            "failed": int,
            "results": list[dict],
            "results_path": str,
        }
    """
    try:
        test_cases = load_golden_data()
        results: List[Dict[str, Any]] = []

        for tc in test_cases:
            # Stub: replace with real pipeline call in PR 3+
            answer = (
                "[STUB] Pipeline not yet wired. "
                f"Query received: {tc['query']}"
            )
            score = score_answer(tc, answer)
            results.append(score)
            status = "PASS" if score["passed"] else "FAIL"
            logger.info("[%s] %s — %s", status, tc["id"], tc["query"])

        passed = sum(1 for r in results if r["passed"])
        failed = len(results) - passed

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        results_path = _RESULTS_DIR / f"eval_{timestamp}.txt"
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w", encoding="utf-8") as fh:
            fh.write(f"LegacyLens Eval — {timestamp}\n")
            fh.write(f"Total: {len(results)}  Passed: {passed}  Failed: {failed}\n\n")
            for r in results:
                status = "PASS" if r["passed"] else "FAIL"
                fh.write(f"[{status}] {r['id']}\n")
                if r["missing_terms"]:
                    fh.write(f"  missing: {r['missing_terms']}\n")
                if r["forbidden_terms"]:
                    fh.write(f"  forbidden: {r['forbidden_terms']}\n")

        logger.info("Eval complete. %d/%d passed. Results: %s", passed, len(results), results_path)

        return {
            "success": failed == 0,
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "results": results,
            "results_path": str(results_path),
        }

    except Exception as exc:
        logger.exception("Eval runner failed: %s", exc)
        return {"success": False, "error": str(exc), "total": 0, "passed": 0, "failed": 0}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = run_eval()
    print(f"\nResult: {'PASS' if summary['success'] else 'FAIL'}")
    print(f"Passed: {summary.get('passed', 0)} / {summary.get('total', 0)}")
