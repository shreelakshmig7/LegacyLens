"""
run_eval.py
-----------
LegacyLens — RAG System for Legacy Enterprise Codebases — Evaluation runner
----------------------------------------------------------------------------
Loads the 20-query golden benchmark from eval/golden_data.yaml and runs each
test case through the full LegacyLens pipeline (retrieval + answer generation),
scoring two metrics: (1) Retrieval Precision — correct file_path/paragraph_name in top-5;
(2) Answer Faithfulness — must_contain / must_not_contain (stubbed in PR 3, real in PR 4).

Pipeline:
  1. Load golden_data.yaml
  2. For each test case: search → rerank → assemble_context → (stub) answer
  3. Score retrieval: expected_chunks present in top-5?
  4. Score answer: must_contain / must_not_contain
  5. Write timestamped results to tests/results/ with both metrics

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import argparse
import logging
import os
import pathlib
from datetime import datetime
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

# Optional dotenv so REPO_PATH is set when running eval
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_GOLDEN_DATA_PATH = _REPO_ROOT / "eval" / "golden_data.yaml"
_RESULTS_DIR = _REPO_ROOT / "tests" / "results"

# Mandatory anchor case IDs (ll-001 through ll-006) for --fast runs
_FAST_CASE_IDS = frozenset({"ll-001", "ll-002", "ll-003", "ll-004", "ll-005", "ll-006"})


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

def score_retrieval_precision(
    test_case: Dict[str, Any],
    top5_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Score whether expected_chunks (file_path, paragraph_name) appear in top-5 results.

    Args:
        test_case: Golden test case with expected_chunks list.
        top5_results: List of result dicts with metadata.file_path, metadata.paragraph_name.

    Returns:
        dict: {
            "id": str,
            "passed": bool,
            "matched_chunks": list[dict],
            "missing_chunks": list[dict],
        }
    """
    expected = test_case.get("expected_chunks") or []
    if not expected:
        return {
            "id": test_case["id"],
            "passed": True,
            "matched_chunks": [],
            "missing_chunks": [],
        }

    def norm_path(p: str) -> str:
        return (p or "").strip().replace("\\", "/").lower()

    def norm_para(p: str) -> str:
        return (p or "").strip().upper()

    def to_relative_path(abs_path: str) -> str:
        """Normalize Chroma absolute path to relative form (data/gnucobol-contrib/...) for comparison."""
        if not abs_path:
            return abs_path
        if "data/gnucobol-contrib" in abs_path:
            return "data/gnucobol-contrib" + abs_path.split("data/gnucobol-contrib", 1)[-1]
        return abs_path

    matched = []
    missing = []
    for exp in expected:
        exp_fp = norm_path(exp.get("file_path") or "")
        exp_para = norm_para(exp.get("paragraph_name") or "")
        if not exp_fp and not exp_para:
            matched.append(exp)
            continue
        found = False
        for res in top5_results:
            meta = res.get("metadata") or {}
            res_fp_raw = norm_path(meta.get("file_path") or "")
            res_fp = to_relative_path(res_fp_raw)
            res_para = norm_para(meta.get("paragraph_name") or "")
            path_ok = (not exp_fp) or (exp_fp == res_fp or exp_fp in res_fp or res_fp.endswith(exp_fp))
            para_ok = (not exp_para) or (exp_para == res_para)
            if path_ok and para_ok:
                found = True
                matched.append(exp)
                break
        if not found:
            missing.append(exp)

    return {
        "id": test_case["id"],
        "passed": len(missing) == 0,
        "matched_chunks": matched,
        "missing_chunks": missing,
    }


def score_answer(test_case: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """
    Score a generated answer against the must_contain / must_not_contain criteria.
    (Answer Faithfulness — stubbed in PR 3, real generation in PR 4.)

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
# Main runner (retrieval wired in PR 3; answer generation stubbed until PR 4)
# ---------------------------------------------------------------------------

def run_eval(fast: bool = False) -> Dict[str, Any]:
    """
    Execute the full evaluation benchmark with retrieval pipeline and dual metrics.

    Metrics:
      1. Retrieval Precision: expected file_path/paragraph_name in top-5 (target >70%).
      2. Answer Faithfulness: must_contain / must_not_contain (stubbed in PR 3).

    Args:
        fast: If True, run only the 6 mandatory anchor cases (ll-001 through ll-006).

    Returns:
        dict: {
            "success": bool,
            "total": int,
            "retrieval_precision_pct": float,
            "retrieval_passed": int,
            "answer_passed": int,
            "results": list[dict],
            "results_path": str,
        }
    """
    try:
        from legacylens.retrieval.context_assembler import assemble_context
        from legacylens.retrieval.reranker import rerank
        from legacylens.retrieval.searcher import search
    except ImportError as exc:
        logger.error("Retrieval imports failed: %s", exc)
        return {"success": False, "error": str(exc), "total": 0, "retrieval_precision_pct": 0.0}

    try:
        test_cases = load_golden_data()
        if fast:
            test_cases = [t for t in test_cases if t.get("id") in _FAST_CASE_IDS]
            logger.info("Fast mode: running %d mandatory cases only", len(test_cases))
        repo_root = os.getenv("REPO_PATH", "")
        retrieval_results: List[Dict[str, Any]] = []
        answer_results: List[Dict[str, Any]] = []

        for tc in test_cases:
            print(f"Starting {tc['id']}...", flush=True)
            query = tc.get("query") or ""

            # Retrieval: search → rerank → assemble
            search_out = search(query)
            print("  searcher done", flush=True)

            if not search_out.get("success"):
                # Timeout or API error: mark case as ERROR and move to next
                err_msg = search_out.get("error", "Retrieval failed")
                retrieval_results.append({
                    "id": tc["id"],
                    "passed": False,
                    "matched_chunks": [],
                    "missing_chunks": tc.get("expected_chunks") or [],
                    "error": err_msg,
                })
                answer_results.append({
                    "id": tc["id"],
                    "passed": False,
                    "missing_terms": tc.get("must_contain", []),
                    "forbidden_terms": [],
                    "error": err_msg,
                })
                logger.warning("[ERROR] %s — %s", tc["id"], err_msg)
                continue

            top5 = search_out.get("data", {}).get("results") or []
            reranked = rerank(top5, query)
            print("  reranker done", flush=True)
            assembled = assemble_context(reranked, repo_root=repo_root)
            print("  context_assembler done", flush=True)

            # Retrieval precision: expected_chunks in top-5?
            ret_score = score_retrieval_precision(tc, reranked)
            retrieval_results.append(ret_score)

            # Answer: stub until PR 4
            answer = (
                "[STUB] Answer generation in PR 4. "
                f"Query: {query}. "
                f"Retrieved {len(reranked)} chunks."
            )
            ans_score = score_answer(tc, answer)
            answer_results.append(ans_score)

            ret_status = "PASS" if ret_score["passed"] else "FAIL"
            ans_status = "PASS" if ans_score["passed"] else "FAIL"
            logger.info(
                "[%s ret / %s ans] %s — %s",
                ret_status,
                ans_status,
                tc["id"],
                tc["query"][:50],
            )

        total = len(test_cases)
        retrieval_passed = sum(1 for r in retrieval_results if r["passed"])
        answer_passed = sum(1 for r in answer_results if r["passed"])
        retrieval_precision_pct = (retrieval_passed / total * 100.0) if total else 0.0

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        results_path = _RESULTS_DIR / f"eval_{timestamp}.txt"
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w", encoding="utf-8") as fh:
            fh.write(f"LegacyLens Eval — {timestamp}\n")
            fh.write(f"Total: {total}\n")
            fh.write(f"Retrieval Precision: {retrieval_passed}/{total} ({retrieval_precision_pct:.1f}%) — target >70%\n")
            fh.write(f"Answer Faithfulness (stub): {answer_passed}/{total}\n\n")
            for i, tc in enumerate(test_cases):
                ret = retrieval_results[i]
                ans = answer_results[i]
                status = "ERROR" if ret.get("error") else ("PASS" if ret["passed"] else "FAIL")
                fh.write(f"[ret {status} / ans {'PASS' if ans['passed'] else 'FAIL'}] {tc['id']}\n")
                if ret.get("error"):
                    fh.write(f"  retrieval ERROR: {ret['error']}\n")
                if ret.get("missing_chunks"):
                    fh.write(f"  retrieval missing: {ret['missing_chunks']}\n")
                if ans.get("missing_terms"):
                    fh.write(f"  answer missing: {ans['missing_terms']}\n")
                if ans.get("forbidden_terms"):
                    fh.write(f"  answer forbidden: {ans['forbidden_terms']}\n")

        logger.info(
            "Eval complete. Retrieval: %d/%d (%.1f%%). Results: %s",
            retrieval_passed,
            total,
            retrieval_precision_pct,
            results_path,
        )

        return {
            "success": retrieval_precision_pct >= 70.0,
            "total": total,
            "retrieval_precision_pct": retrieval_precision_pct,
            "retrieval_passed": retrieval_passed,
            "answer_passed": answer_passed,
            "results": [{"retrieval": r, "answer": a} for r, a in zip(retrieval_results, answer_results)],
            "results_path": str(results_path),
        }
    except Exception as exc:
        logger.exception("Eval runner failed: %s", exc)
        return {
            "success": False,
            "error": str(exc),
            "total": 0,
            "retrieval_precision_pct": 0.0,
            "retrieval_passed": 0,
            "answer_passed": 0,
            "results": [],
            "results_path": "",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LegacyLens eval benchmark")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only 6 mandatory cases (ll-001 through ll-006)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = run_eval(fast=args.fast)
    print(f"\nResult: {'PASS' if summary.get('success') else 'FAIL'}")
    print(f"Retrieval: {summary.get('retrieval_passed', 0)}/{summary.get('total', 0)} ({summary.get('retrieval_precision_pct', 0):.1f}%)")
    print(f"Answer (stub): {summary.get('answer_passed', 0)}/{summary.get('total', 0)}")
