"""
capture_retrieval_results.py
----------------------------
One-off script: run each golden_data query through searcher and print top-5
file_path + paragraph_name so we can align golden_data and fill TBDs.
Run from project root: python -m scripts.capture_retrieval_results
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml
from legacylens.retrieval.reranker import rerank
from legacylens.retrieval.searcher import search

GOLDEN = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "golden_data.yaml")


def main():
    with open(GOLDEN, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cases = data.get("test_cases", [])
    for tc in cases:
        q = tc.get("query", "")
        cid = tc.get("id", "")
        print(f"\n=== {cid} === {q[:60]}...")
        out = search(q)
        if not out.get("success"):
            print("  search failed:", out.get("error"))
            continue
        results = out.get("data", {}).get("results") or []
        reranked = rerank(results, q)
        for i, r in enumerate(reranked[:5]):
            meta = r.get("metadata") or {}
            fp = meta.get("file_path") or ""
            para = meta.get("paragraph_name") or ""
            score = r.get("score", 0)
            # Normalize to relative for display
            if "data/gnucobol-contrib" in fp:
                fp = "data/gnucobol-contrib" + fp.split("data/gnucobol-contrib", 1)[-1]
            print(f"  {i+1}. {fp} | {para} | score={score}")


if __name__ == "__main__":
    main()
