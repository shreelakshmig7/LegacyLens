"""
app.py
------
LegacyLens — RAG System for Legacy Enterprise Codebases — Streamlit UI
---------------------------------------------------------------------
Streamlit interface: query input, call /query/stream, display status, streamed
answer, and retrieved chunks with syntax highlighting, file path, line number,
GitHub deep link, and relevance score. Sidebar: About, Evaluation panel.

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import glob
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import streamlit as st

# Load .env from project root so LEGACYLENS_API_URL (e.g. http://localhost:8001) is set
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_root / ".env")
except ImportError:
    pass

from legacylens.config.constants import LEGACYLENS_API_URL

logger = logging.getLogger(__name__)

# Example questions: 6 buttons in 2 rows of 3
EXAMPLE_QUESTIONS: List[str] = [
    "Where is the main entry point?",
    "What functions modify CUSTOMER-RECORD?",
    "Find all file I/O operations",
    "Show error handling patterns",
    "What are the dependencies of PGMOD1?",
    "Explain what SQL-ERROR paragraph does",
]

# Session state keys
KEY_QUERY_INPUT = "query_input"
KEY_RUN_SEARCH = "run_search"
KEY_PENDING_EXAMPLE = "pending_example_query"  # Set by example buttons; applied before text_input is drawn
KEY_LAST_METADATA = "last_metadata"
KEY_LAST_ANSWER = "last_answer"
KEY_STREAM_ERROR = "stream_error"


def _init_session_state() -> None:
    """Initialize session state keys if missing. Apply pending example query before any widget uses query_input."""
    if KEY_QUERY_INPUT not in st.session_state:
        st.session_state[KEY_QUERY_INPUT] = ""
    if KEY_RUN_SEARCH not in st.session_state:
        st.session_state[KEY_RUN_SEARCH] = False
    if KEY_LAST_METADATA not in st.session_state:
        st.session_state[KEY_LAST_METADATA] = None
    if KEY_LAST_ANSWER not in st.session_state:
        st.session_state[KEY_LAST_ANSWER] = ""
    if KEY_STREAM_ERROR not in st.session_state:
        st.session_state[KEY_STREAM_ERROR] = None
    # Apply pending example before text_input is drawn (Streamlit forbids modifying query_input after widget creation)
    if KEY_PENDING_EXAMPLE in st.session_state:
        st.session_state[KEY_QUERY_INPUT] = st.session_state.pop(KEY_PENDING_EXAMPLE)
        st.session_state[KEY_RUN_SEARCH] = True


def _project_root() -> Path:
    """Return project root (parent of legacylens package)."""
    return Path(__file__).resolve().parent.parent.parent


def _find_latest_eval_file() -> Optional[Path]:
    """Return path to the latest tests/results/eval_*.txt by timestamp in filename."""
    results_dir = _project_root() / "tests" / "results"
    if not results_dir.exists():
        return None
    pattern = str(results_dir / "eval_*.txt")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by filename (eval_YYYYMMDDTHHMMSSZ.txt) descending
    files.sort(reverse=True)
    return Path(files[0])


def _parse_eval_file(path: Path) -> Dict[str, Any]:
    """Parse eval file for retrieval precision, answer faithfulness, timestamp."""
    out: Dict[str, Any] = {
        "timestamp": "",
        "retrieval_precision": "",
        "answer_faithfulness": "",
        "raw_lines": [],
    }
    try:
        text = path.read_text()
    except Exception:
        return out
    lines = text.strip().split("\n")
    out["raw_lines"] = lines
    if not lines:
        return out
    # First line: "LegacyLens Eval — 20260304T011012Z"
    first = lines[0]
    if "—" in first:
        out["timestamp"] = first.split("—")[-1].strip()
    for line in lines[1:]:
        if "Retrieval Precision:" in line:
            out["retrieval_precision"] = line.strip()
        elif "Answer Faithfulness:" in line:
            out["answer_faithfulness"] = line.strip()
    return out


def _stream_query_stream(
    base_url: str,
    query: str,
    status_placeholder: Optional[Any] = None,
    answer_placeholder: Optional[Any] = None,
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """
    Call POST /query/stream; optionally update status_placeholder and answer_placeholder as data arrives.

    First line of response is JSON (metadata). Remaining content may contain
    __STATUS__..., __ERROR__..., or plain answer tokens.
    Returns (metadata_dict, answer_text, error_message).
    """
    url = f"{base_url.rstrip('/')}/query/stream"
    metadata: Optional[Dict[str, Any]] = None
    answer_parts: List[str] = []
    error_msg: Optional[str] = None
    buffer = ""
    first_line_done = False

    def update_status(msg: str) -> None:
        if status_placeholder is not None and msg:
            status_placeholder.info(msg)

    def update_answer(text: str) -> None:
        if answer_placeholder is not None:
            answer_placeholder.markdown(text)

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, json={"query": query}) as response:
                if response.status_code != 200:
                    if response.status_code == 404:
                        error_msg = (
                            "Backend not found (404). Is the API running? "
                            "Run: uvicorn legacylens.api.main:app --port <PORT> (use a free port, e.g. 8002). "
                            "Set LEGACYLENS_API_URL to that API URL (e.g. http://localhost:8002), not the Streamlit URL."
                        )
                    else:
                        try:
                            body = response.read().decode()
                            err_obj = json.loads(body)
                            error_msg = err_obj.get("error", body)
                        except Exception:
                            error_msg = f"HTTP {response.status_code}"
                    return None, "", error_msg
                for chunk in response.iter_text():
                    if chunk:
                        buffer += chunk
                    # First line = JSON metadata
                    if not first_line_done and "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        first_line_done = True
                        try:
                            metadata = json.loads(line)
                        except json.JSONDecodeError:
                            pass
                    if not first_line_done:
                        continue
                    # Process buffer for __STATUS__, __ERROR__, or answer
                    while True:
                        if "__ERROR__" in buffer:
                            idx = buffer.find("__ERROR__")
                            error_msg = buffer[idx + 9 :].strip()
                            buffer = buffer[:idx]
                            break
                        if "__STATUS__" in buffer:
                            idx = buffer.find("__STATUS__")
                            end = idx + 9
                            rest = buffer[end:]
                            msg_end = len(rest)
                            for m in ["__STATUS__", "__ERROR__"]:
                                if m in rest:
                                    msg_end = rest.find(m)
                                    break
                            status_msg = rest[:msg_end].strip()
                            if status_msg:
                                update_status(status_msg)
                            # Do not add status message to answer text
                            buffer = buffer[:idx] + rest[msg_end:]
                            continue
                        break
                    answer_parts.append(buffer)
                    current = "".join(answer_parts).replace("__STATUS__", "").strip()
                    if current and not error_msg:
                        update_answer(current)
                    buffer = ""
                if buffer:
                    answer_parts.append(buffer)
                    current = "".join(answer_parts).replace("__STATUS__", "").strip()
                    if current and not error_msg:
                        update_answer(current)
                if "__ERROR__" in "".join(answer_parts):
                    idx = "".join(answer_parts).find("__ERROR__")
                    error_msg = "".join(answer_parts)[idx + 9 :].strip()
                    answer_parts = []
    except httpx.HTTPError as e:
        error_msg = f"Request failed: {e}"
    except Exception as e:
        logger.exception("stream request failed")
        error_msg = str(e)

    answer_text = "".join(answer_parts).replace("__STATUS__", "").strip()
    return metadata, answer_text, error_msg


def _run_eval_fast() -> Tuple[int, str]:
    """Run python eval/run_eval.py --fast as subprocess. Returns (returncode, stderr+stdout)."""
    project_root = _project_root()
    script = project_root / "eval" / "run_eval.py"
    if not script.exists():
        return -1, f"Script not found: {script}"
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--fast"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = (result.stderr or "") + (result.stdout or "")
        return result.returncode, out
    except subprocess.TimeoutExpired:
        return -1, "Eval timed out (5 min)."
    except Exception as e:
        return -1, str(e)


def _render_sidebar_about() -> None:
    """Sidebar 'About' section: description and usage tips."""
    st.sidebar.header("About")
    st.sidebar.markdown(
        "**LegacyLens** lets you ask natural-language questions about a legacy COBOL codebase. "
        "It uses retrieval (embeddings + reranking) and an LLM to return grounded answers with code snippets and GitHub links."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**What you can ask:**")
    st.sidebar.markdown(
        """
• Explain what a paragraph does  
• Find dependencies of a module  
• Locate file I/O operations  
• Find error handling patterns  
• Search for business logic  
• Find entry points  
"""
    )


def _render_sidebar_eval() -> None:
    """Sidebar 'Evaluation' section: latest eval file, Run Eval (Fast) button, warning."""
    st.sidebar.header("Evaluation")
    latest = _find_latest_eval_file()
    if latest:
        parsed = _parse_eval_file(latest)
        if parsed["timestamp"]:
            st.sidebar.caption(f"Latest: {parsed['timestamp']}")
        if parsed["retrieval_precision"]:
            st.sidebar.text(parsed["retrieval_precision"])
        if parsed["answer_faithfulness"]:
            st.sidebar.text(parsed["answer_faithfulness"])
    else:
        st.sidebar.caption("No eval results yet. Run eval below.")
    st.sidebar.warning("Running eval makes live API calls and may take 2-3 minutes.")
    if st.sidebar.button("Run Eval (Fast)", key="run_eval_btn"):
        eval_script = _project_root() / "eval" / "run_eval.py"
        if not eval_script.exists():
            st.sidebar.error("Eval not available in this deployment")
        else:
            with st.sidebar.spinner("Running eval..."):
                code, out = _run_eval_fast()
            if code == 0:
                st.sidebar.success("Eval completed. Refresh to see latest results.")
            else:
                st.sidebar.error(f"Eval failed: {out[:500]}")


def _render_chunks(metadata: Dict[str, Any]) -> None:
    """For each chunk show st.code (COBOL), caption (file + line), GitHub link, relevance."""
    chunks = metadata.get("chunks") or []
    file_paths = metadata.get("file_paths") or []
    line_numbers = metadata.get("line_numbers") or []
    github_links = metadata.get("github_links") or []
    relevance_scores = metadata.get("relevance_scores") or []
    n = max(len(chunks), len(file_paths), len(line_numbers), len(github_links), len(relevance_scores))
    for i in range(n):
        chunk_text = chunks[i] if i < len(chunks) else ""
        fp = file_paths[i] if i < len(file_paths) else ""
        line_num = line_numbers[i] if i < len(line_numbers) else 0
        link = github_links[i] if i < len(github_links) else ""
        score = relevance_scores[i] if i < len(relevance_scores) else 0.0
        st.code(chunk_text, language="cobol")
        st.caption(f"**File:** `{fp}` · **Line:** {line_num}")
        if link:
            st.markdown(f"[View on GitHub]({link})")
        st.metric("Relevance", f"{score:.2f}")
        st.divider()


def main() -> None:
    """Build Streamlit layout: sidebar (About, Eval), main (query, examples, results)."""
    st.set_page_config(
        page_title="LegacyLens — COBOL Codebase Explorer",
        layout="wide",
    )
    _init_session_state()

    st.title("LegacyLens — COBOL Codebase Explorer")
    st.caption("Ask natural-language questions about your legacy COBOL codebase.")

    # Sidebar
    _render_sidebar_about()
    st.sidebar.markdown("---")
    _render_sidebar_eval()

    # Main: query input
    query = st.text_input(
        "Ask about the codebase",
        key=KEY_QUERY_INPUT,
        placeholder="e.g. Where is the main entry point of this program?",
    )
    # Do not assign to st.session_state[KEY_QUERY_INPUT] after the widget — Streamlit owns it when key= is set.

    # Example questions: 2 rows of 3
    st.markdown("**Try these examples:**")
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        if st.button(EXAMPLE_QUESTIONS[0], key="ex0"):
            st.session_state[KEY_PENDING_EXAMPLE] = EXAMPLE_QUESTIONS[0]
            st.rerun()
    with r1c2:
        if st.button(EXAMPLE_QUESTIONS[1], key="ex1"):
            st.session_state[KEY_PENDING_EXAMPLE] = EXAMPLE_QUESTIONS[1]
            st.rerun()
    with r1c3:
        if st.button(EXAMPLE_QUESTIONS[2], key="ex2"):
            st.session_state[KEY_PENDING_EXAMPLE] = EXAMPLE_QUESTIONS[2]
            st.rerun()
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        if st.button(EXAMPLE_QUESTIONS[3], key="ex3"):
            st.session_state[KEY_PENDING_EXAMPLE] = EXAMPLE_QUESTIONS[3]
            st.rerun()
    with r2c2:
        if st.button(EXAMPLE_QUESTIONS[4], key="ex4"):
            st.session_state[KEY_PENDING_EXAMPLE] = EXAMPLE_QUESTIONS[4]
            st.rerun()
    with r2c3:
        if st.button(EXAMPLE_QUESTIONS[5], key="ex5"):
            st.session_state[KEY_PENDING_EXAMPLE] = EXAMPLE_QUESTIONS[5]
            st.rerun()

    search_clicked = st.button("Search", key="search_btn")
    if search_clicked:
        st.session_state[KEY_RUN_SEARCH] = True

    # Run search when Search clicked or example triggered
    if st.session_state.get(KEY_RUN_SEARCH) and (query or st.session_state.get(KEY_QUERY_INPUT)):
        q = query or st.session_state.get(KEY_QUERY_INPUT, "")
        st.session_state[KEY_RUN_SEARCH] = False
        if not q.strip():
            st.warning("Enter a query first.")
        else:
            base_url = LEGACYLENS_API_URL
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            chunks_placeholder = st.container()

            # Stream with status updates (status and answer update as data arrives)
            status_placeholder.info("Connecting...")
            metadata, answer_text, error_msg = _stream_query_stream(
                base_url, q,
                status_placeholder=status_placeholder,
                answer_placeholder=answer_placeholder,
            )

            if error_msg:
                status_placeholder.empty()
                st.error(error_msg)
            else:
                status_placeholder.empty()
                if metadata:
                    st.subheader("Answer")
                    answer_placeholder.markdown(answer_text)
                    st.session_state[KEY_LAST_METADATA] = metadata
                    st.session_state[KEY_LAST_ANSWER] = answer_text
                    st.subheader("Retrieved chunks")
                    _render_chunks(metadata)
                else:
                    answer_placeholder.markdown(answer_text or "(No answer returned.)")
                    st.session_state[KEY_LAST_ANSWER] = answer_text
                    st.session_state[KEY_LAST_METADATA] = None

    # Show last result on rerun when no new search (e.g. after eval refresh)
    if not st.session_state.get(KEY_RUN_SEARCH) and st.session_state.get(KEY_LAST_METADATA):
        st.subheader("Answer")
        st.markdown(st.session_state.get(KEY_LAST_ANSWER, ""))
        st.subheader("Retrieved chunks")
        _render_chunks(st.session_state[KEY_LAST_METADATA])


if __name__ == "__main__":
    main()
