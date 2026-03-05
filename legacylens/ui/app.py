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
import html
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import streamlit as st
import streamlit.components.v1 as components

# Load .env from project root so LEGACYLENS_API_URL (e.g. http://localhost:8001) is set
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_root / ".env")
except ImportError:
    pass

from legacylens.config.constants import FORTRAN_EXTENSIONS, LEGACYLENS_API_URL

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
KEY_SEARCH_RAN_THIS_RUN = "_search_ran_this_run"
KEY_SKIP_NEXT_LAST_RESULT = "_skip_next_last_result"
KEY_SELECTED_CHUNK_INDEX = "_selected_chunk_index"  # Pre-Search 12: user selects most relevant
KEY_EVAL_RUNNING = "_eval_running"


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
    if KEY_SELECTED_CHUNK_INDEX not in st.session_state:
        st.session_state[KEY_SELECTED_CHUNK_INDEX] = None
    if KEY_EVAL_RUNNING not in st.session_state:
        st.session_state[KEY_EVAL_RUNNING] = False
    # Apply pending example before text_input is drawn (Streamlit forbids modifying query_input after widget creation)
    if KEY_PENDING_EXAMPLE in st.session_state:
        st.session_state[KEY_QUERY_INPUT] = st.session_state.pop(KEY_PENDING_EXAMPLE)
        st.session_state[KEY_RUN_SEARCH] = True


def _project_root() -> Path:
    """Return project root (parent of legacylens package)."""
    return Path(__file__).resolve().parent.parent.parent


def _find_latest_eval_file() -> Optional[Path]:
    """
    Return path to the latest full 20-case eval result file.

    Prefers the path in tests/results/.latest_eval (written by run_eval when
    it finishes) so the UI shows the just-run results after run and after reload.
    Falls back to newest by mtime among eval_* files with "Total: 20".
    """
    results_dir = _project_root() / "tests" / "results"
    if not results_dir.exists():
        return None

    marker = results_dir / ".latest_eval"
    if marker.exists():
        try:
            name = marker.read_text(encoding="utf-8").strip()
            if name and not Path(name).is_absolute():
                candidate = results_dir / name
                if candidate.exists():
                    head = candidate.read_text(encoding="utf-8", errors="ignore")[:500]
                    if "Total: 20" in head:
                        return candidate
        except Exception:
            pass

    pattern = str(results_dir / "eval_*.txt")
    candidates: List[Tuple[float, Path]] = []
    for f in glob.glob(pattern):
        p = Path(f)
        if p.name[len("eval_"):len("eval_") + 1].isdigit():
            try:
                head = p.read_text(encoding="utf-8", errors="ignore")[:500]
                if "Total: 20" in head:
                    candidates.append((p.stat().st_mtime, p))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _parse_eval_file(path: Path) -> Dict[str, Any]:
    """
    Parse eval file for retrieval precision, answer faithfulness, and compact summary.

    Args:
        path: Path to the eval result .txt file.

    Returns:
        dict with keys: timestamp, retrieval_fraction, answer_fraction, summary_line, raw_lines.
    """
    out: Dict[str, Any] = {
        "timestamp": "",
        "retrieval_fraction": "",
        "answer_fraction": "",
        "summary_line": "",
        "raw_lines": [],
    }
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return out
    lines = text.strip().split("\n")
    out["raw_lines"] = lines
    if not lines:
        return out
    # First line: "LegacyLens Eval — 20260304T011012Z"
    first = lines[0]
    if "\u2014" in first:
        out["timestamp"] = first.split("\u2014")[-1].strip()
    for line in lines[1:]:
        # "Retrieval Precision: 12/20 (60.0%) — target >70%"
        if "Retrieval Precision:" in line:
            parts = line.split("Retrieval Precision:")
            fraction = parts[1].strip().split()[0] if len(parts) > 1 else ""
            out["retrieval_fraction"] = fraction
        elif "Answer Faithfulness:" in line:
            parts = line.split("Answer Faithfulness:")
            fraction = parts[1].strip().split()[0] if len(parts) > 1 else ""
            out["answer_fraction"] = fraction
    if out["retrieval_fraction"] or out["answer_fraction"]:
        out["summary_line"] = f"Retrieval: {out['retrieval_fraction']} · Answer: {out['answer_fraction']}"
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
            [sys.executable, str(script)],
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
    """
    Sidebar eval section: one-line summary from latest eval file and Run Evals button.

    Button is disabled while eval is running to prevent double-clicks.
    Status messages stay in the sidebar; main area is not affected.
    """
    latest = _find_latest_eval_file()
    if latest:
        parsed = _parse_eval_file(latest)
        if parsed["summary_line"]:
            st.sidebar.caption(parsed["summary_line"])

    is_running = st.session_state.get(KEY_EVAL_RUNNING, False)
    clicked = st.sidebar.button("Run Evals", key="run_eval_btn", disabled=is_running)

    if clicked and not is_running:
        st.session_state[KEY_EVAL_RUNNING] = True
        st.rerun()

    if is_running:
        eval_script = _project_root() / "eval" / "run_eval.py"
        status = st.sidebar.empty()
        if not eval_script.exists():
            st.session_state[KEY_EVAL_RUNNING] = False
            status.error("Eval script not found")
        else:
            status.info("Running eval…")
            code, out = _run_eval_fast()
            st.session_state[KEY_EVAL_RUNNING] = False
            if code == 0:
                status.success("Done. Reload to refresh summary.")
            else:
                status.error(f"Eval failed: {out[:300]}")
            st.rerun()


def _infer_code_language(file_path: str) -> str:
    """
    Infer syntax highlighting language from file extension (PRD 9.2: COBOL or Fortran).

    Args:
        file_path: Relative or absolute path to source file.

    Returns:
        str: "fortran" for .f/.f90/.for, else "cobol".
    """
    if not file_path or not isinstance(file_path, str):
        return "cobol"
    ext = Path(file_path).suffix.lower()
    if ext in FORTRAN_EXTENSIONS:
        return "fortran"
    return "cobol"


def _fetch_file_content(base_url: str, path: str) -> Tuple[bool, str, Optional[str]]:
    """
    Fetch full file content from GET /file/content.

    Args:
        base_url: API base URL (e.g. http://localhost:8000).
        path: Relative path to file (e.g. data/gnucobol-contrib/.../file.cbl).

    Returns:
        (success, content_or_error, error_msg)
    """
    if not path or not path.strip():
        return False, "", "No file path"
    url = f"{base_url.rstrip('/')}/file/content"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, params={"path": path})
        if resp.status_code != 200:
            try:
                err = resp.json().get("error", resp.text)
            except Exception:
                err = resp.text or f"HTTP {resp.status_code}"
            return False, "", err
        data = resp.json()
        if not data.get("success"):
            return False, "", data.get("error", "Unknown error")
        return True, data.get("content", ""), None
    except httpx.HTTPError as e:
        return False, "", str(e)
    except Exception as e:
        logger.exception("Failed to fetch file content for %s", path)
        return False, "", str(e)


def _render_full_file_with_highlight(
    content: str,
    highlight_start: int,
    highlight_end: int,
) -> str:
    """
    Build HTML for full file view with line numbers and highlighted lines (PRD 9.3).

    Args:
        content: Full file content.
        highlight_start: Start line of retrieved chunk (1-based).
        highlight_end: End line of retrieved chunk (1-based).

    Returns:
        str: HTML string for st.components.v1.html().
    """
    lines = content.splitlines()
    rows = []
    for i, line in enumerate(lines, start=1):
        escaped = html.escape(line).replace(" ", "&nbsp;")
        cls = "highlight" if highlight_start <= i <= highlight_end else ""
        rows.append(
            f'<tr><td class="ln">{i}</td><td class="{cls}">{escaped}</td></tr>'
        )
    table_body = "\n".join(rows)
    return f"""
<style>
  .file-view {{ font-family: monospace; font-size: 12px; max-height: 400px; overflow: auto; }}
  .file-view table {{ border-collapse: collapse; width: 100%; }}
  .file-view td {{ padding: 0 0.5em 0 0; vertical-align: top; }}
  /* Light mode (default) */
  .file-view {{ color: #000; background: #fff; }}
  .file-view td.ln {{ color: #666; user-select: none; text-align: right; min-width: 3em; }}
  .file-view td.highlight {{ background-color: #fff3cd; }}
  /* Dark mode: white text, dark background */
  @media (prefers-color-scheme: dark) {{
    .file-view {{ color: #fff; background: #262730; }}
    .file-view td.ln {{ color: #9ca3af; }}
    .file-view td.highlight {{ background-color: #374151; }}
  }}
</style>
<div class="file-view">
<table><tbody>
{table_body}
</tbody></table>
</div>
"""


def _render_chunks(metadata: Dict[str, Any], base_url: Optional[str] = None) -> None:
    """
    For each chunk: snippet, caption, GitHub link, relevance; expandable full file view (PRD 9.3).
    Pre-Search 12: "Select as most relevant" button per chunk.
    PRD 9.2: Syntax highlighting by file extension (COBOL or Fortran).
    """
    chunks = metadata.get("chunks") or []
    file_paths = metadata.get("file_paths") or []
    line_numbers = metadata.get("line_numbers") or []
    line_ranges = metadata.get("line_ranges") or []
    github_links = metadata.get("github_links") or []
    relevance_scores = metadata.get("relevance_scores") or []
    api_url = base_url or LEGACYLENS_API_URL
    selected_idx = st.session_state.get(KEY_SELECTED_CHUNK_INDEX)
    n = max(
        len(chunks),
        len(file_paths),
        len(line_numbers),
        len(github_links),
        len(relevance_scores),
    )
    # Pre-Search 12: Show selected chunk first when user has chosen one
    order = list(range(n))
    if selected_idx is not None and 0 <= selected_idx < n:
        order = [selected_idx] + [j for j in range(n) if j != selected_idx]
    for idx in order:
        i = idx
        chunk_text = chunks[i] if i < len(chunks) else ""
        fp = file_paths[i] if i < len(file_paths) else ""
        line_num = line_numbers[i] if i < len(line_numbers) else 0
        lr = line_ranges[i] if i < len(line_ranges) else [line_num, line_num]
        link = github_links[i] if i < len(github_links) else ""
        score = relevance_scores[i] if i < len(relevance_scores) else 0.0
        highlight_start = lr[0] if len(lr) >= 1 else line_num
        highlight_end = lr[1] if len(lr) >= 2 else line_num
        lang = _infer_code_language(fp)

        # Pre-Search 12: "Select as most relevant" — allow user to pick which chunk is most relevant
        is_selected = selected_idx == i
        btn_col, _ = st.columns([1, 4])
        with btn_col:
            if is_selected:
                if st.button("✓ Selected — Clear", key=f"select_chunk_{i}"):
                    st.session_state[KEY_SELECTED_CHUNK_INDEX] = None
                    st.rerun()
            else:
                if st.button("Select as most relevant", key=f"select_chunk_{i}"):
                    st.session_state[KEY_SELECTED_CHUNK_INDEX] = i
                    st.rerun()

        st.code(chunk_text, language=lang)
        st.caption(f"**File:** `{fp}` · **Line:** {line_num}")
        if link:
            st.markdown(f"[View on GitHub]({link})")
        st.metric("Relevance", f"{score:.2f}")

        # PRD 9.3: Expand to view full file with retrieved lines highlighted
        with st.expander("Show full file", expanded=False):
            if fp:
                ok, content, err = _fetch_file_content(api_url, fp)
                if ok:
                    html_frag = _render_full_file_with_highlight(
                        content, highlight_start, highlight_end
                    )
                    components.html(html_frag, height=420, scrolling=True)
                else:
                    st.error(err or "Could not load file")
            else:
                st.caption("No file path available")

        st.divider()


def _has_retrieved_chunks(metadata: Optional[Dict[str, Any]]) -> bool:
    """
    Return True when metadata contains at least one non-empty retrieved chunk.

    Args:
        metadata: Stream/API metadata dict that may include a "chunks" list.

    Returns:
        bool: True if at least one chunk has non-whitespace text.
    """
    if not metadata:
        return False
    chunks = metadata.get("chunks") or []
    if not isinstance(chunks, list):
        return False
    return any(isinstance(chunk, str) and chunk.strip() for chunk in chunks)


def _is_not_found_answer(answer_text: Optional[str]) -> bool:
    """
    Return True when the answer indicates nothing was found or context doesn't answer the question.

    Used to hide the "Retrieved chunks" section when the answer is a not-found /
    out-of-scope / context-doesn't-answer style response (matches answer_generator templates).

    Args:
        answer_text: The generated answer string.

    Returns:
        bool: True if the answer is a not-found style response.
    """
    if not answer_text or not isinstance(answer_text, str):
        return False
    lower = answer_text.lower()
    if "not found" not in lower:
        return False
    markers = (
        "indexed codebase",
        "outside the scope",
        "requested information",
        "retrieved context",
        "no matching paragraph",
        "retrieved chunks",
    )
    return any(m in lower for m in markers)


def main() -> None:
    """Build Streamlit layout: sidebar (About, Eval), main (query, examples, results)."""
    st.set_page_config(
        page_title="LegacyLens — COBOL Codebase Explorer",
        layout="wide",
    )
    st.markdown(
        "<style>[data-testid='stStatusWidget']{display:none!important}</style>",
        unsafe_allow_html=True,
    )
    _init_session_state()

    st.title("LegacyLens — COBOL Codebase Explorer")
    st.caption("Ask natural-language questions about your legacy COBOL codebase.")

    # Sidebar
    _render_sidebar_about()
    st.sidebar.markdown("---")
    _render_sidebar_eval()

    # Main: query input inside a form so Enter key submits (PRD 9.1: button + keyboard shortcut)
    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "Ask about the codebase",
            key=KEY_QUERY_INPUT,
            placeholder="e.g. Where is the main entry point of this program?",
            label_visibility="visible",
        )
        search_clicked = st.form_submit_button("Search")
    st.caption("Press **Enter** to search")
    if search_clicked:
        st.session_state[KEY_LAST_METADATA] = None
        st.session_state[KEY_LAST_ANSWER] = ""
        st.session_state[KEY_STREAM_ERROR] = None
        st.session_state[KEY_SELECTED_CHUNK_INDEX] = None
        st.session_state[KEY_RUN_SEARCH] = True
        st.rerun()

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

    # Run search when Search clicked or example triggered
    if st.session_state.get(KEY_RUN_SEARCH) and (query or st.session_state.get(KEY_QUERY_INPUT)):
        st.session_state[KEY_SEARCH_RAN_THIS_RUN] = True
        st.session_state[KEY_SKIP_NEXT_LAST_RESULT] = True
        st.session_state[KEY_LAST_METADATA] = None
        st.session_state[KEY_LAST_ANSWER] = None
        st.session_state[KEY_STREAM_ERROR] = None
        st.session_state[KEY_SELECTED_CHUNK_INDEX] = None
        q = query or st.session_state.get(KEY_QUERY_INPUT, "")
        st.session_state[KEY_RUN_SEARCH] = False
        if not q.strip():
            st.warning("Enter a query first.")
        else:
            base_url = LEGACYLENS_API_URL

            # Reserve result area so previous answer and chunks are cleared immediately (Streamlit keeps old content until we write)
            result_placeholder = st.empty()
            chunks_placeholder = st.empty()
            with result_placeholder.container():
                st.subheader("Answer")
                st.caption("Searching and analyzing codebase...")
            chunks_placeholder.empty()

            with st.spinner("Searching and analyzing codebase..."):
                metadata, answer_text, error_msg = _stream_query_stream(
                    base_url, q,
                )

            if error_msg:
                st.session_state[KEY_STREAM_ERROR] = f"Unexpected error: {error_msg}"
                result_placeholder.empty()
                chunks_placeholder.empty()
                st.error(st.session_state[KEY_STREAM_ERROR])
            else:
                st.session_state[KEY_LAST_ANSWER] = answer_text
                show_chunks = (
                    metadata
                    and _has_retrieved_chunks(metadata)
                    and not _is_not_found_answer(answer_text)
                )
                if show_chunks:
                    st.session_state[KEY_LAST_METADATA] = metadata
                else:
                    st.session_state[KEY_LAST_METADATA] = None

                with result_placeholder.container():
                    st.subheader("Answer")
                    st.markdown(answer_text or "(No answer returned.)")
                if show_chunks:
                    with chunks_placeholder.container():
                        with st.expander("Retrieved chunks", expanded=False):
                            _render_chunks(metadata, base_url=LEGACYLENS_API_URL)
                else:
                    chunks_placeholder.empty()

    # Show last result when we didn't run a search this run; skip one run after a search to avoid duplicate
    skip_next = st.session_state.pop(KEY_SKIP_NEXT_LAST_RESULT, None)
    if skip_next:
        pass
    elif st.session_state.get(KEY_STREAM_ERROR):
        st.error(st.session_state[KEY_STREAM_ERROR])
    elif st.session_state.get(KEY_LAST_ANSWER):
        st.subheader("Answer")
        st.markdown(st.session_state[KEY_LAST_ANSWER])
        last_answer = st.session_state.get(KEY_LAST_ANSWER) or ""
        if (
            st.session_state.get(KEY_LAST_METADATA)
            and not _is_not_found_answer(last_answer)
        ):
            with st.expander("Retrieved chunks", expanded=False):
                _render_chunks(st.session_state[KEY_LAST_METADATA], base_url=LEGACYLENS_API_URL)


if __name__ == "__main__":
    main()
