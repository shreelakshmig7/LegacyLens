#!/usr/bin/env bash
# Run Streamlit UI with PYTHONPATH set so 'legacylens' is importable.
# Usage: from project root, run:  bash scripts/run_streamlit.sh
# Or:    source .venv/bin/activate && PYTHONPATH=. streamlit run legacylens/ui/app.py

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
if [ -d .venv ]; then
  source .venv/bin/activate
fi
exec streamlit run legacylens/ui/app.py "$@"
