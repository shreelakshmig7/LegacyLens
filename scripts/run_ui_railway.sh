#!/usr/bin/env sh
# LegacyLens — Run Streamlit UI for Railway (bind 0.0.0.0 and PORT to avoid 502).
# Use this as the Start Command for the legacylens-ui service in Railway.
# Running via a shell guarantees $PORT is expanded (Railway may not expand it in raw start commands).
set -e
PORT="${PORT:-8501}"
echo "Starting Streamlit on 0.0.0.0:${PORT}"
exec streamlit run legacylens/ui/app.py --server.port="$PORT" --server.address=0.0.0.0
