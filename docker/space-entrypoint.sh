#!/bin/sh
# API in the background, Streamlit in the foreground (the Space's public port).
set -eu
rbirag serve --host 127.0.0.1 --port 8000 &
api=$!
trap 'kill "$api" 2>/dev/null || true' EXIT INT TERM
i=0
until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" 2>/dev/null; do
  i=$((i + 1))
  if [ "$i" -gt 120 ]; then echo "API did not start" >&2; exit 1; fi
  if ! kill -0 "$api" 2>/dev/null; then echo "API exited" >&2; exit 1; fi
  sleep 1
done
exec streamlit run ui/streamlit_app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0
