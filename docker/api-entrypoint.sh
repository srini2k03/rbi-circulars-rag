#!/bin/sh
# Optionally build a tiny index from the bundled HTML fixtures (for smoke tests and
# first-run demos), then start the API. Otherwise POST /ingest builds the index.
set -eu
INDEX_DIR="${RBIRAG_INDEX_DIR:-/data/indexes}/${RBIRAG_CHUNKER:-section}"
if [ "${RBIRAG_BOOTSTRAP_FIXTURES:-0}" = "1" ] && [ ! -f "${INDEX_DIR}/chunks.jsonl" ]; then
  echo "bootstrapping a fixture index into ${INDEX_DIR}"
  mkdir -p "${RBIRAG_RAW_DIR:-/data/raw}/html"
  for f in /app/fixtures/html/notification_1*.html; do
    id=$(basename "$f" .html | sed -E 's/^notification_([0-9]+).*/\1/')
    case "$id" in *[!0-9]*) ;; *) cp "$f" "${RBIRAG_RAW_DIR:-/data/raw}/html/${id}.html" ;; esac
  done
  rbirag ingest
  rbirag index
fi
exec rbirag serve --host 0.0.0.0 --port "${RBIRAG_API_PORT:-8000}"
