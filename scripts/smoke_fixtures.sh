#!/usr/bin/env bash
# End-to-end smoke on the HTML fixtures, fully offline:
# ingest -> chunk (fixed | semantic | section) -> index (BM25 + embedded Qdrant, fake embedder)
# -> query in every retrieval mode.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$ROOT/.smoke}"
rm -rf "$WORK"
mkdir -p "$WORK/raw/html"
for f in "$ROOT"/tests/fixtures/html/notification_*.html; do
  id="$(basename "$f" .html | sed -E 's/^notification_([0-9]+).*/\1/')"
  [[ "$id" =~ ^[0-9]+$ ]] && cp "$f" "$WORK/raw/html/$id.html"
done

export RBIRAG_RAW_DIR="$WORK/raw"
export RBIRAG_PROCESSED_DIR="$WORK/processed"
export RBIRAG_INDEX_DIR="$WORK/indexes"
export RBIRAG_EMBEDDER=fake
export RBIRAG_RERANKER=fake
export RBIRAG_CHUNK_SIZE=128
export RBIRAG_CHUNK_OVERLAP=16
export RBIRAG_SECTION_MIN_TOKENS=24
export RBIRAG_SEMANTIC_MIN_TOKENS=24

rbirag ingest --stats-out "$WORK/stats.json"
for chunker in fixed semantic section; do
  rbirag index --chunker "$chunker"
  for mode in bm25 dense hybrid hybrid_rerank; do
    echo "=== chunker=$chunker mode=$mode"
    rbirag query "How often must KYC be updated for high risk customers?" \
      --chunker "$chunker" --mode "$mode" --top-k 3
  done
done
echo "smoke OK"
