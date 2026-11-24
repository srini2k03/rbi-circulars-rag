#!/usr/bin/env bash
# Commit a CI job's outputs onto the *latest* main, retrying when another run pushed first.
#
#   scripts/commit_outputs.sh MESSAGE SRC_DIR:DEST_DIR [SRC_DIR:DEST_DIR ...] [--log NEW_LINES.jsonl]
#
# Each attempt: fetch origin/main, reset to it, replace every DEST_DIR's files with SRC_DIR's
# (other files in DEST_DIR are kept), append the lines of NEW_LINES.jsonl to
# results/experiments.jsonl (an append-only log, so concurrent runs never conflict), commit
# and push. A rejected push repeats the whole loop; nothing is merged as text.
set -euo pipefail

MESSAGE=$1; shift
REMOTE=${PUBLISH_REMOTE:-origin}
BRANCH=${PUBLISH_BRANCH:-main}
ATTEMPTS=${PUBLISH_ATTEMPTS:-6}
PAIRS=()
LOG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --log) LOG=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2 ;;
    *) src=${1%%:*}; dest=${1#*:}; PAIRS+=("$(cd "$src" && pwd):$dest"); shift ;;
  esac
done

git config user.name >/dev/null || git config user.name "github-actions[bot]"
git config user.email >/dev/null || git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for attempt in $(seq 1 "$ATTEMPTS"); do
  git fetch --quiet "$REMOTE" "$BRANCH"
  git reset --quiet --hard "$REMOTE/$BRANCH"
  for pair in "${PAIRS[@]}"; do
    src=${pair%%:*}; dest=${pair#*:}
    mkdir -p "$dest"
    cp -R "$src/." "$dest/"
    git add -A "$dest"
  done
  if [ -n "$LOG" ] && [ -s "$LOG" ]; then
    mkdir -p results
    cat "$LOG" >> results/experiments.jsonl
    git add results/experiments.jsonl
  fi
  if git diff --cached --quiet; then
    echo "nothing to commit"
    exit 0
  fi
  git commit --quiet -m "$MESSAGE"
  if git push --quiet "$REMOTE" "HEAD:$BRANCH"; then
    echo "pushed on attempt ${attempt}"
    exit 0
  fi
  echo "push rejected (attempt ${attempt}); refetching"
  sleep $((attempt * ${PUBLISH_BACKOFF_SECONDS:-10}))
done
echo "::error::could not push after ${ATTEMPTS} attempts"
exit 1
