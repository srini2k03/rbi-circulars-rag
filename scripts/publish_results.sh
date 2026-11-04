#!/usr/bin/env bash
# Publish one evaluate.yml run's outputs onto the *latest* main without textual merges.
#
#   scripts/publish_results.sh RUN_DIR
#
# RUN_DIR layout (as downloaded by the report job):
#   RUN_DIR/plan/...           this run's plan + calibration (copied into results/)
#   RUN_DIR/retrieval/...      fresh retrieval outputs, if this run re-ran retrieval
#   RUN_DIR/generation/**      record/status/gold-check files from every generate/judge job
#
# Each attempt: fetch origin/main, reset to it, union-merge this run's generation records
# into the committed ones (rbirag eval merge-records), copy this run's planning/retrieval
# outputs, regenerate every derived file (tables, charts, run_metadata, README section)
# from the merged data, commit and push. A rejected push repeats the whole loop, so derived
# files are never rebased or merged as text and no successful record is lost.
#
# Test hooks: PUBLISH_REPORT_CMD (default: rbirag eval report --update-readme README.md),
# PUBLISH_BEFORE_PUSH (command run before each push), PUBLISH_REMOTE/PUBLISH_BRANCH,
# PUBLISH_ATTEMPTS, PUBLISH_MESSAGE.
set -euo pipefail

RUN_DIR=$(cd "$1" && pwd)
REMOTE=${PUBLISH_REMOTE:-origin}
BRANCH=${PUBLISH_BRANCH:-main}
ATTEMPTS=${PUBLISH_ATTEMPTS:-6}
REPORT_CMD=${PUBLISH_REPORT_CMD:-rbirag eval report --update-readme README.md}
MESSAGE=${PUBLISH_MESSAGE:-Update evaluation results}

if find "$RUN_DIR" -name 'fatal-*.json' | grep -q .; then
  find "$RUN_DIR" -name 'fatal-*.json' -exec cat {} \;
  echo "::error::A fatal configuration error (unknown model or rejected key) occurred in this run; nothing is published."
  exit 1
fi

git config user.name >/dev/null || git config user.name "github-actions[bot]"
git config user.email >/dev/null || git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for attempt in $(seq 1 "$ATTEMPTS"); do
  git fetch --quiet "$REMOTE" "$BRANCH"
  git reset --quiet --hard "$REMOTE/$BRANCH"
  git clean -fdq -- results
  mkdir -p results/generation results/retrieval

  if [ -d "$RUN_DIR/plan" ]; then cp -R "$RUN_DIR/plan/." results/; fi
  if [ -d "$RUN_DIR/retrieval" ] && find "$RUN_DIR/retrieval" -name 'rows-*.json' | grep -q .; then
    find "$RUN_DIR/retrieval" -type f -exec cp {} results/retrieval/ \;
  fi
  # per-run status files describe only this run
  rm -f results/generation/status-*.json
  if [ -d "$RUN_DIR/generation" ]; then
    find "$RUN_DIR/generation" -type f \( -name 'status-*.json' -o -name 'gold_check-*.json' \) \
      -exec cp {} results/generation/ \;
    rbirag eval merge-records --incoming "$RUN_DIR/generation" --into results/generation
  fi

  eval "$REPORT_CMD"

  git add -A results README.md
  if git diff --cached --quiet; then
    echo "nothing to publish"
    exit 0
  fi
  git commit --quiet -m "$MESSAGE"
  if [ -n "${PUBLISH_BEFORE_PUSH:-}" ]; then (eval "$PUBLISH_BEFORE_PUSH"); fi
  if git push --quiet "$REMOTE" "HEAD:$BRANCH"; then
    echo "published on attempt ${attempt}"
    exit 0
  fi
  echo "push rejected (attempt ${attempt}); refetching and regenerating"
  sleep $((attempt * ${PUBLISH_BACKOFF_SECONDS:-10}))
done
echo "::error::could not publish after ${ATTEMPTS} attempts"
exit 1
