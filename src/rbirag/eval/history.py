"""Append-only experiment log (``results/experiments.jsonl``).

One line per evaluation run: what ran (command, configuration hashes), on what (commit,
corpus and test-set hashes, split sizes), where (CI run id), when, and its headline
numbers. Lines are never rewritten, so the file is the history of every reported result.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENTS_FILE = "experiments.jsonl"


def log_experiment(results_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Append ``entry`` (plus timestamp, CI run id and runner info) and return it."""
    record = {
        "logged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "ci_run_id": os.environ.get("GITHUB_RUN_ID"),
        "ci_workflow": os.environ.get("GITHUB_WORKFLOW"),
        "runner": os.environ.get("RUNNER_NAME") or os.uname().nodename,
        **entry,
    }
    # CI jobs write their new lines to a separate file (RBIRAG_EXPERIMENT_LOG) and append it
    # to the committed log only when they publish (scripts/commit_outputs.sh --log).
    override = os.environ.get("RBIRAG_EXPERIMENT_LOG")
    path = Path(override) if override else results_dir / EXPERIMENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str, sort_keys=True) + "\n")
    return record


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def read_experiments(results_dir: Path) -> list[dict[str, Any]]:
    path = results_dir / EXPERIMENTS_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
