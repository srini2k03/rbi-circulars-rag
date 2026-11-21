"""Run records: what a live evaluation run used, so it can be reported and replayed.

Every ``rbirag comply eval run`` writes ``results/compliance/runs/<system>-<split>-<backend>.json``
next to its predictions. The record holds the configured model and every setting that
shapes the prompts, so ``rbirag comply eval replay`` can rebuild the exact requests and
answer them from the committed response cache.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rbirag.config import Settings

RUNS_DIR = Path("results/compliance/runs")
PREDICTIONS_DIR = Path("results/compliance/predictions")
# Settings that change what the model is asked (and therefore the cache keys).
PROMPT_SETTINGS = (
    "compliance_search_mode",
    "compliance_chunker",
    "compliance_search_top_k",
    "compliance_search_depth_factor",
    "compliance_snippet_chars",
    "compliance_fetch_max_tokens",
    "compliance_passage_chars",
    "compliance_entity",
    "compliance_entity_filter",
    "compliance_as_of",
    "compliance_max_steps",
    "compliance_planner_max_tokens",
    "compliance_judge_max_tokens",
    "compliance_judge_context_tokens",
    "compliance_abstain_threshold",
    "compliance_require_verified_quote",
    "anthropic_effort",
    "chunk_size",
    "chunk_overlap",
)


class RunRecord(BaseModel):
    system: str
    split: str
    backend: str
    model: str  # the configured model id (cache key); served models are per prediction
    predictions: str
    item_ids: list[str]
    settings: dict[str, Any]
    corpus: str = "corpus-v2"
    generated_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    )
    commit: str | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    completed: int = 0
    errors: int = 0


def run_name(system: str, split: str, backend: str) -> str:
    return f"{system}-{split}-{backend}"


def current_commit() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def prompt_settings(settings: Settings) -> dict[str, Any]:
    return {k: getattr(settings, k) for k in PROMPT_SETTINGS}


def save_run(record: RunRecord, runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run_name(record.system, record.split, record.backend)}.json"
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_runs(runs_dir: Path = RUNS_DIR, *, split: str | None = None) -> list[RunRecord]:
    if not runs_dir.exists():
        return []
    records = [
        RunRecord.model_validate(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(runs_dir.glob("*.json"))
    ]
    return [r for r in records if split is None or r.split == split]


def apply_run_settings(settings: Settings, record: RunRecord) -> Settings:
    return settings.model_copy(update={**record.settings, "compliance_llm_backend": record.backend})
