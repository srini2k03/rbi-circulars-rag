"""Generation records on disk: loading, and conflict-free merging of several runs.

Records are per-item facts keyed by ``(config, backend, qa_id)`` (the dev/test split is a
function of the qa_id, so it is part of the key implicitly). Different workflow runs and
shards produce overlapping record sets; :func:`merge_records` combines them as a *union*,
preferring the most useful version of each item, so no successful work is ever lost and
no derived file has to be merged textually.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

from rbirag.eval.gen_metrics import GenerationRecord

logger = logging.getLogger(__name__)

RecordKey = tuple[str, str, str]  # (config, backend, qa_id)


def load_records(path: Path) -> dict[RecordKey, GenerationRecord]:
    """Last record wins per (config, backend, qa_id) within one file."""
    out: dict[RecordKey, GenerationRecord] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = GenerationRecord.model_validate_json(line)
            except ValueError:
                continue
            out[(rec.config, rec.backend, rec.qa_id)] = rec
    return out


def judged(rec: GenerationRecord) -> bool:
    return rec.judge_faithfulness is not None or rec.judge_relevance is not None


def quality(rec: GenerationRecord) -> int:
    """0 = failed/pending answer, 1 = answered (verdict pending or not needed but
    attempted), 2 = answered and complete (judged, or needs no verdict)."""
    if rec.error is not None:
        return 0
    needs_verdict = rec.answerable and not rec.refused
    if not needs_verdict or judged(rec):
        return 2
    return 1


def better(a: GenerationRecord, b: GenerationRecord) -> GenerationRecord:
    """The record to keep for one key.

    * If the question wording differs (a pair was edited), the newer record wins.
    * Otherwise a successful, judged record beats an unjudged one, which beats an error;
      between equals the newer ``updated_at`` wins (``b`` wins exact ties, so later
      inputs override earlier ones).
    """
    if a.question != b.question:
        return b if (b.updated_at or "") >= (a.updated_at or "") else a
    qa, qb = quality(a), quality(b)
    if qa != qb:
        return a if qa > qb else b
    return b if (b.updated_at or "") >= (a.updated_at or "") else a


def merge_records(
    sets: Iterable[Mapping[RecordKey, GenerationRecord]],
) -> dict[RecordKey, GenerationRecord]:
    out: dict[RecordKey, GenerationRecord] = {}
    for records in sets:
        for key, rec in records.items():
            out[key] = better(out[key], rec) if key in out else rec
    return out


def load_tree(root: Path) -> dict[RecordKey, GenerationRecord]:
    """Union of every ``records-*.jsonl`` below ``root`` (recursively)."""
    if not root.exists():
        return {}
    return merge_records(load_records(p) for p in sorted(root.rglob("records-*.jsonl")))


def canonical_name(backend: str, config: str) -> str:
    return f"records-{backend}-{config}.jsonl"


def write_canonical(records: Mapping[RecordKey, GenerationRecord], out_dir: Path) -> list[Path]:
    """One compact file per (backend, config), sorted by qa_id; replaces every other
    ``records-*.jsonl`` (e.g. shard files) in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[GenerationRecord]] = {}
    for (config, backend, _), rec in records.items():
        groups.setdefault((backend, config), []).append(rec)
    keep: set[Path] = set()
    for (backend, config), recs in groups.items():
        path = out_dir / canonical_name(backend, config)
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in sorted(recs, key=lambda r: r.qa_id):
                fh.write(json.dumps(rec.model_dump(), ensure_ascii=False, default=str) + "\n")
        tmp.replace(path)
        keep.add(path)
    for stale in out_dir.glob("records-*.jsonl"):
        if stale not in keep:
            stale.unlink()
    return sorted(keep)


def merge_into(target: Path, *incoming: Path) -> dict[str, int]:
    """Merge every record set under ``incoming`` into ``target`` (a results/generation
    directory) and rewrite it canonically. Returns counts for logging."""
    base = load_tree(target)
    new = [load_tree(p) for p in incoming]
    merged = merge_records([base, *new])
    write_canonical(merged, target)
    stats = {
        "before": len(base),
        "incoming": sum(len(n) for n in new),
        "after": len(merged),
        "complete": sum(quality(r) == 2 for r in merged.values()),
        "answered": sum(quality(r) >= 1 for r in merged.values()),
    }
    logger.info("merged records: %s", stats)
    return stats
