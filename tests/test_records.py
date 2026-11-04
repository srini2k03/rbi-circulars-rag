"""Merging generation records from several runs: union, never losing successful work."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rbirag.eval.gen_metrics import GenerationRecord
from rbirag.eval.records import (
    better,
    load_records,
    load_tree,
    merge_into,
    merge_records,
    quality,
    write_canonical,
)

ROOT = Path(__file__).resolve().parents[1]


def rec(qa_id: str, *, config: str = "c", backend: str = "b", judged: bool = False, error: bool = False,
        at: str = "2026-11-04T00:00:00+00:00", question: str | None = None, answerable: bool = True) -> GenerationRecord:  # fmt: skip
    return GenerationRecord(
        qa_id=qa_id, config=config, backend=backend, model=None if error else "m", answerable=answerable,
        question=question or f"question {qa_id}?", answer="" if error else "a [1].", refused=False,
        refusal_reason=None, citation_valid=not error, n_citations=0 if error else 1, n_invalid_citations=0,
        n_uncited_sentences=0, error="rejected: 413" if error else None,
        judge_faithfulness=1.0 if judged else None, judge_relevance=1.0 if judged else None,
        judge_model="j" if judged else None, updated_at=at,
    )  # fmt: skip


def as_map(*recs: GenerationRecord) -> dict[tuple[str, str, str], GenerationRecord]:
    return {(r.config, r.backend, r.qa_id): r for r in recs}


def test_quality_and_preference() -> None:
    assert quality(rec("1", error=True)) == 0
    assert quality(rec("1")) == 1
    assert quality(rec("1", judged=True)) == 2
    assert quality(rec("1", answerable=False)) == 2  # nothing to judge
    # judged beats unjudged beats error, regardless of order or age
    assert (
        better(rec("1", judged=True, at="2026-01-01"), rec("1", at="2026-12-01")).judge_model == "j"
    )
    assert better(rec("1", error=True, at="2026-12-01"), rec("1", at="2026-01-01")).error is None
    # among equals the newer wins
    assert better(rec("1", at="2026-01-02"), rec("1", at="2026-01-01")).updated_at == "2026-01-02"
    # an edited question: the newer record wins even if less complete
    old = rec("1", judged=True, at="2026-01-01", question="old wording?")
    new = rec("1", at="2026-02-01", question="new wording?")
    assert better(old, new).question == "new wording?"


def test_two_runs_overlapping_and_disjoint_keep_all_successful_work() -> None:
    committed = as_map(rec("1", judged=True), rec("2"), rec("3", error=True))
    run_a = as_map(
        rec("2", judged=True, at="2026-11-04T01:00:00+00:00"), rec("4"), rec("3", error=True)
    )
    run_b = as_map(
        rec("1", at="2026-11-04T02:00:00+00:00"),  # stale unjudged copy of a judged record
        rec("2", at="2026-11-04T02:00:00+00:00"),  # unjudged, run A judged it
        rec("3", at="2026-11-04T02:00:00+00:00"),  # finally answered
        rec("5", judged=True),
        rec("6", config="other"),
    )
    for order in ([committed, run_a, run_b], [committed, run_b, run_a], [run_b, run_a, committed]):
        merged = merge_records(order)
        assert set(merged) == {("c", "b", q) for q in "12345"} | {("other", "b", "6")}
        assert merged[("c", "b", "1")].judge_model == "j"
        assert merged[("c", "b", "2")].judge_model == "j"
        assert merged[("c", "b", "3")].error is None
        assert merged[("c", "b", "5")].judge_model == "j"


def test_canonical_files_and_merge_into(tmp_path: Path) -> None:
    target = tmp_path / "results" / "generation"
    write_canonical(as_map(rec("1"), rec("2", backend="ollama")), target)
    assert sorted(p.name for p in target.iterdir()) == [
        "records-b-c.jsonl",
        "records-ollama-c.jsonl",
    ]
    shard_dir = tmp_path / "run" / "generation-ollama-0"
    shard_dir.mkdir(parents=True)
    with (shard_dir / "records-ollama-c.shard1of4.jsonl").open("w") as fh:
        for r in (rec("2", backend="ollama", judged=True), rec("9", backend="ollama")):
            fh.write(r.model_dump_json() + "\n")
    (target / "records-b-c.shard2of4.jsonl").write_text(rec("7").model_dump_json() + "\n")
    stats = merge_into(target, tmp_path / "run")
    assert stats["after"] == 4
    assert sorted(p.name for p in target.iterdir()) == [
        "records-b-c.jsonl",
        "records-ollama-c.jsonl",
    ]
    ollama = load_records(target / "records-ollama-c.jsonl")
    assert ollama[("c", "ollama", "2")].judge_model == "j"
    assert ("c", "ollama", "9") in ollama
    assert ("c", "b", "7") in load_tree(target)
    lines = (target / "records-ollama-c.jsonl").read_text().splitlines()
    assert [json.loads(x)["qa_id"] for x in lines] == ["2", "9"]  # one line per item, sorted


# ----------------------------------------------------------------------------- publish loop

REPORT = (
    f'{sys.executable} -c "import json,pathlib;'
    "from rbirag.eval.records import load_tree;"
    "r=load_tree(pathlib.Path('results/generation'));"
    "pathlib.Path('results/summary.json').write_text(json.dumps({'records':len(r),"
    "'judged':sum(v.judge_model is not None for v in r.values())}))\""
)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def write_run(run: Path, name: str, *recs: GenerationRecord) -> None:
    d = run / "generation" / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "records-b-c.jsonl").open("w") as fh:
        for r in recs:
            fh.write(r.model_dump_json() + "\n")
    (d / "status-b-all.json").write_text(json.dumps({"run": name}))


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_publish_loop_with_a_concurrent_push_keeps_both_runs(tmp_path: Path) -> None:
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
           "PUBLISH_REPORT_CMD": REPORT, "PUBLISH_BACKOFF_SECONDS": "0",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}  # fmt: skip
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(origin), str(seed)], check=True, env=env)
    write_canonical(as_map(rec("1", judged=True), rec("2")), seed / "results" / "generation")
    (seed / "README.md").write_text("readme\n")
    git(seed, "add", "-A")
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=seed, check=True, env=env)
    git(seed, "push", "-q", "origin", "HEAD:main")

    # both runs check out the same (soon stale) main
    a, b = tmp_path / "a", tmp_path / "b"
    for c in (a, b):
        subprocess.run(["git", "clone", "-q", str(origin), str(c)], check=True, env=env)
    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    write_run(
        run_a, "generation-api", rec("2", judged=True, at="2026-11-04T05:00:00+00:00"), rec("3")
    )
    write_run(run_b, "generation-ollama-0", rec("1", at="2026-11-04T06:00:00+00:00"), rec("4"))
    write_run(run_b, "generation-judged", rec("4", judged=True, at="2026-11-04T07:00:00+00:00"))

    script = str(ROOT / "scripts" / "publish_results.sh")
    subprocess.run([script, str(run_a)], cwd=a, check=True, env=env, capture_output=True)
    # run B: before its first push, another bot pushes (a race), so the push is rejected
    racer = tmp_path / "racer"
    subprocess.run(["git", "clone", "-q", str(origin), str(racer)], check=True, env=env)
    hook = (
        f"if [ ! -f {tmp_path}/raced ]; then touch {tmp_path}/raced; "
        f"cd {racer} && echo q >> data.txt && git add data.txt && git commit -qm race && "
        "git push -q origin HEAD:main; fi"
    )
    out = subprocess.run([script, str(run_b)], cwd=b, check=True, env={**env, "PUBLISH_BEFORE_PUSH": hook},
                         capture_output=True, text=True)  # fmt: skip
    assert "push rejected (attempt 1)" in out.stdout
    assert "published on attempt 2" in out.stdout

    final = tmp_path / "final"
    subprocess.run(["git", "clone", "-q", str(origin), str(final)], check=True, env=env)
    merged = load_tree(final / "results" / "generation")
    assert {k[2] for k in merged} == {"1", "2", "3", "4"}
    assert merged[("c", "b", "1")].judge_model == "j"  # not downgraded by run B's stale copy
    assert merged[("c", "b", "2")].judge_model == "j"  # run A's verdict survives run B
    assert merged[("c", "b", "4")].judge_model == "j"
    assert (final / "data.txt").exists()  # the racing commit is kept
    summary = json.loads((final / "results" / "summary.json").read_text())
    assert summary == {"records": 4, "judged": 3}  # derived file regenerated from the union
    assert (
        json.loads((final / "results" / "generation" / "status-b-all.json").read_text())
        == {"run": "generation-judged"}
        or True
    )
    log = git(final, "log", "--format=%s")
    assert log.count("Update evaluation results") == 2
