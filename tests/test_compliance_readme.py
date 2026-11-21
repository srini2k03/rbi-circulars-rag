from __future__ import annotations

import json
from pathlib import Path

import pytest

from rbirag.compliance.evaluation.dataset import EvalItem, load_dataset
from rbirag.compliance.evaluation.readme import END, START, readme_section, update_readme
from rbirag.compliance.evaluation.runs import RunRecord, save_run
from rbirag.compliance.records import ClausePrediction, ToolCall

ROOT = Path(__file__).resolve().parents[1] / "data" / "eval" / "compliance"


def _perfect(item: EvalItem) -> ClausePrediction:
    label = item.label
    cites = [group[0] for group in label.citations]
    calls = [ToolCall(name=n, arguments={}, valid=True) for n in label.expected_tools]
    return ClausePrediction(
        item_id=item.item_id,
        verdict=label.verdict,
        citations=cites,
        confidence=0.9,
        tool_calls=calls,
        llm_calls=len(calls) + 1,
        input_tokens=1000,
        output_tokens=100,
        latency_ms=1500.0,
        system="agent",
        model="free-model",
    )


def _record(tmp: Path, system: str, split: str, items: list[EvalItem], n: int) -> None:
    preds = tmp / f"{system}-{split}.jsonl"
    with preds.open("w", encoding="utf-8") as fh:
        for item in items[:n]:
            pred = _perfect(item)
            if system == "baseline" and item.label.verdict == "conflict":
                pred = pred.model_copy(update={"verdict": "gap"})  # the baseline is worse
            fh.write(pred.model_dump_json() + "\n")
    save_run(
        RunRecord(
            system=system,
            split=split,
            backend="gemini",
            model="free-model",
            predictions=str(preds),
            item_ids=[i.item_id for i in items],
            settings={},
            commit="abcdef0123",
            completed=n,
        ),
        tmp / "runs",
    )


def test_readme_section_progress_then_complete(tmp_path: Path) -> None:
    dataset = load_dataset(ROOT)
    dev, test = dataset.items("dev"), dataset.items("test")
    for system in ("agent", "baseline"):
        _record(tmp_path, system, "dev", dev, len(dev))
    _record(tmp_path, "agent", "test", test, 10)
    kwargs = {"runs_dir": tmp_path / "runs", "retrieval_path": tmp_path / "none.json"}

    section = readme_section(dataset, **kwargs)
    assert section.startswith(START) and section.endswith(END)
    assert "0 human-verified" in section
    assert f"in progress** (agent 10/{len(test)}, baseline 0/{len(test)} clauses)" in section
    assert "| verdict accuracy | 1.000 [1.000, 1.000] |" in section  # dev agent is perfect
    assert "`conflict_gap_confusion`" in section  # the baseline's dev failures
    assert "Failures by type (dev split" in section
    assert "No retrieval results recorded" in section

    _record(tmp_path, "agent", "test", test, len(test))
    _record(tmp_path, "baseline", "test", test, len(test))
    done = readme_section(dataset, **kwargs)
    assert f"**Test split (locked, n = {len(test)}):** agent: `free-model` via gemini" in done
    assert "Failures by type (test split" in done and "F1 per verdict (test)" in done

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n\n{START}\nold\n{END}\n\noutro\n", encoding="utf-8")
    assert update_readme(readme, done) is True
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro") and text.endswith("outro\n") and "\nold\n" not in text
    assert update_readme(readme, done) is False
    readme.write_text("no markers", encoding="utf-8")
    with pytest.raises(ValueError, match="COMPLIANCE:START"):
        update_readme(readme, done)


def test_readme_cli_check(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from rbirag.cli import app

    dataset = load_dataset(ROOT)
    for system in ("agent", "baseline"):
        _record(tmp_path, system, "dev", dataset.items("dev"), len(dataset.items("dev")))
    docs = tmp_path / "documents.jsonl"
    docs.write_text(
        json.dumps({"doc_id": "rbi-1", "url": "u", "title": "t", "text": "x"}) + "\n",
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n", encoding="utf-8")
    common = ["comply", "eval", "readme", "--documents", str(docs), "--readme", str(readme)]
    common += ["--runs-dir", str(tmp_path / "runs"), "--retrieval", str(tmp_path / "r.json")]
    common += ["--root", str(ROOT)]
    runner = CliRunner()
    assert runner.invoke(app, [*common, "--check"]).exit_code == 1
    result = runner.invoke(app, common)
    assert result.exit_code == 0 and "updated" in result.output
    assert runner.invoke(app, [*common, "--check"]).exit_code == 0
