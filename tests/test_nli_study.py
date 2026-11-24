"""Evidence windows, claim splitting and the NLI-vs-judge study (fake NLI)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from typer.testing import CliRunner

from rbirag.document import Document, write_documents_jsonl
from rbirag.eval.gen_metrics import GenerationRecord
from rbirag.eval.nli import FakeNLI
from rbirag.eval.nli_study import study_answer, summarize_study
from rbirag.generation.evidence import is_non_answer, premise_windows, split_claims

SOURCE = " ".join(f"Rule {i} says banks must keep record {i} for {i} years." for i in range(12))


def test_premise_windows_cover_everything() -> None:
    windows = premise_windows(SOURCE)
    assert len(windows) == 6  # 12 sentences, 3 per window, stride 2 (+ the tail window)
    covered = set()
    for w in windows:
        assert SOURCE[w.start : w.end] == w.text
        covered.update(range(w.start, w.end))
    assert all(i in covered for i, ch in enumerate(SOURCE) if not ch.isspace())
    long = " ".join(["word"] * 300)
    parts = premise_windows(long, max_words=120)
    assert len(parts) == 4 and parts[-1].end == len(long)
    assert premise_windows("") == []


def test_split_claims_and_non_answers() -> None:
    assert split_claims("Banks must do X; the limit is Rs 5 lakh, and the RBI may extend it.") == [
        "Banks must do X", "the limit is Rs 5 lakh", "the RBI may extend it",
    ]  # fmt: skip
    assert split_claims("KYC and AML rules apply to banks and NBFCs.") == [
        "KYC and AML rules apply to banks and NBFCs"
    ]
    assert is_non_answer("The circular does not specify the penalty.")
    assert is_non_answer("No information is available in the sources.")
    assert not is_non_answer("Banks must keep record 3 for 3 years [1].")


def _record(answer: str, judge: float = 1.0) -> GenerationRecord:
    return GenerationRecord(
        qa_id="q1", config="fixed-1024-bm25", backend="ollama", model="m", answerable=True,
        question="How long must banks keep record 7?", answer=answer, refused=False,
        refusal_reason=None, citation_valid=True, n_citations=1, n_invalid_citations=0,
        n_uncited_sentences=0, judge_faithfulness=judge, nli_faithfulness=None,
    )  # fmt: skip


def test_windows_rescue_evidence_past_the_cutoff() -> None:
    padding = " ".join(f"Filler sentence number {i} is unrelated text." for i in range(80))
    sources = [padding + " " + SOURCE, "Unrelated text about currency chests and coins here."]

    class LengthSensitiveNLI(FakeNLI):
        """Overlap NLI that only sees the first 512 words of the premise."""

        def entailment(self, pairs):  # type: ignore[no-untyped-def]
            return super().entailment([(" ".join(p.split()[:512]), h) for p, h in pairs])

    row = study_answer(
        _record("Banks must keep record 7 for 7 years [1]."),
        sources, LengthSensitiveNLI(), lambda t: len(t.split()),
    )  # fmt: skip
    assert row.nli["whole_chunk"] == 0.0
    assert row.nli["windows_cited"] == 1.0 and row.nli["windows_any"] == 1.0
    assert row.rescued_by_windows == 1 and row.rescued_beyond_cutoff == 1
    assert row.premises_truncated == 1
    summary = summarize_study([row, study_answer(_record("The circular does not specify it."),
                                                 sources, FakeNLI(), lambda t: len(t.split()))])  # fmt: skip
    assert summary["answers"] == 2 and summary["non_answers"]["count"] == 1
    assert summary["by_backend"]["ollama"]["answers"] == 2


def test_nli_study_cli(fixture_docs: list[Document], tmp_path: Path) -> None:
    from rbirag.cli import app
    from rbirag.eval.ablate import append_jsonl

    docs = tmp_path / "docs.jsonl"
    write_documents_jsonl(fixture_docs, docs)
    rec = _record("Periodic updation of KYC is required every two years for high risk "
                  "customers [1].").model_copy(update={"question": "KYC updation high risk?"})  # fmt: skip
    append_jsonl(
        Path("results/generation/records-ollama-fixed-1024-bm25.jsonl"), [rec.model_dump()]
    )
    out = tmp_path / "analysis"
    res = CliRunner().invoke(app, ["eval", "nli-study", "--docs", str(docs), "--nli", "fake",
                                   "--out-dir", str(out), "--no-log"])  # fmt: skip
    assert res.exit_code == 0, res.output
    summary = json.loads((out / "nli_study.json").read_text())
    assert summary["answers"] == 1 and summary["nli_model"] == "fake-overlap-nli"
    with gzip.open(out / "nli_study_answers.jsonl.gz", "rt") as fh:
        assert json.loads(fh.readline())["qa_id"] == "q1"
    assert not Path("results/experiments.jsonl").exists()  # --no-log wrote nothing
