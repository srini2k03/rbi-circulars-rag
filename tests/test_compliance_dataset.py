from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rbirag.compliance.clauses import split_clauses
from rbirag.compliance.evaluation.dataset import (
    DatasetError,
    check_gold,
    dataset_stats,
    load_dataset,
    render_pdf,
    render_policy,
    validate_dataset,
)
from rbirag.compliance.evaluation.metrics import segmentation_score
from rbirag.compliance.policy import load_policy
from rbirag.document import write_documents_jsonl
from tests.conftest import make_doc

REPO_SET = Path(__file__).parents[1] / "data" / "eval" / "compliance"

POLICIES = [
    {
        "policy_id": "kyc",
        "title": "Sample Bank KYC Policy",
        "topic": "kyc",
        "split": "dev",
        "format": "markdown",
        "sections": [{"number": "1", "title": "Records"}, {"number": "2", "title": "Staff"}],
    },
    {
        "policy_id": "frm",
        "title": "Sample Bank Fraud Policy",
        "topic": "fraud",
        "split": "test",
        "format": "plain",
        "sections": [{"number": "1", "title": "Reporting"}],
    },
]


def label(item_id: str, **kw: object) -> dict[str, object]:
    policy, number = item_id.split("/")
    base: dict[str, object] = {
        "item_id": item_id,
        "policy_id": policy,
        "section": number.split(".")[0],
        "number": number,
        "text": "The Bank shall preserve transaction records for at least five years.",
        "verdict": "compliant",
        "citations": [["rbi-1"]],
        "evidence": [
            {
                "doc_id": "rbi-1",
                "paragraph": "50",
                "quote": "records for at least five years from the date",
            }
        ],
        "rationale": "Matches paragraph 50(1).",
        "tags": [],
        "expected_tools": ["search_regulations", "compare_clause"],
    }
    base.update(kw)
    return base


LABELS = [
    label("kyc/1.1"),
    label(
        "kyc/1.2",
        text="Customer identification records shall be kept for three years:\n(a) after closure; and\n(b) after the last transaction.",
        verdict="conflict",
        tags=["numeric_threshold"],
    ),
    label(
        "kyc/2.1",
        text="Staff are entitled to twelve days of casual leave a year.",
        verdict="not_applicable",
        citations=[],
        evidence=[],
        tags=["out_of_scope"],
        expected_tools=[],
    ),
    label(
        "frm/1.1",
        text="Frauds shall be reported to the Board within one month of detection.",
        verdict="insufficient_evidence",
        citations=[],
        evidence=[],
        tags=["not_in_corpus"],
        expected_tools=["search_regulations"],
    ),
]


def write_set(
    root: Path, policies: list[dict[str, object]], labels: list[dict[str, object]]
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "policies.json").write_text(json.dumps(policies), encoding="utf-8")
    (root / "labels.jsonl").write_text(
        "\n".join(json.dumps(x) for x in labels) + "\n", encoding="utf-8"
    )
    return root


def test_load_items_and_stats(tmp_path: Path) -> None:
    ds = load_dataset(write_set(tmp_path / "set", POLICIES, LABELS))
    assert [i.item_id for i in ds.items("dev")] == ["kyc/1.1", "kyc/1.2", "kyc/2.1"]
    assert [i.item_id for i in ds.items("test")] == ["frm/1.1"]
    it = ds.items("dev")[0]
    assert it.clause_text.startswith("1.1 The Bank") and it.section_title == "Records"
    stats = dataset_stats(ds)
    assert stats["items"] == 4 and stats["human_verified"] == 0
    assert stats["by_split"] == {
        "dev": {"compliant": 1, "conflict": 1, "not_applicable": 1},
        "test": {"insufficient_evidence": 1},
    }


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"verdict": "conflict", "citations": []}, "needs at least one citation group"),
        ({"evidence": []}, "needs evidence quotes"),
        ({"verdict": "not_applicable", "evidence": []}, "must not have citation groups"),
        ({"verdict": "not_applicable", "citations": [], "evidence": []}, "tagged out_of_scope"),
        (
            {"verdict": "insufficient_evidence", "citations": [], "evidence": []},
            "tagged not_in_corpus",
        ),
        (
            {"evidence": [{"doc_id": "rbi-7", "quote": "something long enough to quote"}]},
            "not a citation",
        ),
        ({"section": "9"}, "unknown section"),
        ({"section": "2"}, "is not inside section"),
        ({"number": "1.9"}, "item_id must be"),
        ({"text": "1.1 The Bank shall preserve records."}, "must not repeat the number"),
    ],
)
def test_validation_problems(tmp_path: Path, change: dict[str, object], problem: str) -> None:
    labels = [label("kyc/1.1", **change), *LABELS[1:]]
    with pytest.raises(DatasetError, match=problem):
        load_dataset(write_set(tmp_path / "set", POLICIES, labels))


def test_structural_problems(tmp_path: Path) -> None:
    ds = load_dataset(write_set(tmp_path / "set", POLICIES, LABELS))
    ds.labels.append(ds.labels[0])
    ds.policies[0].sections.append(ds.policies[0].sections[0].model_copy(update={"number": "3"}))
    problems = validate_dataset(ds)
    assert any("duplicate item" in p for p in problems)
    assert any("section 3 has no clauses" in p for p in problems)
    with pytest.raises(DatasetError, match=r"labels\.jsonl:1"):
        load_dataset(write_set(tmp_path / "bad", POLICIES, [{"item_id": "x"}]))


def test_rejected_items_are_excluded(tmp_path: Path) -> None:
    labels = [
        label("kyc/1.1", reviews=[{"reviewer": "human:a", "decision": "rejected"}]),
        *LABELS[1:],
    ]
    ds = load_dataset(write_set(tmp_path / "set", POLICIES, labels))
    assert "kyc/1.1" not in [i.item_id for i in ds.items()]
    assert "kyc/1.1" in [i.item_id for i in ds.items(include_rejected=True)]
    accepted = [
        label("kyc/1.1", reviews=[{"reviewer": "human:a", "decision": "accepted"}]),
        *LABELS[1:],
    ]
    ds = load_dataset(write_set(tmp_path / "set2", POLICIES, accepted))
    assert dataset_stats(ds)["human_verified"] == 1


@pytest.mark.parametrize("policy_id", ["kyc", "frm"])
def test_rendered_policies_split_back_into_the_gold_clauses(tmp_path: Path, policy_id: str) -> None:
    ds = load_dataset(write_set(tmp_path / "set", POLICIES, LABELS))
    policy = next(p for p in ds.policies if p.policy_id == policy_id)
    labels = ds.labels_for(policy_id)
    text = render_policy(policy, labels)
    clauses = split_clauses(load_policy(text.encode(), filename="p.md").text)
    gold = [f"{lb.number} {lb.text}" for lb in labels]
    assert segmentation_score(gold, [c.text for c in clauses]).f1 == 1.0


def test_pdf_rendering_round_trip(tmp_path: Path) -> None:
    ds = load_dataset(write_set(tmp_path / "set", POLICIES, LABELS))
    policy = ds.policies[0]
    labels = ds.labels_for("kyc")
    pdf = render_pdf(render_policy(policy, labels), tmp_path / "kyc.pdf")
    clauses = split_clauses(load_policy(pdf).text)
    assert [c.number for c in clauses] == ["1.1", "1.2", "2.1"]


def test_check_gold_and_cli(tmp_path: Path) -> None:
    root = write_set(tmp_path / "set", POLICIES, LABELS)
    ds = load_dataset(root)
    good = make_doc(
        "50. (1) maintain records for at least five years from the date of transaction",
        doc_id="rbi-1",
    )
    assert check_gold(ds, {"rbi-1": good}).ok
    bad = make_doc("records for three years", doc_id="rbi-1")
    result = check_gold(ds, {"rbi-1": bad})
    assert not result.ok and len(result.missing_quotes) == 2
    assert check_gold(ds, {}).missing_documents == ["kyc/1.1: rbi-1", "kyc/1.2: rbi-1"]

    from rbirag.cli import app

    corpus = tmp_path / "docs.jsonl"
    write_documents_jsonl([good], corpus)
    runner = CliRunner()
    out_file = tmp_path / "gold.json"
    res = runner.invoke(
        app,
        [
            "comply",
            "eval",
            "check",
            "--root",
            str(root),
            "--documents",
            str(corpus),
            "--out",
            str(out_file),
        ],
    )
    assert res.exit_code == 0, res.output
    assert json.loads(out_file.read_text())["missing_quotes"] == []
    res = runner.invoke(app, ["comply", "eval", "stats", "--root", str(root)])
    assert res.exit_code == 0 and '"items": 4' in res.output
    res = runner.invoke(app, ["comply", "eval", "segmentation", "--root", str(root)])
    assert res.exit_code == 0 and "| **all** | pdf | 4 | 4 | 1.00 | 1.00 | 1.00 |" in res.output
    res = runner.invoke(app, ["comply", "eval", "render", "--root", str(root), "--pdf"])
    assert (
        res.exit_code == 0
        and (root / "policies" / "frm.md").exists()
        and (root / "policies" / "kyc.pdf").exists()
    )

    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        json.dumps(
            {
                "item_id": "frm/1.1",
                "verdict": "insufficient_evidence",
                "system": "baseline",
                "confidence": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    res = runner.invoke(
        app, ["comply", "eval", "score", str(preds), "--root", str(root), "--split", "test"]
    )
    assert res.exit_code == 0, res.output
    assert "baseline (test, n=1)" in res.output
    bad_root = write_set(
        tmp_path / "bad", POLICIES, [label("kyc/1.1", verdict="gap", citations=[])]
    )
    assert runner.invoke(app, ["comply", "eval", "stats", "--root", str(bad_root)]).exit_code == 1


# --------------------------------------------------------------------- the committed set


@pytest.fixture(scope="module")
def committed():  # type: ignore[no-untyped-def]
    if not (REPO_SET / "labels.jsonl").exists():
        pytest.skip("labelled set not committed yet")
    return load_dataset(REPO_SET)


def test_committed_set_is_consistent(committed) -> None:  # type: ignore[no-untyped-def]
    stats = dataset_stats(committed)
    assert 50 <= stats["items"] <= 90
    verdicts = stats["by_verdict"]
    assert set(verdicts) == {
        "compliant",
        "conflict",
        "gap",
        "not_applicable",
        "insufficient_evidence",
    }
    assert min(verdicts.values()) >= 6
    assert stats["policies_by_split"]["dev"] and stats["policies_by_split"]["test"]
    for tag in (
        "outdated_rule",
        "partial_compliance",
        "out_of_scope",
        "not_in_corpus",
        "entity_confusion",
    ):
        assert stats["tags"].get(tag, 0) >= 3, tag


def test_committed_policies_are_rendered_and_split_cleanly(committed) -> None:  # type: ignore[no-untyped-def]
    for policy in committed.policies:
        labels = committed.labels_for(policy.policy_id)
        text = render_policy(policy, labels)
        on_disk = REPO_SET / "policies" / f"{policy.policy_id}.md"
        assert on_disk.read_text(encoding="utf-8") == text, (
            f"re-render {on_disk} (rbirag comply eval render)"
        )
        clauses = split_clauses(load_policy(on_disk).text)
        gold = [f"{lb.number} {lb.text}" for lb in labels]
        assert segmentation_score(gold, [c.text for c in clauses]).f1 == 1.0, policy.policy_id
