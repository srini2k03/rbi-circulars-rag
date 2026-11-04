from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from rbirag.config import Settings
from rbirag.document import Document
from rbirag.eval.ablate import (
    RetrievalConfig,
    build_config_index,
    grid,
    load_records,
    plan_generation,
    planned_calls,
    run_generation_eval,
    run_retrieval_ablation,
)
from rbirag.eval.dataset import GoldSpan, QAPair
from rbirag.eval.nli import FakeNLI
from rbirag.generation.llm import FakeLLM, Message, QuotaExhaustedError, ScriptedLLM
from rbirag.generation.rag import RAGPipeline
from rbirag.index.embedder import FakeEmbedder
from rbirag.retrieval.pipeline import Retriever
from rbirag.retrieval.rerank import FakeReranker

JUDGE_JSON = json.dumps(
    {
        "claims": [{"claim": "c", "supported": True, "source_ids": [1]}],
        "relevance": 5,
        "relevance_reason": "ok",
    }
)


def gold_for(doc: Document, phrase: str) -> GoldSpan:
    i = doc.text.index(phrase)
    return GoldSpan(doc_id=doc.doc_id, start=i, end=i + len(phrase))


@pytest.fixture
def qa(fixture_docs: list[Document]) -> list[QAPair]:
    by = {d.source_id: d for d in fixture_docs}
    return [
        QAPair(qa_id="kyc", question="How often is periodic updation of KYC for high risk customers?",
               answer="Every two years.", qtype="numeric",
               gold=[gold_for(by[12655], "at least once in every two years for high risk customers")]),
        QAPair(qa_id="bulk", question="Differential interest rates on bulk deposits for small finance banks",
               answer="Allowed above 3 crore.", qtype="factual",
               gold=[gold_for(by[12701], "Small Finance Banks shall be permitted to offer differential rates")]),
        QAPair(qa_id="firms", question="Form ODI reporting through the FIRMS portal deadline",
               answer="Within 30 days.", qtype="procedural",
               gold=[gold_for(by[12590], "within 30 days of the transaction")]),
        QAPair(qa_id="none", question="What is the SEBI cap on mutual fund expense ratios?", answer="",
               qtype="unanswerable", answerable=False),
    ]  # fmt: skip


def test_grid() -> None:
    configs = grid()
    assert len(configs) == 4 * 3 * 3
    assert RetrievalConfig.parse("section-512-hybrid_rerank") == RetrievalConfig(
        "section", 512, "hybrid_rerank"
    )


def test_retrieval_ablation_rows(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    per_query = tmp_path / "pq.jsonl"
    rows = run_retrieval_ablation(
        fixture_docs, qa, settings=Settings(), embedder=FakeEmbedder(64), reranker=FakeReranker(),
        chunkers=["fixed", "section"], sizes=[64, 128], modes=["bm25", "hybrid_rerank"],
        per_query_path=per_query,
    )  # fmt: skip
    assert len(rows) == 2 * 2 * 2
    for r in rows:
        assert r["n_queries"] == 3  # the unanswerable question is not scored
        assert 0 <= r["chunk_ndcg@10"] <= 1
        assert r["doc_recall@10"] == 1.0  # 3 documents: every gold doc is in the top 10
        lo, hi = r["chunk_ndcg@10_ci"]
        assert lo <= r["chunk_ndcg@10"] <= hi
    lines = [json.loads(x) for x in per_query.read_text().splitlines()]
    assert len(lines) == 8 * 4
    rerank = [x for x in lines if x["config"].endswith("hybrid_rerank")]
    assert all(x["confidence"] is not None for x in rerank)


def test_plan_and_planned_calls(qa: list[QAPair]) -> None:
    rows = [
        {"config": "section-512-hybrid_rerank", "mode": "hybrid_rerank", "chunk_ndcg@10": 0.8},
        {"config": "fixed-512-hybrid_rerank", "mode": "hybrid_rerank", "chunk_ndcg@10": 0.7},
        {"config": "section-512-bm25", "mode": "bm25", "chunk_ndcg@10": 0.6},
        {"config": "section-512-dense", "mode": "dense", "chunk_ndcg@10": 0.5},
        {"config": "fixed-256-dense", "mode": "dense", "chunk_ndcg@10": 0.4},
    ]
    plan = plan_generation(rows, qa, full_configs=1, subset_configs=2, subset_size=2)
    assert plan.full == ["section-512-hybrid_rerank"]
    assert plan.subset == ["section-512-bm25", "section-512-dense"]  # one per other mode
    assert len(plan.subset_ids) == 2 + 1  # 2 answerable + round(2 * 1/3) unanswerable
    calls = planned_calls(plan, qa, ["openai_compatible", "ollama"])
    assert calls["generate[openai_compatible]"] == 4 + 3 + 3
    assert calls["judge (all backends)"] == (3 + 2 + 2) * 2
    assert calls["api_calls_upper_bound"] == 10 + 14  # ollama generation is local


def make_pipeline(fixture_docs: list[Document], llm: object) -> RAGPipeline:
    _, bundle = build_config_index(fixture_docs, "section", 128, Settings(), FakeEmbedder(64))
    retriever = Retriever(
        bundle.store,
        bm25=bundle.bm25,
        dense=bundle.dense,
        reranker=FakeReranker(),
        mode="hybrid_rerank",
    )
    return RAGPipeline(retriever, llm, top_k=3)  # type: ignore[arg-type]


def test_generation_eval_resumable(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    records = tmp_path / "records.jsonl"
    pipeline = make_pipeline(fixture_docs, FakeLLM())
    judge = ScriptedLLM(lambda m, s: JUDGE_JSON, model="judge-model")
    run = run_generation_eval(
        pipeline,
        qa,
        config="c",
        records_path=records,
        judge=judge,
        nli=FakeNLI(),
        embedder=FakeEmbedder(64),
    )
    assert run.complete
    assert run.generated == 4
    recs = load_records(records)
    assert len(recs) == 4
    unans = recs[("c", "fake", "none")]
    assert unans.refused  # gated by low confidence
    assert unans.judge_faithfulness is None
    kyc = recs[("c", "fake", "kyc")]
    assert kyc.judge_faithfulness == 1.0
    assert kyc.judge_relevance == 1.0
    assert kyc.judge_model == "judge-model"
    assert kyc.nli_faithfulness is not None
    assert kyc.emb_relevance is not None
    n_judge_calls = len(judge.calls)
    assert n_judge_calls == sum(1 for r in recs.values() if r.answerable and not r.refused)
    again = run_generation_eval(pipeline, qa, config="c", records_path=records, judge=judge)
    assert (again.generated, again.judged, again.skipped) == (0, 0, 4)
    assert len(judge.calls) == n_judge_calls


def test_judge_quota_does_not_stop_generation_and_judging_resumes(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    records = tmp_path / "records.jsonl"
    pipeline = make_pipeline(fixture_docs, FakeLLM())

    def no_quota(messages: Sequence[Message], system: str | None) -> str:
        raise QuotaExhaustedError("daily")

    run = run_generation_eval(
        pipeline, qa, config="c", records_path=records, judge=ScriptedLLM(no_quota)
    )
    assert not run.complete
    assert run.generated == len(qa)  # generation continued without verdicts
    assert run.judge_stopped == "QuotaExhaustedError"
    assert run.stopped_because == "judge: QuotaExhaustedError"
    done = run_generation_eval(
        pipeline, qa, config="c", records_path=records, judge=ScriptedLLM(lambda m, s: JUDGE_JSON)
    )
    assert done.complete
    assert done.judged >= 1
    assert all(
        r.judge_faithfulness is not None
        for r in load_records(records).values()
        if r.answerable and not r.refused
    )


def test_report_writes_tables_and_charts(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    pytest.importorskip("matplotlib")
    from rbirag.eval.calibrate import calibrate_threshold
    from rbirag.eval.report import build_report, run_metadata

    rows = run_retrieval_ablation(
        fixture_docs, qa, settings=Settings(), embedder=FakeEmbedder(32), reranker=FakeReranker(),
        chunkers=["section"], sizes=[64, 128], modes=["bm25", "hybrid_rerank"],
    )  # fmt: skip
    records_path = tmp_path / "r.jsonl"
    run_generation_eval(make_pipeline(fixture_docs, FakeLLM()), qa, config="c", records_path=records_path,
                        judge=ScriptedLLM(lambda m, s: JUDGE_JSON))  # fmt: skip
    cal = calibrate_threshold([0.9, 0.2, 0.8], [True, False, True])
    out = tmp_path / "results"
    written = build_report(
        out, retrieval_rows=rows, generation_records=list(load_records(records_path).values()),
        calibration=cal, metadata=run_metadata(qa, judge_model="judge"), n_resamples=200,
    )  # fmt: skip
    names = {p.name for p in written}
    assert {"retrieval_ablation.csv", "generation.md", "calibration.png", "recall_at_k.png",
            "chunk_size_sweep.png", "latency_vs_quality.png", "mrr_by_config.png"} <= names  # fmt: skip
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    md = (out / "README.md").read_text()
    assert "section-128-hybrid_rerank" in md
    assert "human-verified: 0" in md
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["questions"] == 4


def test_incomplete_configs_are_not_headline_results() -> None:
    from rbirag.eval.gen_metrics import GenerationRecord
    from rbirag.eval.report import generation_markdown, generation_rows, incomplete_markdown, mix

    def rec(
        i: int, config: str, judged: bool, judge_model: str = "gemini-3.8-flash"
    ) -> GenerationRecord:
        return GenerationRecord(
            qa_id=f"q{i}", config=config, backend="openai_compatible", model="gemini-3.5-flash-lite",
            answerable=True, question="q", answer="a [1].", refused=False, refusal_reason=None,
            citation_valid=True, n_citations=1, n_invalid_citations=0, n_uncited_sentences=0,
            judge_model=judge_model if judged else None,
            judge_faithfulness=1.0 if judged else None, judge_relevance=1.0 if judged else None,
            judge_error=None if judged else "no JSON object found in judge output",
        )  # fmt: skip

    good = [
        rec(i, "good", True, "gemini-3.8-flash" if i < 7 else "gemini-3.7-flash") for i in range(10)
    ]
    bad = [rec(i, "bad", i < 3) for i in range(10)]
    partial = [rec(i, "partial", True) for i in range(4)]  # 4 of 10 planned
    rows = generation_rows(
        good + bad + partial, planned={"good": 10, "bad": 10, "partial": 10}, n_resamples=100
    )
    status = {r["config"]: r["status"] for r in rows}
    assert status == {
        "good": "complete",
        "bad": "incomplete (3/10)",
        "partial": "incomplete (4/10)",
    }
    headline = generation_markdown(rows)
    assert "| good |" in headline
    assert "| bad |" not in headline
    assert "| partial |" not in headline
    side = incomplete_markdown(rows)
    assert "incomplete (3/10)" in side
    assert "incomplete (4/10)" in side
    good_row = next(r for r in rows if r["config"] == "good")
    assert good_row["judge_model"] == "gemini-3.8-flash 70%, gemini-3.7-flash 30%"
    assert mix([None, None]) == "-"


def test_generation_eval_continues_after_rejected_items(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    from rbirag.generation.llm import RequestRejectedError

    records = tmp_path / "records.jsonl"
    calls = {"n": 0}

    def judge_reply(messages: Sequence[Message], system: str | None) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RequestRejectedError("HTTP 400: json_validate_failed")
        return JUDGE_JSON

    def answers(messages: Sequence[Message], system: str | None) -> str:
        if "FIRMS" in messages[-1].content.split("Question:")[-1]:
            raise RequestRejectedError("HTTP 400: bad request")
        return "Banks must comply with the rule [1]."

    pipeline = make_pipeline(fixture_docs, ScriptedLLM(answers))
    run = run_generation_eval(
        pipeline, qa, config="c", records_path=records, judge=ScriptedLLM(judge_reply)
    )
    assert run.complete
    recs = load_records(records)
    assert recs[("c", "fake", "firms")].error is not None
    assert recs[("c", "fake", "firms")].error.startswith("rejected:")
    judged = [r for r in recs.values() if r.judge_error and r.judge_error.startswith("rejected:")]
    assert len(judged) == 1
    assert sum(r.judge_faithfulness is not None for r in recs.values()) >= 1


def test_cached_answers_are_flagged_and_add_no_tokens(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    from rbirag.eval.report import generation_rows
    from rbirag.generation.cache import CachedLLM

    llm = CachedLLM(
        ScriptedLLM(lambda m, s: "Banks must update KYC [1].", model="m", backend="anthropic"),
        tmp_path / "cache",
    )
    pipeline = make_pipeline(fixture_docs, llm)
    judge = CachedLLM(
        ScriptedLLM(lambda m, s: JUDGE_JSON, model="j", backend="anthropic"), tmp_path / "cache"
    )
    run_generation_eval(pipeline, qa, config="a", records_path=tmp_path / "a.jsonl", judge=judge)
    run_generation_eval(pipeline, qa, config="b", records_path=tmp_path / "b.jsonl", judge=judge)
    first = [r for r in load_records(tmp_path / "a.jsonl").values() if r.model]
    second = [r for r in load_records(tmp_path / "b.jsonl").values() if r.model]
    assert first and not any(r.cached for r in first)
    assert second and all(r.cached for r in second)  # same prompts -> served from the cache
    assert all(r.judge_cached for r in second if r.judge_model)
    rows = {r["config"]: r for r in generation_rows(first + second, n_resamples=50)}
    assert rows["b"]["input_tokens"] == 0
    assert rows["b"]["judge_tokens"] == 0
    assert rows["b"]["n_cached"] == len(second)
    assert rows["a"]["input_tokens"] > 0
    assert "latency_ms_p50" not in rows["b"]  # cached answers have no meaningful latency


def test_unreachable_ollama_stops_cleanly_instead_of_crashing(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    import httpx

    from rbirag.generation.llm import LLMError
    from rbirag.generation.ollama import OllamaError, OllamaLLM

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    llm = OllamaLLM(client=httpx.Client(transport=httpx.MockTransport(down)))
    assert issubclass(OllamaError, LLMError)
    run = run_generation_eval(
        make_pipeline(fixture_docs, llm), qa, config="c", records_path=tmp_path / "r.jsonl"
    )
    assert not run.complete
    assert run.stopped_because == "OllamaError"


def test_judge_only_never_generates(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    from rbirag.eval.ablate import JudgeOnlyLLM

    records = tmp_path / "r.jsonl"
    # answers exist for the first two questions only, without verdicts
    run_generation_eval(
        make_pipeline(fixture_docs, FakeLLM()), qa[:2], config="c", records_path=records
    )
    judge = ScriptedLLM(lambda m, s: JUDGE_JSON, model="judge")
    run = run_generation_eval(
        make_pipeline(fixture_docs, JudgeOnlyLLM("fake")), qa, config="c", records_path=records,
        judge=judge, judge_only=True,
    )  # fmt: skip
    assert run.complete
    assert run.generated == 0
    assert run.skipped == len(qa)  # nothing generated: existing and missing answers are skipped
    recs = load_records(records)
    assert len(recs) == 2
    assert all(
        r.judge_faithfulness is not None for r in recs.values() if r.answerable and not r.refused
    )
    assert run.judged == len(judge.calls)


def test_deadline_stops_cleanly_and_flushes_records(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    records = tmp_path / "records.jsonl"
    pipeline = make_pipeline(fixture_docs, FakeLLM())
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0])
    run = run_generation_eval(
        pipeline, qa, config="c", records_path=records, deadline=50.0, clock=lambda: next(ticks)
    )
    assert not run.complete
    assert run.stopped_because == "deadline"
    assert run.generated == 2
    assert len(load_records(records)) == 2  # finished items are on disk
    rest = run_generation_eval(pipeline, qa, config="c", records_path=records)
    assert rest.complete
    assert (rest.generated, rest.skipped) == (len(qa) - 2, 2)


def test_deadline_in_the_past_does_no_work(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    records = tmp_path / "records.jsonl"
    run = run_generation_eval(
        make_pipeline(fixture_docs, FakeLLM()),
        qa,
        config="c",
        records_path=records,
        judge_only=True,
        deadline=1.0,
        clock=lambda: 2.0,
    )
    assert (run.generated, run.judged, run.stopped_because) == (0, 0, "deadline")


def test_shards_partition_ids_and_resume_from_seed(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    from rbirag.eval.cli import parse_shard, shard_of

    assert parse_shard("2/4") == (2, 4)
    import typer

    with pytest.raises(typer.BadParameter, match="2/4"):
        parse_shard("5/4")
    ids = [f"q{i}" for i in range(200)]
    groups = {n: [q for q in ids if shard_of(q, 4) == n] for n in (1, 2, 3, 4)}
    assert sorted(q for g in groups.values() for q in g) == sorted(ids)
    assert all(groups[n] for n in groups)  # roughly balanced, none empty
    assert all(shard_of(q, 4) == shard_of(q, 4) for q in ids)  # stable

    pipeline = make_pipeline(fixture_docs, FakeLLM())
    canonical = tmp_path / "records-fake-c.jsonl"
    run_generation_eval(pipeline, qa[:2], config="c", records_path=canonical)
    shard_file = tmp_path / "records-fake-c.shard1of2.jsonl"
    run = run_generation_eval(
        pipeline, qa, config="c", records_path=shard_file, seed_from=canonical
    )
    assert (run.generated, run.skipped) == (len(qa) - 2, 2)  # committed work is not redone


# ----------------------------------------------------------------------------- local judge


def low_judge_json(relevance: int = 2) -> str:
    return json.dumps(
        {
            "claims": [
                {"claim": "a", "supported": True, "source_ids": [1]},
                {"claim": "b", "supported": False, "source_ids": []},
            ],
            "relevance": relevance,
            "relevance_reason": "partial",
        }
    )


def test_switching_judge_rejudges_and_keeps_old_verdicts(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    records = tmp_path / "records.jsonl"
    pipeline = make_pipeline(fixture_docs, FakeLLM())
    big = ScriptedLLM(lambda m, s: JUDGE_JSON, model="api-120b")
    run_generation_eval(pipeline, qa, config="c", records_path=records, judge=big)
    small = ScriptedLLM(lambda m, s: low_judge_json(), model="local-7b")
    # same judge accepted: nothing to redo
    again = run_generation_eval(
        pipeline, qa, config="c", records_path=records, judge=small,
        accepted_judges={"api-120b", "local-7b"},
    )  # fmt: skip
    assert again.judged == 0
    run = run_generation_eval(
        pipeline, qa, config="c", records_path=records, judge=small, accepted_judges={"local-7b"}
    )
    judged = [r for r in load_records(records).values() if r.answerable and not r.refused]
    assert run.judged == len(judged) > 0
    assert run.judge_seconds >= 0
    for r in judged:
        assert r.judge_model == "local-7b"
        assert r.judge_faithfulness == 0.5
        assert r.verdicts["api-120b"] == {"faithfulness": 1.0, "relevance": 1.0}
        assert r.verdicts["local-7b"] == {"faithfulness": 0.5, "relevance": 0.25}
    # a third pass has nothing to do
    rerun = run_generation_eval(
        pipeline, qa, config="c", records_path=records, judge=small, accepted_judges={"local-7b"}
    )
    assert rerun.judged == 0


def test_failed_local_verdict_is_not_retried_forever(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    from rbirag.eval.ablate import needs_judge
    from rbirag.eval.records import merge_records

    records = tmp_path / "records.jsonl"
    pipeline = make_pipeline(fixture_docs, FakeLLM())
    big = ScriptedLLM(lambda m, s: JUDGE_JSON, model="api-120b")
    run_generation_eval(pipeline, qa, config="c", records_path=records, judge=big)
    committed = load_records(records)
    shard = tmp_path / "records.shard1of2.jsonl"
    broken = ScriptedLLM(lambda m, s: "not json", model="local-7b")
    run_generation_eval(
        pipeline, qa, config="c", records_path=shard, seed_from=records, judge=broken,
        accepted_judges={"local-7b"},
    )  # fmt: skip
    merged = merge_records([committed, load_records(shard)])
    for r in merged.values():
        if r.answerable and not r.refused:
            assert r.judge_model == "api-120b"  # the old verdict wins the merge...
            assert "local-7b" in r.verdicts  # ...but remembers the failed local attempt
            assert not needs_judge(r, {"local-7b"})


def test_plan_items_subset_only_backend(qa: list[QAPair]) -> None:
    from rbirag.eval.ablate import plan_items

    rows = [
        {"config": "fixed-128-bm25", "mode": "bm25", "chunk_ndcg@10": 0.9},
        {"config": "fixed-128-hybrid", "mode": "hybrid", "chunk_ndcg@10": 0.8},
        {"config": "fixed-128-dense", "mode": "dense", "chunk_ndcg@10": 0.7},
    ]
    plan = plan_generation(
        rows, qa, full_configs=1, subset_configs=2, subset_size=2,
        subset_only_backends=["openai_compatible"],
    )  # fmt: skip
    api = plan_items(plan.model_dump(), "openai_compatible")
    assert api == [("fixed-128-bm25", plan.subset_ids)]
    local = plan_items(plan.model_dump(), "ollama")
    assert [c for c, _ in local] == ["fixed-128-bm25", "fixed-128-hybrid", "fixed-128-dense"]
    assert local[0][1] == plan.all_ids
    calls = planned_calls(plan, qa, ["openai_compatible", "ollama"])
    assert calls["generate[openai_compatible]"] == len(plan.subset_ids)
    assert calls["generate[ollama]"] == len(plan.all_ids) + 2 * len(plan.subset_ids)
    # plans written before subset-only backends existed still load
    legacy = {k: v for k, v in plan.model_dump().items() if k != "subset_only_backends"}
    assert len(plan_items(legacy, "openai_compatible")) == 3


def test_sharded_judging_covers_every_answer_once(
    fixture_docs: list[Document], qa: list[QAPair], tmp_path: Path
) -> None:
    from rbirag.eval.cli import shard_of
    from rbirag.eval.records import merge_into

    gen = tmp_path / "results" / "generation"
    canonical = gen / "records-fake-c.jsonl"
    pipeline = make_pipeline(fixture_docs, FakeLLM())
    run_generation_eval(pipeline, qa, config="c", records_path=canonical)  # answers, no verdicts
    total = 0
    for i in (1, 2, 3):
        mine = [p for p in qa if shard_of(p.qa_id, 3) == i]
        judge = ScriptedLLM(lambda m, s: JUDGE_JSON, model="local-7b")
        run = run_generation_eval(
            pipeline, mine, config="c", records_path=gen / f"records-fake-c.shard{i}of3.jsonl",
            seed_from=canonical, judge=judge, judge_only=True, accepted_judges={"local-7b"},
        )  # fmt: skip
        total += run.judged
        assert len(judge.calls) == run.judged
    merge_into(gen)
    final = load_records(canonical)
    need = [r for r in final.values() if r.answerable and not r.refused]
    assert total == len(need) > 0
    assert all(r.judge_model == "local-7b" for r in need)
    assert sorted(p.name for p in gen.iterdir()) == ["records-fake-c.jsonl"]


def test_rows_label_sample_judge_and_complete() -> None:
    from rbirag.eval.gen_metrics import GenerationRecord
    from rbirag.eval.report import generation_markdown, generation_rows, judge_line

    def rec(i: int, backend: str) -> GenerationRecord:
        return GenerationRecord(
            qa_id=str(i), config="c", backend=backend, model="m", answerable=True,
            question="q?", answer="a [1].", refused=False, refusal_reason=None,
            citation_valid=True, n_citations=1, n_invalid_citations=0, n_uncited_sentences=0,
            judge_model="local-7b", judge_faithfulness=1.0, judge_relevance=1.0,
        )  # fmt: skip

    recs = [rec(i, "api") for i in range(9)] + [rec(i, "ollama") for i in range(10)]
    rows = generation_rows(
        recs,
        planned={"c/api": 10, "c/ollama": 12},
        samples={"c/api": "fixed subset", "c/ollama": "full test set"},
        n_resamples=100,
    )
    by = {r["backend"]: r for r in rows}
    assert by["api"]["status"] == "complete"  # 9/10 >= 90%
    assert by["api"]["sample"] == "fixed subset (n=10)"
    assert by["ollama"]["status"] == "incomplete (10/12)"
    md = generation_markdown(rows)
    assert "fixed subset (n=10)" in md and "local-7b" in md
    assert "ollama" not in md  # incomplete rows are not headline results
    line = judge_line(
        {
            "judge": {"model": "local-7b", "description": "local-7b, local", "seconds_per_verdict": 300.0},
            "judge_agreement": [
                {"other": "api-120b", "n": 12, "faithfulness": {"pearson": 0.71},
                 "relevance": {"kappa_quadratic": 0.55}}
            ],
        }
    )  # fmt: skip
    assert "Judge: local-7b, local" in line
    assert "300 s per verdict" in line
    assert "api-120b n=12: faithfulness r=0.71, relevance weighted kappa=0.55" in line
