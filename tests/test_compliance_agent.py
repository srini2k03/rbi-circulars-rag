from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from rbirag.chunking import build_chunker
from rbirag.compliance.agent import ComplianceAgent
from rbirag.compliance.baseline import FixedPipeline
from rbirag.compliance.checker import check_policy, render_markdown
from rbirag.compliance.context import SHORTENED, fit_history, history_tokens
from rbirag.compliance.evaluation.dataset import EvalItem, Label
from rbirag.compliance.evaluation.metrics import score_item
from rbirag.compliance.evaluation.run import load_predictions, run_items
from rbirag.compliance.guards import Proposal, decide
from rbirag.compliance.judge import (
    ClauseInput,
    ClauseJudge,
    Judgement,
    normalise_verdict,
    parse_json_object,
    verify_quote,
)
from rbirag.compliance.llm import (
    CachedChatModel,
    CacheMissError,
    ChatMessage,
    OpenAIToolChat,
    ScriptedChatModel,
    final,
    tool_call,
)
from rbirag.compliance.policy import policy_from_text
from rbirag.compliance.records import ClausePrediction, EvidenceQuote
from rbirag.compliance.regulations import RegulationStore
from rbirag.compliance.tools import ToolBox
from rbirag.config import Settings
from rbirag.generation.llm import QuotaExhaustedError, RequestRejectedError
from rbirag.index.build import build_index, chunk_documents
from rbirag.index.embedder import FakeEmbedder
from rbirag.retrieval.pipeline import make_retriever
from rbirag.retrieval.rerank import FakeReranker
from tests.conftest import make_doc

CB_RULE = (
    "the bank shall carry out periodic updation at least once in every two years for "
    "high-risk customers, once in every eight years for medium risk customers and once in "
    "every 10 years for low-risk customers"
)
OLD = make_doc(
    "RBI/2015-16/108\nDBR.AML.BC.No.81/14.01.001/2015-16\nFebruary 25, 2016\n\n"
    "38. Periodic updation of KYC shall be carried out at least once in every two years "
    "for high risk customers.",
    doc_id="rbi-500",
    title="Master Direction - Know Your Customer (KYC) Direction, 2016",
    date=dt.date(2016, 2, 25),
    circular_number="RBI/2015-16/108",
    ref_number="DBR.AML.BC.No.81/14.01.001/2015-16",
)
CB = make_doc(
    "RBI/2025-26/300\nDOR.AML.REC.1/14.01.002/2025-26\nNovember 28, 2025\n\n"
    f"42. Periodic updation of KYC\n\n(1) {CB_RULE}.\n\n"
    "50. Record management\n\n(1) maintain all necessary records of transactions for at "
    "least five years from the date of transaction.\n\n"
    "90. Repeal Provisions\n\nThe following stand repealed: "
    "DBR.AML.BC.No.81/14.01.001/2015-16 dated February 25, 2016.",
    doc_id="rbi-900",
    title="Reserve Bank of India (Commercial Banks - Know Your Customer) Directions, 2025",
    date=dt.date(2025, 11, 28),
    circular_number="RBI/2025-26/300",
    ref_number="DOR.AML.REC.1/14.01.002/2025-26",
)
UCB = make_doc(
    "RBI/2025-26/301\nDOR.AML.REC.2/14.01.003/2025-26\nNovember 28, 2025\n\n"
    "42. Periodic updation of KYC\n\n(1) the bank shall carry out periodic updation of KYC at "
    "least once in every two years for high-risk customers of urban co-operative banks.",
    doc_id="rbi-901",
    title="Reserve Bank of India (Urban Co-operative Banks - Know Your Customer) Directions, 2025",
    date=dt.date(2025, 11, 28),
    circular_number="RBI/2025-26/301",
    ref_number="DOR.AML.REC.2/14.01.003/2025-26",
)
CLAUSE = ClauseInput(
    label="3.2",
    text="Periodic updation of KYC shall be carried out at least once every three years for "
    "high risk customers.",
    policy_title="Sample Bank KYC Policy",
    context="Customer Due Diligence",
)


@pytest.fixture(scope="module")
def store() -> RegulationStore:
    s = Settings(chunker="fixed", chunk_size=64, chunk_overlap=8)
    docs = [OLD, CB, UCB]
    bundle = build_index(
        chunk_documents(docs, build_chunker("fixed", s)), embedder=FakeEmbedder(64)
    )
    retriever = make_retriever(
        s, bundle.store, bm25=bundle.bm25, dense=bundle.dense, reranker=FakeReranker()
    )
    return RegulationStore(retriever, {d.doc_id: d for d in docs}, top_k=5, fetch_max_tokens=300)


def is_judge(messages: Sequence[ChatMessage]) -> bool:
    return bool(messages) and (messages[0].content or "").startswith("You compare one clause")


def tool_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    return [m for m in messages if m.role == "tool"]


JUDGE_CONFLICT = {
    "verdict": "Non-compliant",
    "citations": ["rbi-900"],
    "evidence": [
        {"doc_id": "rbi-900", "quote": "at least once in every two years for high-risk customers"}
    ],
    "confidence": 0.9,
    "rationale": "Paragraph 42(1) requires updation at least every two years.",
}


def scripted_agent_model(
    planner: list[ChatMessage], judge: dict[str, Any] = JUDGE_CONFLICT
) -> ScriptedChatModel:
    steps = list(planner)

    def reply(messages: Sequence[ChatMessage], tools: Any) -> ChatMessage:
        if is_judge(messages):
            return ChatMessage(role="assistant", content="```json\n" + json.dumps(judge) + "\n```")
        return steps.pop(0)

    return ScriptedChatModel(reply)


def agent(store: RegulationStore, chat: Any, **kw: Any) -> ComplianceAgent:
    return ComplianceAgent(store, chat, judge_context_tokens=600, **kw)


# ----------------------------------------------------------------------------- search / tools


def test_search_filters_other_entities_and_shows_status(store: RegulationStore) -> None:
    result = store.search("periodic updation of KYC high-risk customers")
    ids = [h.doc_id for h in result.hits]
    assert "rbi-901" not in ids and "rbi-900" in ids
    assert result.excluded_other_entity == 1
    cb = next(h for h in result.hits if h.doc_id == "rbi-900")
    assert cb.entity == "Commercial Banks"
    old = next(h for h in result.hits if h.doc_id == "rbi-500")
    assert old.replaced_by == ["rbi-900"]
    unfiltered = store.search("periodic updation of KYC high-risk customers", entity_filter=False)
    assert "rbi-901" in [h.doc_id for h in unfiltered.hits]


@pytest.mark.parametrize(
    ("name", "args", "error"),
    [
        ("browse_web", '{"q": "x"}', "unknown tool"),
        ("search_regulations", "{not json", "not valid JSON"),
        ("search_regulations", '["q"]', "must be a JSON object"),
        ("search_regulations", "{}", "missing argument"),
        ("search_regulations", '{"query": "kyc", "mode": "x"}', "unexpected argument"),
        ("search_regulations", '{"query": "kyc", "top_k": 50}', "top_k"),
        ("search_regulations", '{"query": "kyc", "date_from": "2025/01/01"}', "YYYY-MM-DD"),
        ("search_regulations", '{"query": "k"}', "query must be"),
        ("fetch_notification", '{"doc_id": "rbi-123456"}', "no notification"),
        ("compare_clause", '{"doc_ids": []}', "doc_ids must be"),
        ("compare_clause", '{"doc_ids": "rbi-900"}', "doc_ids must be"),
        ("compare_clause", '{"doc_ids": ["rbi-900", "rbi-1"]}', "no notification"),
    ],
)
def test_invalid_tool_calls_are_recorded_not_executed(
    store: RegulationStore, name: str, args: str, error: str
) -> None:
    judge = ClauseJudge(store, ScriptedChatModel([]), as_of="2026-11-11", entity="Commercial Banks")
    box = ToolBox(store, judge, CLAUSE)
    outcome = box.run(name, args)
    assert not outcome.record.valid and error in (outcome.record.error or "")
    assert outcome.content.startswith("ERROR:")
    assert box.calls == [outcome.record] and not box.seen_doc_ids


def test_valid_calls_track_seen_ids_and_repeats(store: RegulationStore) -> None:
    judge = ClauseJudge(store, ScriptedChatModel([]), as_of="2026-11-11", entity="Commercial Banks")
    box = ToolBox(store, judge, CLAUSE)
    first = box.run("search_regulations", '{"query": "periodic updation KYC"}')
    assert first.record.valid and first.retrieval_confidence is not None
    payload = json.loads(first.content)
    assert {h["doc_id"] for h in payload["hits"]} == box.seen_doc_ids
    again = box.run("search_regulations", '{"query": "periodic updation KYC"}')
    assert again.content.endswith("(repeated call)") and len(box.calls) == 2
    fetched = box.run("fetch_notification", '{"doc_id": "12345"}')
    assert not fetched.record.valid
    fetched = box.run("fetch_notification", '{"doc_id": "900", "focus": "periodic updation"}')
    view = json.loads(fetched.content)
    assert view["doc_id"] == "rbi-900" and view["entity"] == "Commercial Banks"
    assert any("two years" in p["text"] for p in view["passages"])


# ----------------------------------------------------------------------------- judge


def test_judge_parsing_and_quote_verification(store: RegulationStore) -> None:
    chat = ScriptedChatModel(
        [
            ChatMessage(
                role="assistant",
                content="Here you go:\n"
                + json.dumps(
                    {
                        **JUDGE_CONFLICT,
                        "citations": ["rbi-900", "rbi-777"],
                        "evidence": [
                            {
                                "doc_id": "rbi-900",
                                "quote": "at least once in every two years for high-risk customers",
                            },
                            {
                                "doc_id": "rbi-900",
                                "quote": "once in every three years for all customers of the bank",
                            },
                            {
                                "doc_id": "rbi-777",
                                "quote": "not shown to the judge at all, so it is dropped",
                            },
                        ],
                        "confidence": 85,
                    }
                ),
            )
        ]
    )
    judge = ClauseJudge(
        store, chat, as_of="2026-11-11", entity="Commercial Banks", context_tokens=600
    )
    judgement, _result, shown = judge.judge(CLAUSE, ["rbi-900"])
    assert judgement.verdict == "conflict" and judgement.citations == ["rbi-900"]
    assert [e.verified for e in judgement.evidence] == [True, False]
    assert judgement.confidence == 0.85
    user = chat.calls[0][0][1].content or ""
    assert (
        "[rbi-900] Reserve Bank of India (Commercial Banks" in user
        and "addressed to: Commercial Banks" in user
    )
    assert chat.calls[0][1] is None  # the judge gets no tools
    assert any("two years" in t for t in shown)


def test_judge_unparseable_reply_abstains() -> None:
    j = ClauseJudge.parse("I think it is fine.", ["rbi-900"], [])
    assert j.verdict == "insufficient_evidence" and not j.parsed and j.confidence == 0.0
    j = ClauseJudge.parse('{"verdict": "maybe"}', ["rbi-900"], [])
    assert not j.parsed


def test_helpers() -> None:
    assert parse_json_object('noise {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert parse_json_object("[1, 2]") is None and parse_json_object(None) is None
    assert normalise_verdict("Partially Compliant") == "gap"
    assert normalise_verdict("not-applicable") == "not_applicable"
    assert normalise_verdict("N/A") == "not_applicable"
    assert normalise_verdict(3) is None
    text = [
        "The bank shall \u201cpreserve\u201d records for at least five years \u2013 from the date."
    ]
    assert verify_quote(
        'the bank shall "preserve" records for at least five years - from the date', text
    )
    assert verify_quote(
        "preserve records for at least five yeers - from the date", text
    )  # small edit
    assert not verify_quote("records", text)  # too short to verify
    assert not verify_quote("records shall be kept for ten years after closure", text)


# ----------------------------------------------------------------------------- guards


def judgement(verified: bool = True, **kw: Any) -> Judgement:
    base: dict[str, Any] = {
        "verdict": "conflict",
        "citations": ["rbi-900"],
        "evidence": [EvidenceQuote(doc_id="rbi-900", quote="q" * 20, verified=verified)],
        "confidence": 0.9,
        "rationale": "r",
        "doc_ids": ["rbi-900"],
    }
    base.update(kw)
    return Judgement(**base)


TITLES = {d.doc_id: d.title for d in (OLD, CB, UCB)}


def run_decide(
    proposal: Proposal | None, j: Judgement | None, seen: set[str] | None = None, **kw: Any
):  # type: ignore[no-untyped-def]
    return decide(
        proposal,
        j,
        seen_doc_ids=seen if seen is not None else {"rbi-900", "rbi-901", "rbi-500"},
        titles=TITLES,
        replaced_by=lambda d: ["rbi-900"] if d == "rbi-500" else [],
        **kw,
    )


def test_decide_drops_unseen_and_wrong_entity_citations() -> None:
    d = run_decide(
        Proposal(verdict="conflict", citations=["rbi-900", "rbi-901", "rbi-404"], confidence=0.8),
        judgement(),
    )
    assert d.verdict == "conflict" and d.citations == ["rbi-900"]
    assert any("never shown" in n for n in d.notes) and any(
        "another entity type" in n for n in d.notes
    )
    assert d.confidence == pytest.approx(0.8 * 0.85)


def test_decide_abstains_without_support() -> None:
    d = run_decide(Proposal(verdict="compliant", citations=["rbi-900"]), None)
    assert d.verdict == "insufficient_evidence" and "no compare_clause" in d.notes[-2]
    assert d.citations == [] and "considered: rbi-900" in d.notes[-1]
    d = run_decide(
        Proposal(verdict="conflict", citations=["rbi-404"]),
        judgement(citations=[]),
        seen={"rbi-900"},
    )
    assert d.verdict == "insufficient_evidence" and any("no valid citation" in n for n in d.notes)
    d = run_decide(None, judgement(verified=False))
    assert d.verdict == "insufficient_evidence" and any(
        "no evidence quote could be verified" in n for n in d.notes
    )
    d = run_decide(None, judgement(verified=False), require_verified_quote=False)
    assert d.verdict == "conflict" and d.confidence == pytest.approx(0.9 * 0.7)


def test_decide_replaced_citation_and_threshold() -> None:
    j = judgement(
        citations=["rbi-500"],
        evidence=[EvidenceQuote(doc_id="rbi-500", quote="q" * 20, verified=True)],
    )
    d = run_decide(None, j)
    assert d.verdict == "conflict" and d.confidence == pytest.approx(0.9 * 0.8)
    assert any("marked as replaced" in n for n in d.notes)
    d = run_decide(None, j, abstain_threshold=0.75)
    assert d.verdict == "insufficient_evidence" and any("below threshold" in n for n in d.notes)


def test_decide_non_substantive_and_empty() -> None:
    d = run_decide(Proposal(verdict="not_applicable", citations=["rbi-900"], confidence=0.9), None)
    assert d.verdict == "not_applicable" and d.citations == []
    d = run_decide(None, None)
    assert d.verdict == "insufficient_evidence" and d.confidence == 0.0


# ----------------------------------------------------------------------------- agent loop


def test_agent_search_compare_final(store: RegulationStore) -> None:
    chat = scripted_agent_model(
        [
            tool_call("search_regulations", {"query": "periodic updation KYC high risk customers"}),
            tool_call("compare_clause", {"doc_ids": ["rbi-900"]}, call_id="c2"),
            final(
                {
                    "verdict": "conflict",
                    "citations": ["rbi-900"],
                    "confidence": 0.85,
                    "rationale": "Two years, not three.",
                }
            ),
        ]
    )
    pred = agent(store, chat).check(CLAUSE, item_id="kyc/3.2")
    assert pred.error is None and pred.verdict == "conflict" and pred.citations == ["rbi-900"]
    assert [c.name for c in pred.tool_calls] == ["search_regulations", "compare_clause"]
    assert all(c.valid for c in pred.tool_calls) and pred.llm_calls == 4  # 3 planner + 1 judge
    assert pred.evidence and pred.evidence[0].verified
    assert pred.confidence == pytest.approx(0.85) and pred.input_tokens > 0
    # the planner saw the tool results and the tool schema
    planner_calls = [c for c in chat.calls if not is_judge(c[0])]
    assert planner_calls[0][1] is not None and len(tool_messages(planner_calls[-1][0])) == 2

    label = Label(
        item_id="kyc/3.2",
        policy_id="kyc",
        section="3",
        number="3.2",
        text=CLAUSE.text,
        verdict="conflict",
        citations=[["rbi-900"]],
        rationale="two years, not three",
        evidence=[{"doc_id": "rbi-900", "quote": CB_RULE}],  # type: ignore[list-item]
        expected_tools=["search_regulations", "compare_clause"],
    )
    score = score_item(
        EvalItem(label=label, split="dev", policy_title="P", section_title="S"), pred, titles=TITLES
    )
    assert score.success and score.tool_selection_ok and score.failure is None


def test_agent_answers_out_of_scope_without_tools(store: RegulationStore) -> None:
    chat = ScriptedChatModel(
        [
            final(
                {
                    "verdict": "not_applicable",
                    "citations": [],
                    "confidence": 0.95,
                    "rationale": "Internal.",
                }
            )
        ]
    )
    clause = ClauseInput(label="6.1", text="Staff get twelve days of casual leave a year.")
    pred = agent(store, chat).check(clause)
    assert pred.verdict == "not_applicable" and pred.steps == 0 and pred.llm_calls == 1


def test_agent_guardrails_reject_unsupported_answer(store: RegulationStore) -> None:
    chat = ScriptedChatModel(
        [final({"verdict": "compliant", "citations": ["rbi-900"], "confidence": 0.99})]
    )
    pred = agent(store, chat).check(CLAUSE)
    assert pred.verdict == "insufficient_evidence" and pred.citations == []
    assert any("never shown" in n for n in pred.notes)


def test_agent_step_budget_forces_final_answer(store: RegulationStore) -> None:
    searches = [
        tool_call("search_regulations", {"query": f"kyc query number {i}"}, call_id=f"s{i}")
        for i in range(3)
    ]
    chat = scripted_agent_model(
        [*searches, final({"verdict": "insufficient_evidence", "citations": [], "confidence": 0.3})]
    )
    pred = agent(store, chat, max_steps=3).check(CLAUSE)
    assert pred.verdict == "insufficient_evidence" and pred.steps == 3
    last_messages, last_tools = chat.calls[-1]
    assert last_tools is None and "tool budget is used up" in (last_messages[-1].content or "")


def test_agent_repairs_malformed_final_and_falls_back_to_judgement(store: RegulationStore) -> None:
    chat = scripted_agent_model(
        [
            tool_call("search_regulations", {"query": "periodic updation KYC"}),
            tool_call("compare_clause", {"doc_ids": ["rbi-900"]}, call_id="c2"),
            ChatMessage(role="assistant", content="The clause conflicts."),
            ChatMessage(role="assistant", content="Still not JSON."),
        ]
    )
    pred = agent(store, chat).check(CLAUSE)
    assert pred.verdict == "conflict" and pred.citations == ["rbi-900"]
    assert any("adopted the comparison" in n for n in pred.notes)


def test_agent_accepts_tool_calls_written_as_text(store: RegulationStore) -> None:
    chat = scripted_agent_model(
        [
            ChatMessage(
                role="assistant",
                content=json.dumps(
                    {"tool": "search_regulations", "arguments": {"query": "periodic updation KYC"}}
                ),
            ),
            ChatMessage(
                role="assistant",
                content='{"name": "compare_clause", "arguments": {"doc_ids": ["rbi-900"]}}',
            ),
            final({"verdict": "conflict", "citations": ["rbi-900"], "confidence": 0.8}),
        ]
    )
    pred = agent(store, chat).check(CLAUSE)
    assert [c.name for c in pred.tool_calls] == ["search_regulations", "compare_clause"]
    assert pred.verdict == "conflict"


def test_agent_runs_compare_when_the_planner_skips_it(store: RegulationStore) -> None:
    chat = scripted_agent_model(
        [
            tool_call("search_regulations", {"query": "periodic updation KYC high risk customers"}),
            tool_call("fetch_notification", {"doc_id": "rbi-900"}, call_id="f2"),
            # a substantive answer without compare_clause: the agent compares for it
            final({"verdict": "compliant", "citations": ["rbi-900"], "confidence": 0.9}),
        ]
    )
    pred = agent(store, chat).check(CLAUSE)
    assert [c.name for c in pred.tool_calls] == [
        "search_regulations",
        "fetch_notification",
        "compare_clause",
    ]
    assert pred.tool_calls[-1].arguments == {"doc_ids": ["rbi-900"]}
    # the judge's verdict is adopted, not the planner's
    assert pred.verdict == "conflict" and pred.citations == ["rbi-900"]
    assert pred.evidence and pred.evidence[0].verified
    assert any("without compare_clause" in n for n in pred.notes)


def test_agent_auto_compare_needs_a_notification(store: RegulationStore) -> None:
    chat = scripted_agent_model(
        [final({"verdict": "compliant", "citations": ["rbi-900"], "confidence": 0.9})]
    )
    pred = agent(store, chat).check(CLAUSE)  # rbi-900 never shown: nothing to compare
    assert pred.verdict == "insufficient_evidence" and pred.steps == 0


def test_fit_history_shortens_oldest_results_first() -> None:
    search = json.dumps(
        {
            "hits": [
                {
                    "doc_id": f"rbi-{i}",
                    "title": f"T{i}",
                    "entity": "Commercial Banks",
                    "snippet": "s" * 2000,
                }
                for i in range(3)
            ],
            "confidence": 0.5,
        }
    )
    fetch = json.dumps(
        {
            "doc_id": "rbi-1",
            "title": "T1",
            "passages": [{"para": "42", "heading": "h", "text": "p" * 6000}],
        }
    )
    compare = json.dumps({"verdict": "gap", "citations": ["rbi-1"]})
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="clause"),
        ChatMessage(role="tool", tool_call_id="a", content=search),
        ChatMessage(role="tool", tool_call_id="b", content=fetch),
        ChatMessage(role="tool", tool_call_id="c", content=compare),
    ]
    assert history_tokens(messages) > 3000
    assert fit_history(messages, 2000) == 1  # the search result alone is enough
    assert messages[2].content and messages[2].content.startswith(SHORTENED)
    assert "rbi-2" in messages[2].content and "sss" not in messages[2].content
    assert messages[3].content == fetch
    assert fit_history(messages, 300) == 1
    assert '"passages_read":["42"]' in (messages[3].content or "")
    assert messages[4].content == compare  # comparisons are kept
    assert fit_history(messages, 10) == 0  # nothing left to shorten


def test_agent_retries_a_request_rejected_as_too_large(store: RegulationStore) -> None:
    steps = [
        tool_call("search_regulations", {"query": "periodic updation KYC high risk customers"}),
        tool_call("compare_clause", {"doc_ids": ["rbi-900"]}, call_id="c2"),
    ]
    rejected: list[int] = []

    def reply(messages: Sequence[ChatMessage], tools: Any) -> ChatMessage:
        if is_judge(messages):
            return ChatMessage(role="assistant", content=json.dumps(JUDGE_CONFLICT))
        if steps:
            return steps.pop(0)
        if not rejected and not any((m.content or "").startswith(SHORTENED) for m in messages):
            rejected.append(1)
            raise RequestRejectedError("HTTP 413: Request too large for model")
        return final({"verdict": "conflict", "citations": ["rbi-900"], "confidence": 0.8})

    pred = agent(store, ScriptedChatModel(reply), planner_context_tokens=100_000).check(CLAUSE)
    assert rejected and pred.error is None and pred.verdict == "conflict"


def test_agent_errors(store: RegulationStore) -> None:
    def boom(messages: Any, tools: Any) -> ChatMessage:
        raise ValueError("provider exploded")

    pred = agent(store, ScriptedChatModel(boom)).check(CLAUSE, item_id="x")
    assert (
        pred.error and "provider exploded" in pred.error and pred.verdict == "insufficient_evidence"
    )

    def quota(messages: Any, tools: Any) -> ChatMessage:
        raise QuotaExhaustedError("daily quota")

    with pytest.raises(QuotaExhaustedError):
        agent(store, ScriptedChatModel(quota)).check(CLAUSE)


# ----------------------------------------------------------------------------- baseline, checker, runner


def test_baseline_one_search_one_judgement(store: RegulationStore) -> None:
    chat = scripted_agent_model([])
    pred = FixedPipeline(store, chat, judge_context_tokens=600).check(CLAUSE, item_id="kyc/3.2")
    assert pred.system == "baseline" and pred.llm_calls == 1
    assert [c.name for c in pred.tool_calls] == ["search_regulations", "compare_clause"]
    assert pred.verdict == "conflict" and pred.citations == ["rbi-900"]
    assert pred.tool_calls[1].arguments["doc_ids"][0] in {"rbi-900", "rbi-500"}


def test_check_policy_report(store: RegulationStore) -> None:
    doc = policy_from_text(
        "# Sample KYC Policy\n\n## 3. Customer Due Diligence\n\n"
        f"3.2 {CLAUSE.text}\n\n## 6. Administration\n\n6.1 Staff get twelve days of casual leave a year.\n",
        name="kyc.md",
    )

    def reply(messages: Sequence[ChatMessage], tools: Any) -> ChatMessage:
        if is_judge(messages):
            return ChatMessage(role="assistant", content=json.dumps(JUDGE_CONFLICT))
        user = messages[1].content or ""
        if "casual leave" in user:
            return final(
                {"verdict": "not_applicable", "confidence": 0.9, "rationale": "Internal HR."}
            )
        n = len(tool_messages(messages))
        if n == 0:
            return tool_call("search_regulations", {"query": "periodic updation KYC"})
        if n == 1:
            return tool_call("compare_clause", {"doc_ids": ["rbi-900"]})
        return final(
            {
                "verdict": "conflict",
                "citations": ["rbi-900"],
                "confidence": 0.8,
                "rationale": "Two years.",
            }
        )

    progress: list[str] = []
    report = check_policy(
        doc,
        agent(store, ScriptedChatModel(reply)),
        store,
        as_of="2026-11-11",
        entity="Commercial Banks",
        on_progress=lambda i, n, f: progress.append(f"{i}/{n}"),
    )
    assert progress == ["1/2", "2/2"]
    assert report.summary["conflict"] == 1 and report.summary["not_applicable"] == 1
    conflict = report.findings[0]
    assert conflict.clause.label == "3.2" and conflict.sources[0].doc_id == "rbi-900"
    assert conflict.sources[0].quotes and report.llm_calls == 5
    md = render_markdown(report)
    assert "| 3.2 | CONFLICT | 0.80 | rbi-900 |" in md and "[rbi-900](" in md


def test_run_items_is_resumable(store: RegulationStore, tmp_path: Path) -> None:
    labels = [
        Label(
            item_id=f"kyc/{n}",
            policy_id="kyc",
            section="1",
            number=n,
            text=CLAUSE.text,
            verdict="not_applicable",
            rationale="internal matter",
            tags=["out_of_scope"],
        )
        for n in ("1.1", "1.2")
    ]
    items = [EvalItem(label=lb, split="dev", policy_title="P", section_title="S") for lb in labels]

    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def check(self, clause: ClauseInput, *, item_id: str | None = None) -> ClausePrediction:
            self.calls += 1
            error = "timeout" if item_id == "kyc/1.2" and self.calls <= 2 else None
            return ClausePrediction(item_id=item_id or "", verdict="not_applicable", error=error)

    out = tmp_path / "preds.jsonl"
    checker = Flaky()
    first = run_items(items, checker, out)
    assert first["kyc/1.2"].error == "timeout" and checker.calls == 2
    second = run_items(items, checker, out)  # 1.1 skipped, errored 1.2 re-run
    assert checker.calls == 3 and second["kyc/1.2"].error is None
    assert set(load_predictions(out)) == {"kyc/1.1", "kyc/1.2"}
    run_items(items, checker, out, resume=False)
    assert checker.calls == 5 and len(out.read_text().splitlines()) == 2


# ----------------------------------------------------------------------------- chat models


def test_cached_chat_model_modes(tmp_path: Path) -> None:
    inner = ScriptedChatModel([final({"verdict": "compliant"}), final({"verdict": "gap"})])
    msgs = [ChatMessage(role="user", content="hello")]
    cached = CachedChatModel(inner, tmp_path, "use")
    first = cached.chat(msgs, max_tokens=10)
    again = cached.chat(msgs, max_tokens=10)
    assert not first.cached and again.cached and len(inner.calls) == 1
    replay = CachedChatModel(inner, tmp_path, "replay")
    assert replay.chat(msgs, max_tokens=10).cached
    with pytest.raises(CacheMissError):
        replay.chat(msgs, max_tokens=11)  # a different request
    refresh = CachedChatModel(inner, tmp_path, "refresh")
    assert '"gap"' in (refresh.chat(msgs, max_tokens=10).message.content or "")
    assert '"gap"' in (
        CachedChatModel(inner, tmp_path, "replay").chat(msgs, max_tokens=10).message.content or ""
    )


def test_openai_tool_chat_request_and_response() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {
                                        "name": "search_regulations",
                                        "arguments": {"query": "kyc"},
                                    },
                                },
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 15},
            },
        )

    chat = OpenAIToolChat(
        "test-model",
        base_url="https://llm.test/v1",
        api_key="k",
        backend_name="groq",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "search_regulations", "parameters": {"type": "object"}},
        }
    ]
    history = [
        ChatMessage(role="system", content="sys"),
        tool_call("fetch_notification", {"doc_id": "rbi-1"}, call_id="old"),
        ChatMessage(role="tool", tool_call_id="old", content="{}"),
    ]
    result = chat.chat(history, tools=tools, max_tokens=50)
    body = seen["body"]
    assert body["tools"] == tools and body["tool_choice"] == "auto" and body["temperature"] == 0
    assert body["messages"][1]["tool_calls"][0]["function"]["name"] == "fetch_notification"
    assert body["messages"][2] == {"role": "tool", "content": "{}", "tool_call_id": "old"}
    call = result.message.tool_calls[0]
    assert call.id == "call_9" and json.loads(call.arguments) == {"query": "kyc"}
    assert (
        result.usage.input_tokens == 120
        and result.backend == "groq"
        and result.model == "test-model"
    )
    chat.chat([ChatMessage(role="user", content="x")], json_object=True)
    assert (
        seen["body"]["response_format"] == {"type": "json_object"} and "tools" not in seen["body"]
    )


# ----------------------------------------------------------------------------- factory and CLI


def test_build_chat_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from rbirag.compliance.llm import ReplayOnlyModel, build_chat_model

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    s = Settings(compliance_llm_cache_dir=tmp_path, llm_cache_dir=tmp_path)
    groq = build_chat_model(s)
    assert isinstance(groq, CachedChatModel) and groq.mode == "use"
    assert isinstance(groq.inner, OpenAIToolChat)
    assert groq.model == "openai/gpt-oss-20b" and groq.backend == "groq"
    assert groq.inner.chain == ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    gemini = build_chat_model(s, "gemini", cache_mode="off")
    assert isinstance(gemini, OpenAIToolChat) and "generativelanguage" in gemini.base_url
    assert gemini.model.startswith("gemini-")
    ollama = build_chat_model(s, "ollama", model="qwen2.5:7b-instruct")
    assert isinstance(ollama, CachedChatModel) and isinstance(ollama.inner, OpenAIToolChat)
    assert (
        ollama.inner.base_url == "http://localhost:11434/v1/"
        and ollama.model == "qwen2.5:7b-instruct"
    )
    # replay needs no credentials: a stub with the configured model id answers from the cache
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    replay = build_chat_model(s, "anthropic", cache_mode="replay")
    assert isinstance(replay, CachedChatModel) and replay.mode == "replay"
    assert isinstance(replay.inner, ReplayOnlyModel) and replay.model == "claude-haiku-5-5"
    with pytest.raises(CacheMissError):
        replay.chat([ChatMessage(role="user", content="never recorded")])
    pinned = build_chat_model(
        s.model_copy(
            update={"compliance_model": "qwen/qwen3.8-27b", "compliance_model_fallbacks": []}
        ),
        cache_mode="off",
    )
    assert isinstance(pinned, OpenAIToolChat) and pinned.chain == ["qwen/qwen3.8-27b"]
    with pytest.raises(ValueError, match="unknown chat backend"):
        build_chat_model(s, "openai")


def test_cli_check_and_eval_run(
    store: RegulationStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import rbirag.compliance.cli as comply_cli
    from rbirag.cli import app

    def reply(messages: Sequence[ChatMessage], tools: Any) -> ChatMessage:
        return ChatMessage(role="assistant", content=json.dumps(JUDGE_CONFLICT))

    def fake_build(system: str, **kw: Any):  # type: ignore[no-untyped-def]
        assert system == "baseline"
        return FixedPipeline(store, ScriptedChatModel(reply), judge_context_tokens=600), store

    monkeypatch.setattr(comply_cli, "build_checker", fake_build)
    policy = tmp_path / "p.md"
    policy.write_text(f"# P\n\n## 3. CDD\n\n3.2 {CLAUSE.text}\n", encoding="utf-8")
    runner = CliRunner()
    md_out, json_out = tmp_path / "r.md", tmp_path / "r.json"
    res = runner.invoke(
        app,
        [
            "comply",
            "check",
            str(policy),
            "--system",
            "baseline",
            "--out",
            str(md_out),
            "--json-out",
            str(json_out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert (
        "| 3.2 | CONFLICT |" in md_out.read_text()
        and json.loads(json_out.read_text())["summary"]["conflict"] == 1
    )

    root = tmp_path / "set"
    root.mkdir()
    (root / "policies.json").write_text(
        json.dumps(
            [
                {
                    "policy_id": "kyc",
                    "title": "KYC",
                    "topic": "kyc",
                    "split": "dev",
                    "sections": [{"number": "3", "title": "CDD"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    label = {
        "item_id": "kyc/3.2",
        "policy_id": "kyc",
        "section": "3",
        "number": "3.2",
        "text": CLAUSE.text,
        "verdict": "conflict",
        "citations": [["rbi-900"]],
        "rationale": "two years, not three",
        "evidence": [{"doc_id": "rbi-900", "quote": CB_RULE}],
    }
    (root / "labels.jsonl").write_text(json.dumps(label) + "\n", encoding="utf-8")
    preds = tmp_path / "preds.jsonl"
    res = runner.invoke(
        app,
        [
            "comply",
            "eval",
            "run",
            "--system",
            "baseline",
            "--split",
            "dev",
            "--root",
            str(root),
            "--out",
            str(preds),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "| verdict accuracy | 1.000" in res.output
    assert load_predictions(preds)["kyc/3.2"].verdict == "conflict"


def test_eval_replay_reproduces_detects_changes_and_staleness(
    store: RegulationStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import rbirag.compliance.cli as comply_cli
    from rbirag.cli import app
    from rbirag.compliance.llm import ReplayOnlyModel

    cache_dir = tmp_path / "llm_cache"
    prompt_suffix = {"value": ""}

    def reply(messages: Sequence[ChatMessage], tools: Any) -> ChatMessage:
        return ChatMessage(role="assistant", content=json.dumps(JUDGE_CONFLICT))

    def fake_build(system: str, *, settings: Any = None, backend: Any = None, model: Any = None):  # type: ignore[no-untyped-def]
        mode = settings.compliance_llm_cache_mode if settings is not None else "use"
        inner: Any = (
            ReplayOnlyModel(model or "m", "groq")
            if mode == "replay"
            else ScriptedChatModel(reply, model=model or "m")
        )
        chat = CachedChatModel(inner, cache_dir, mode)
        pipe = FixedPipeline(
            store,
            chat,
            judge_context_tokens=600,
            entity=prompt_suffix["value"] or "Commercial Banks",
        )
        return pipe, store

    monkeypatch.setattr(comply_cli, "build_checker", fake_build)
    root = tmp_path / "set"
    root.mkdir()
    (root / "policies.json").write_text(
        json.dumps(
            [
                {
                    "policy_id": "kyc",
                    "title": "KYC",
                    "topic": "kyc",
                    "split": "dev",
                    "sections": [{"number": "3", "title": "CDD"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    label = {
        "item_id": "kyc/3.2",
        "policy_id": "kyc",
        "section": "3",
        "number": "3.2",
        "text": CLAUSE.text,
        "verdict": "conflict",
        "citations": [["rbi-900"]],
        "rationale": "two years, not three",
        "evidence": [{"doc_id": "rbi-900", "quote": CB_RULE}],
    }
    (root / "labels.jsonl").write_text(json.dumps(label) + "\n", encoding="utf-8")
    runs, preds = tmp_path / "runs", tmp_path / "baseline-dev-groq.jsonl"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "comply",
            "eval",
            "run",
            "--system",
            "baseline",
            "--split",
            "dev",
            "--root",
            str(root),
            "--out",
            str(preds),
            "--runs-dir",
            str(runs),
            "--backend",
            "groq",
            "--model",
            "m",
        ],
    )
    assert res.exit_code == 0, res.output
    args = ["comply", "eval", "replay", "--runs-dir", str(runs), "--root", str(root)]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    assert "OK   baseline/dev/groq: 1 items replayed" in res.output
    # a recorded prediction that the code no longer reproduces -> regression
    recorded = json.loads(preds.read_text())
    preds.write_text(json.dumps({**recorded, "verdict": "gap"}) + "\n", encoding="utf-8")
    res = runner.invoke(app, args)
    assert res.exit_code == 1 and "DIFF" in res.output and "gap" in res.output
    # different prompts -> requests not in the cache -> stale, needs a live run
    preds.write_text(json.dumps(recorded) + "\n", encoding="utf-8")
    prompt_suffix["value"] = "Commercial Bank branches"  # changes the judge's system prompt
    res = runner.invoke(app, args)
    assert res.exit_code == 2 and "STALE" in res.output
