from __future__ import annotations

import pytest

from rbirag.eval.judge import (
    JudgeParseError,
    build_judge_prompt,
    extract_json_object,
    judge_answer,
    parse_judge_output,
)
from rbirag.eval.nli import FakeNLI, nli_check
from rbirag.generation.citations import parse_citations
from rbirag.generation.llm import LLMResponse, ScriptedLLM

CANNED_PLAIN = (
    '{"claims": [{"claim": "KYC every two years", "supported": true, "source_ids": [1]},'
    ' {"claim": "Fee is Rs 500", "supported": false, "source_ids": []}],'
    ' "relevance": 4, "relevance_reason": "mostly answers"}'
)
CANNED_FENCED = """Here is my evaluation:
```json
{"claims": [{"claim": "A", "supported": "yes", "source_ids": ["[2]"]}], "relevance": "5", "relevance_reason": "x"}
```
Hope this helps {not json}."""


def test_parse_plain_json() -> None:
    v = parse_judge_output(CANNED_PLAIN)
    assert v.faithfulness == 0.5
    assert v.relevance == 4
    assert v.relevance_score == 0.75
    assert v.claims[0].source_ids == [1]


def test_parse_fenced_with_chatter_and_string_types() -> None:
    v = parse_judge_output(CANNED_FENCED)
    assert v.claims[0].supported is True
    assert v.claims[0].source_ids == [2]
    assert v.relevance == 5


def test_parse_clamps_and_handles_empty_claims() -> None:
    v = parse_judge_output('{"claims": [], "relevance": 9, "relevance_reason": ""}')
    assert v.relevance == 5
    assert v.faithfulness is None
    assert parse_judge_output('{"claims": [], "relevance": 0}').relevance == 1


@pytest.mark.parametrize(
    "text",
    ["no json here", '{"claims": "nope", "relevance": 3}', '{"relevance": "high"}', "{broken"],
)
def test_parse_errors(text: str) -> None:
    with pytest.raises(JudgeParseError):
        parse_judge_output(text)


def test_extract_json_with_braces_in_strings() -> None:
    assert extract_json_object('prefix {"a": "x } y", "b": {"c": 1}} suffix') == {
        "a": "x } y",
        "b": {"c": 1},
    }


def test_judge_answer_end_to_end_with_mock() -> None:
    llm = ScriptedLLM([CANNED_PLAIN], model="gemini-3.8-flash", backend="openai_compatible")
    res = judge_answer(
        llm, "Q?", "KYC every two years [1]. Fee is Rs 500 [2].", ["src one", "src two"]
    )
    assert res.verdict is not None
    assert res.verdict.faithfulness == 0.5
    assert res.model == "gemini-3.8-flash"
    system, msgs = llm.calls[0]
    assert system is not None
    assert "[1] src one" in msgs[0].content
    assert "[2]" not in msgs[0].content.split("ANSWER:")[1]  # citations stripped from the answer
    bad = judge_answer(ScriptedLLM(["sorry"]), "Q?", "A.", ["s"])
    assert bad.verdict is None
    assert bad.error is not None


def test_judge_declined() -> None:
    class Declines(ScriptedLLM):
        def generate(self, messages, **kw):  # type: ignore[no-untyped-def,override]
            return LLMResponse(text="", model="m", backend="b", refused=True)

    assert judge_answer(Declines(replies=[]), "Q", "A", ["s"]).error == "judge declined"


def test_build_judge_prompt_without_sources() -> None:
    assert "(none)" in build_judge_prompt("Q", "A", [])


def test_nli_check_hand_computed() -> None:
    sources = [
        "periodic updation of kyc every two years for high risk customers",
        "late submission fee applies to delayed reports",
    ]
    answer = (
        "Periodic updation of KYC is every two years for high risk customers [1]. "
        "A late submission fee applies to delayed reports [1]. "
        "Penalty amount is ten thousand rupees for banks."
    )
    report = parse_citations(answer, 2)
    res = nli_check(FakeNLI(), report, sources, threshold=0.99)
    # sentence 1 entailed by [1]; sentence 2 cites [1] but is only entailed by [2];
    # sentence 3 is uncited -> checked against all sources -> not entailed.
    assert res.n_sentences == 3
    assert res.faithfulness == pytest.approx(1 / 3)
    assert res.n_citation_pairs == 2
    assert res.citation_accuracy == 0.5


def test_nli_skips_refusals() -> None:
    from rbirag.generation.prompts import REFUSAL_TEXT

    res = nli_check(FakeNLI(), parse_citations(REFUSAL_TEXT, 1), ["s"])
    assert res.faithfulness is None
