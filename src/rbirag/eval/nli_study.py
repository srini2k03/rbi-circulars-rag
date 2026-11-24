"""Why does the LLM judge rate answers ~0.95 faithful while NLI says ~0.21?

For the committed generation records of one BM25 configuration (whose sources can be
re-retrieved exactly: BM25 is deterministic), this re-scores every answer with several
NLI protocols and measures what separates them:

* ``whole_chunk`` - the original protocol (each answer sentence vs each cited source, the
  full passage as premise). Reproducing the committed scores validates the setup.
* ``windows_cited`` - the same sentence vs short windows of the cited sources.
* ``windows_any`` - vs windows of any retrieved source (ignores citation mistakes).
* ``claims_windows`` - sentences split into clauses, each clause vs windows of any source.

plus diagnostics: how often the premise exceeds the model's 512-piece input (truncation),
whether the supporting window lies beyond that cut-off, answers that only say "the
documents do not specify X" (the judge counts them as faithful, NLI cannot entail them),
and the per-answer agreement of each protocol with the judge.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from rbirag.eval.gen_metrics import GenerationRecord
from rbirag.eval.nli import NLIScorer, nli_check
from rbirag.eval.stats import bootstrap_ci
from rbirag.generation.citations import parse_citations, strip_citations
from rbirag.generation.evidence import is_non_answer, premise_windows, split_claims

MODEL_MAX_PIECES = 512
VARIANTS = ("whole_chunk", "windows_cited", "windows_any", "claims_windows")


class AnswerStudy(BaseModel):
    qa_id: str
    config: str
    backend: str
    judge: float | None
    committed_nli: float | None
    nli: dict[str, float | None] = Field(default_factory=dict)  # variant -> faithfulness
    nli_low_threshold: dict[str, float | None] = Field(default_factory=dict)  # tau = 0.1
    non_answer: bool = False
    n_sentences: int = 0
    n_uncited: int = 0
    n_claims: int = 0
    premises: int = 0
    premises_truncated: int = 0
    rescued_by_windows: int = 0  # sentences unsupported whole-chunk, supported by a window
    rescued_beyond_cutoff: int = 0  # ... whose best window starts after the 512-piece cut
    sentences: list[dict[str, Any]] = Field(default_factory=list)


def study_answer(
    rec: GenerationRecord,
    sources: Sequence[str],
    scorer: NLIScorer,
    count_pieces: Callable[[str], int],
    *,
    threshold: float = 0.5,
) -> AnswerStudy:
    report = parse_citations(rec.answer, len(sources))
    out = AnswerStudy(
        qa_id=rec.qa_id, config=rec.config, backend=rec.backend,
        judge=rec.judge_faithfulness, committed_nli=rec.nli_faithfulness,
        non_answer=is_non_answer(rec.answer),
    )  # fmt: skip
    whole = nli_check(scorer, report, list(sources), threshold=threshold)
    out.nli["whole_chunk"] = whole.faithfulness
    sents = [(strip_citations(s.sentence), s.citations) for s in report.sentences]
    sents = [(h, c) for h, c in sents if len(h.split()) >= 3]
    if not sents or report.is_refusal:
        return out
    windows = [premise_windows(src) for src in sources]
    pieces = [count_pieces(src) for src in sources]
    out.n_sentences = len(sents)
    out.n_uncited = sum(1 for _, c in sents if not [i for i in c if 1 <= i <= len(sources)])
    # one batched call: (premise, hypothesis) for every unit the protocols need
    pairs: list[tuple[str, str]] = []
    index: dict[tuple[str, int, int], int] = {}  # (kind, sentence/claim id, window id)
    flat_windows = [(si, w) for si, ws in enumerate(windows) for w in ws]
    claims: list[tuple[int, str]] = []
    for i, (hyp, cited) in enumerate(sents):
        for wi, (_, w) in enumerate(flat_windows):
            index[("s", i, wi)] = len(pairs)
            pairs.append((w.text, hyp))
        cited_idx = [c - 1 for c in cited if 1 <= c <= len(sources)]
        for t in cited_idx or range(len(sources)):  # the original whole-passage premises
            index[("w", i, t)] = len(pairs)
            pairs.append((sources[t], hyp))
        claims.extend((i, claim) for claim in split_claims(hyp))
    claim_alias: dict[int, int] = {}  # claim id -> sentence id when the claim is the sentence
    for ci, (i, claim) in enumerate(claims):
        if claim == sents[i][0]:
            claim_alias[ci] = i
            continue
        for wi, (_, w) in enumerate(flat_windows):
            index[("c", ci, wi)] = len(pairs)
            pairs.append((w.text, claim))
    probs = scorer.entailment(pairs) if pairs else []
    out.n_claims = len(claims)

    def best(kind: str, unit: int, allowed: set[int] | None) -> tuple[float, int | None]:
        if kind == "c" and unit in claim_alias:
            kind, unit = "s", claim_alias[unit]
        top, arg = 0.0, None
        for wi, (src, _) in enumerate(flat_windows):
            if allowed is not None and src not in allowed:
                continue
            p = probs[index[(kind, unit, wi)]]
            if p > top:
                top, arg = p, wi
        return top, arg

    cited_ok = low_cited = any_ok = low_any = 0
    for i, (hyp, cited) in enumerate(sents):
        valid = {c - 1 for c in cited if 1 <= c <= len(sources)}
        p_cited, _ = best("s", i, valid or None)
        p_any, arg = best("s", i, None)
        cited_ok += p_cited >= threshold
        any_ok += p_any >= threshold
        low_cited += p_cited >= 0.1
        low_any += p_any >= 0.1
        targets = sorted(valid) if valid else list(range(len(sources)))
        whole_p = max((probs[index[("w", i, t)]] for t in targets), default=0.0)
        detail: dict[str, Any] = {"sentence": hyp[:300], "whole": round(whole_p, 4),
                                  "best_window": round(p_any, 4)}  # fmt: skip
        if whole_p < threshold <= p_any and arg is not None:
            out.rescued_by_windows += 1
            src, w = flat_windows[arg]
            offset = count_pieces(sources[src][: w.start])
            budget = MODEL_MAX_PIECES - count_pieces(hyp) - 3
            detail["window_offset_pieces"] = offset
            if offset + count_pieces(w.text) > budget:
                out.rescued_beyond_cutoff += 1
                detail["beyond_cutoff"] = True
        out.sentences.append(detail)
    out.premises = len(sources)
    out.premises_truncated = sum(1 for n in pieces if n > MODEL_MAX_PIECES - 40)
    out.nli["windows_cited"] = cited_ok / len(sents)
    out.nli["windows_any"] = any_ok / len(sents)
    out.nli_low_threshold["windows_cited"] = low_cited / len(sents)
    out.nli_low_threshold["windows_any"] = low_any / len(sents)
    claim_ok = sum(best("c", ci, None)[0] >= threshold for ci in range(len(claims)))
    out.nli["claims_windows"] = claim_ok / len(claims) if claims else None
    return out


def _corr(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize_study(rows: Sequence[AnswerStudy], *, seed: int = 13) -> dict[str, Any]:
    judged = [r for r in rows if r.judge is not None and r.nli.get("whole_chunk") is not None]
    out: dict[str, Any] = {"answers": len(judged)}
    judge = [float(r.judge or 0.0) for r in judged]
    out["judge_mean"] = bootstrap_ci(judge, seed=seed).model_dump()
    reproduced = [
        r for r in judged if r.committed_nli is not None
        and abs((r.nli["whole_chunk"] or 0.0) - r.committed_nli) < 1e-6
    ]  # fmt: skip
    out["whole_chunk_reproduces_committed"] = f"{len(reproduced)}/{len(judged)}"
    for v in VARIANTS:
        vals = [r.nli.get(v) for r in judged]
        pairs = [(j, x) for j, x in zip(judge, vals, strict=True) if x is not None]
        out[v] = {
            "mean": bootstrap_ci([x for _, x in pairs], seed=seed).model_dump(),
            "pearson_r_vs_judge": _corr([j for j, _ in pairs], [x for _, x in pairs]),
            "mean_abs_diff_vs_judge": float(np.mean([abs(j - x) for j, x in pairs]))
            if pairs else None,
            "fully_supported_share": float(np.mean([x >= 1.0 for _, x in pairs]))
            if pairs else None,
        }  # fmt: skip
    for v in ("windows_cited", "windows_any"):
        vals = [
            r.nli_low_threshold.get(v) for r in judged if r.nli_low_threshold.get(v) is not None
        ]
        out[f"{v}_tau0.1_mean"] = float(np.mean(vals)) if vals else None  # type: ignore[arg-type]
    non = [r for r in judged if r.non_answer]
    out["non_answers"] = {
        "count": len(non),
        "judge_mean": float(np.mean([r.judge or 0.0 for r in non])) if non else None,
        "whole_chunk_mean": float(np.mean([r.nli["whole_chunk"] or 0.0 for r in non]))
        if non else None,
    }  # fmt: skip
    factual = [r for r in judged if not r.non_answer]
    out["excluding_non_answers"] = {
        v: float(np.mean([r.nli.get(v) or 0.0 for r in factual])) if factual else None
        for v in VARIANTS
    }
    out["excluding_non_answers"]["judge"] = (
        float(np.mean([r.judge or 0.0 for r in factual])) if factual else None
    )
    premises = sum(r.premises for r in judged)
    out["premises"] = premises
    out["premises_truncated_share"] = (
        sum(r.premises_truncated for r in judged) / premises if premises else None
    )
    sentences = sum(r.n_sentences for r in judged)
    out["sentences"] = sentences
    out["uncited_sentence_share"] = (
        sum(r.n_uncited for r in judged) / sentences if sentences else None
    )
    rescued = sum(r.rescued_by_windows for r in judged)
    out["sentences_rescued_by_windows"] = rescued
    out["rescued_beyond_512_cutoff"] = sum(r.rescued_beyond_cutoff for r in judged)
    out["by_backend"] = {
        b: {
            "answers": len(sub),
            "judge": float(np.mean([r.judge or 0.0 for r in sub])),
            **{v: float(np.mean([r.nli.get(v) or 0.0 for r in sub])) for v in VARIANTS},
        }
        for b in sorted({r.backend for r in judged})
        if (sub := [r for r in judged if r.backend == b])
    }
    return out
