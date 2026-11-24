"""``rbirag eval nli-study``: the LLM-judge vs NLI faithfulness investigation."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from rbirag.config import get_settings
from rbirag.eval.cli import _csv, eval_app


def committed_answers(s: Any, documents: Sequence[Any], configs: Sequence[str]) -> Iterator[Any]:
    """``(record, sources)`` for every judged, answered, answerable committed answer of the
    BM25 ``configs``; the sources are re-retrieved (BM25 is deterministic, so they are the
    passages the generator was shown)."""
    from rbirag.chunking import build_chunker
    from rbirag.eval.ablate import RetrievalConfig, settings_for
    from rbirag.eval.records import load_records
    from rbirag.generation.rag import to_sources
    from rbirag.index.build import build_index
    from rbirag.retrieval.params import RetrievalParams
    from rbirag.retrieval.pipeline import Retriever

    for name in configs:
        cfg = RetrievalConfig.parse(name)
        if cfg.mode != "bm25":
            raise typer.BadParameter(f"{name}: only BM25 configs can be re-retrieved offline")
        chunker = build_chunker(cfg.chunker, settings_for(s, cfg.size))
        bundle = build_index([c for d in documents for c in chunker.chunk(d)], embedder=None)
        retriever = Retriever(bundle.store, bm25=bundle.bm25, params=RetrievalParams(mode="bm25"))
        for path in sorted((s.results_dir / "generation").glob(f"records-*-{name}.jsonl")):
            for rec in load_records(path).values():
                if rec.refused or not rec.answerable or rec.error or rec.judge_faithfulness is None:
                    continue
                result = retriever.retrieve(rec.question, top_k=s.answer_top_k)
                yield rec, to_sources(result.results)


@eval_app.command("verify-citations")
def verify_citations(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    configs: Annotated[str, typer.Option(help="BM25 generation configs.")] = "fixed-1024-bm25",
    out_dir: Annotated[Path | None, typer.Option(help="Default: results/analysis")] = None,
    scorer: Annotated[str, typer.Option(help="lexical | nli")] = "lexical",
    threshold: Annotated[float | None, typer.Option(help="Default: the scorer's.")] = None,
    log: Annotated[bool, typer.Option(help="Append to results/experiments.jsonl.")] = True,
) -> None:
    """Run the citation verifier on the committed answers: verdicts, invalid / irrelevant
    citations, unsupported claims, possibly fabricated circular numbers, and how the
    verdict relates to the LLM judge (a comparison, not a ground truth)."""
    import numpy as np

    from rbirag.document import read_documents_jsonl
    from rbirag.eval.history import log_experiment
    from rbirag.generation.rag import build_verifier
    from rbirag.generation.verify import SourceRef, verify_answer

    s = get_settings()
    out = out_dir or s.results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    documents = read_documents_jsonl(docs or s.documents_path)
    verifier = build_verifier(scorer, s)
    assert verifier is not None
    tau = threshold if threshold is not None else (0.5 if scorer == "nli" else s.verify_threshold)
    per: list[dict[str, Any]] = []
    for rec, sources in committed_answers(s, documents, _csv(configs)):
        refs = [SourceRef(n=x.n, doc_id=x.doc_id, text=x.text, circular_number=x.circular_number,
                          ref_number=x.ref_number) for x in sources]  # fmt: skip
        rep = verify_answer(rec.answer, refs, verifier, threshold=tau)
        per.append({
            "qa_id": rec.qa_id, "config": rec.config, "backend": rec.backend,
            "judge": rec.judge_faithfulness, "verdict": rep.verdict,
            "supported_share": rep.supported_share, "claims": len(rep.claims),
            "unsupported": len(rep.unsupported_claims), "invalid": len(rep.invalid_citations),
            "irrelevant": sum(1 for c in rep.citations if c.relevant is False),
            "citations": sum(1 for c in rep.citations if c.exists),
            "unknown_refs": rep.unknown_references, "non_answer": rep.non_answer,
            "uncited_sentences": rep.uncited_sentences,
        })  # fmt: skip
    summary: dict[str, Any] = {"scorer": verifier.name, "threshold": tau, "answers": len(per)}
    for backend in [*sorted({p["backend"] for p in per}), "all"]:
        sub = [p for p in per if backend in ("all", p["backend"])]
        claims = sum(p["claims"] for p in sub)
        cits = sum(p["citations"] for p in sub)
        shares = [p["supported_share"] for p in sub if p["supported_share"] is not None]
        judge_full = [p for p in sub if (p["judge"] or 0) >= 0.999]
        n_unsupported = sum(p["unsupported"] for p in sub)
        n_irrelevant = sum(p["irrelevant"] for p in sub)
        summary[backend] = {
            "answers": len(sub),
            "verdicts": {v: sum(1 for p in sub if p["verdict"] == v)
                         for v in ("supported", "partially_supported", "unsupported", "no_claims")},
            "mean_supported_share": float(np.mean(shares)) if shares else None,
            "unsupported_claim_rate": n_unsupported / claims if claims else None,
            "answers_with_invalid_citation": sum(1 for p in sub if p["invalid"]),
            "irrelevant_citation_rate": n_irrelevant / cits if cits else None,
            "answers_with_unknown_circular_numbers": sum(1 for p in sub if p["unknown_refs"]),
            "non_answers": sum(1 for p in sub if p["non_answer"]),
            "judge_fully_faithful": len(judge_full),
            "of_which_verifier_supported": sum(
                1 for p in judge_full if p["verdict"] == "supported"
            ),
        }  # fmt: skip
    name = f"citation_verification-{scorer}"
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with gzip.open(out / f"{name}_answers.jsonl.gz", "wt", encoding="utf-8") as fh:
        for p in per:
            fh.write(json.dumps(p) + "\n")
    if log:
        log_experiment(
            s.results_dir,
            {"experiment": "citation-verification", "command": "rbirag eval verify-citations",
             "scorer": verifier.name, "threshold": tau, "answers": len(per),
             "verdicts": summary["all"]["verdicts"]},
        )  # fmt: skip
    typer.echo(json.dumps(summary["all"], indent=1))


@eval_app.command("nli-study")
def nli_study(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    configs: Annotated[
        str, typer.Option(help="BM25 generation configs to study (sources re-retrieved).")
    ] = "fixed-1024-bm25",
    out_dir: Annotated[Path | None, typer.Option(help="Default: results/analysis")] = None,
    nli: Annotated[str, typer.Option(help="cross-encoder | fake")] = "cross-encoder",
    limit: Annotated[int, typer.Option(help="Study at most this many answers (0 = all).")] = 0,
    log: Annotated[bool, typer.Option(help="Append to results/experiments.jsonl.")] = True,
) -> None:
    """Re-score committed answers with several NLI protocols and compare with the judge."""
    from rbirag.document import read_documents_jsonl
    from rbirag.eval.history import log_experiment
    from rbirag.eval.nli import CrossEncoderNLI, FakeNLI
    from rbirag.eval.nli_study import VARIANTS, study_answer, summarize_study

    s = get_settings()
    out = out_dir or s.results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    documents = read_documents_jsonl(docs or s.documents_path)
    scorer: CrossEncoderNLI | FakeNLI = (
        FakeNLI() if nli == "fake" else CrossEncoderNLI(s.nli_model, device=s.device)
    )
    rows = []
    for rec, sources in committed_answers(s, documents, _csv(configs)):
        rows.append(study_answer(rec, [x.text for x in sources], scorer, scorer.count_pieces))
        if limit and len(rows) >= limit:
            break
        if len(rows) % 25 == 0:
            typer.echo(f"studied {len(rows)} answers")
    summary = summarize_study(rows, seed=s.seed)
    summary["nli_model"] = scorer.name
    summary["configs"] = _csv(configs)
    (out / "nli_study.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with gzip.open(out / "nli_study_answers.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(r.model_dump_json() + "\n")
    if log:
        means = {v: summary[v]["mean"]["mean"] for v in VARIANTS}
        log_experiment(
            s.results_dir,
            {
                "experiment": "nli-vs-judge-study",
                "command": "rbirag eval nli-study",
                "nli_model": scorer.name,
                "answers": summary["answers"],
                "means": means,
            },
        )
    typer.echo(json.dumps({k: summary[k] for k in ("answers", "whole_chunk_reproduces_committed")}))


@eval_app.command("query-understanding")
def query_understanding(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    inputs: Annotated[Path, typer.Option(help="AI-authored behavioural inputs.")] = Path(
        "data/eval/query_understanding_ai.jsonl"
    ),
    out_dir: Annotated[Path | None, typer.Option(help="Default: results/analysis")] = None,
    log: Annotated[bool, typer.Option(help="Append to results/experiments.jsonl.")] = True,
) -> None:
    """How often query understanding flags real test questions (false alarms), the
    unanswerable ones (scope), and the AI-authored ambiguous / multi-part / out-of-scope
    inputs; and whether follow-up rewriting helps retrieval (BM25, the primary index)."""
    import numpy as np

    from rbirag.chunking import build_chunker
    from rbirag.document import read_documents_jsonl
    from rbirag.eval.ablate import settings_for
    from rbirag.eval.cli import dev_test, load_qa
    from rbirag.eval.history import log_experiment
    from rbirag.eval.retrieval_metrics import query_metrics
    from rbirag.index.build import build_index
    from rbirag.query.understanding import analyse_query
    from rbirag.retrieval.params import RetrievalParams
    from rbirag.retrieval.pipeline import Retriever

    s = get_settings()
    out = out_dir or s.results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    documents = read_documents_jsonl(docs or s.documents_path)
    chunker = build_chunker("fixed", settings_for(s, 1024))
    bundle = build_index([c for d in documents for c in chunker.chunk(d)], embedder=None)
    retriever = Retriever(bundle.store, bm25=bundle.bm25, params=RetrievalParams(mode="bm25"))

    def known(tok: str) -> bool:
        return bundle.bm25.idf(tok) != 0.0

    pairs = load_qa(s)
    _, test = dev_test(s, pairs)
    by_id = {p.qa_id: p for p in pairs}
    rows = [json.loads(x) for x in inputs.read_text("utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["set"] != "_about"]
    report: dict[str, Any] = {"inputs": str(inputs), "label": "AI-authored behavioural inputs"}

    def rate(items: list[Any], pred: Callable[[Any], bool]) -> dict[str, Any]:
        hits = sum(1 for x in items if pred(x))
        return {"flagged": hits, "n": len(items), "rate": hits / len(items) if items else None}

    analysed = {p.qa_id: analyse_query(p.question, vocabulary=known) for p in test}
    answerable = [analysed[p.qa_id] for p in test if p.answerable]
    unanswerable = [analysed[p.qa_id] for p in test if not p.answerable]
    report["test_answerable"] = {
        "out_of_scope (false abstain)": rate(answerable, lambda a: a.out_of_scope),
        "ambiguous": rate(answerable, lambda a: a.ambiguous),
        "multi_part": rate(answerable, lambda a: len(a.parts) > 1),
        "types": dict(Counter(a.qtype for a in answerable)),
    }
    report["test_unanswerable"] = {
        "out_of_scope (correct abstain)": rate(unanswerable, lambda a: a.out_of_scope),
        "reasons": dict(Counter(a.scope_reason for a in unanswerable if a.scope_reason)),
    }
    for name, pred in (
        ("out_of_scope", lambda a: a.out_of_scope),
        ("ambiguous", lambda a: a.ambiguous),
        ("multi_part", lambda a: len(a.parts) > 1),
    ):
        items = [analyse_query(r["question"], vocabulary=known) for r in rows if r["set"] == name]
        report[f"ai_{name}"] = rate(items, pred)
    # follow-ups: retrieval with the elliptical turn alone, rewritten, and the full question
    variants: dict[str, list[dict[str, float]]] = {"alone": [], "rewritten": [], "full": []}
    flagged = 0
    for r in (r for r in rows if r["set"] == "follow_up"):
        pair = by_id[r["qa_id"]]
        a = analyse_query(r["question"], history=r["history"], vocabulary=known)
        flagged += a.follow_up
        for name, q in (("alone", r["question"]), ("rewritten", a.retrieval_query),
                        ("full", pair.question)):  # fmt: skip
            chunks = [x.chunk for x in retriever.retrieve(q, top_k=10).results]
            doc = query_metrics(chunks, pair, level="doc")
            chunk = query_metrics(chunks, pair, level="chunk")
            variants[name].append({"doc_recall@5": doc.recall[5], "chunk_ndcg@10": chunk.ndcg})
    report["follow_up"] = {
        "n": len(variants["alone"]),
        "detected_as_follow_up": flagged,
        **{
            name: {m: float(np.mean([v[m] for v in vals])) for m in ("doc_recall@5",
                                                                     "chunk_ndcg@10")}
            for name, vals in variants.items() if vals
        },
        "note": "BM25 on the primary index; 'full' is the original test question (upper bound)",
    }  # fmt: skip
    (out / "query_understanding.json").write_text(json.dumps(report, indent=1), "utf-8")
    if log:
        log_experiment(s.results_dir, {"experiment": "query-understanding",
                                       "command": "rbirag eval query-understanding",
                                       "follow_up": report["follow_up"]})  # fmt: skip
    typer.echo(json.dumps(report, indent=1))


@eval_app.command("review-answers")
def review_answers(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    n: Annotated[int, typer.Option(help="Answers in the sample.")] = 40,
    reviewer: Annotated[str, typer.Option(help="Your name or handle (stored).")] = "human",
    seed: Annotated[int, typer.Option()] = 13,
) -> None:
    """Label a stratified sample of generated answers (faithful / partially / unfaithful,
    correct or not) against the passages they were generated from. Resumable; writes
    data/eval/answer_reviews.jsonl."""
    from rbirag.document import read_documents_jsonl
    from rbirag.eval.human_review import (
        REVIEW_FILE,
        load_check_rows,
        review_loop,
        sample_for_review,
    )

    s = get_settings()
    checks = load_check_rows(s.results_dir)
    if not checks:
        raise typer.BadParameter("run `rbirag eval verify-citations` first (it lists the answers)")
    sample = sample_for_review(list(checks.values()), n, seed=seed)
    wanted = {(x["qa_id"], x["config"], x["backend"]) for x in sample}
    documents = read_documents_jsonl(docs or s.documents_path)
    shown: dict[tuple[str, str, str], tuple[Any, Any]] = {}
    for rec, sources in committed_answers(s, documents, sorted({x["config"] for x in sample})):
        key = (rec.qa_id, rec.config, rec.backend)
        if key in wanted:
            shown[key] = (rec, sources)

    def show(item: dict[str, Any]) -> str:
        rec, sources = shown[(item["qa_id"], item["config"], item["backend"])]
        parts = [f"\n=== {rec.qa_id} ({rec.backend})", f"Q: {rec.question}", ""]
        parts += [f"[{x.n}] {x.title} ({x.date})\n{x.text[:1500]}\n" for x in sources]
        parts += [f"ANSWER: {rec.answer}"]
        return "\n".join(parts)

    items = [x for x in sample if (x["qa_id"], x["config"], x["backend"]) in shown]
    added = review_loop(items, show, s.eval_dir / REVIEW_FILE, reviewer=reviewer, echo=typer.echo)
    typer.echo(f"saved {added} reviews to {s.eval_dir / REVIEW_FILE}")


@eval_app.command("review-agreement")
def review_agreement(
    out: Annotated[
        Path | None, typer.Option(help="Default: results/analysis/review_agreement.json")
    ] = None,
) -> None:
    """Cohen's kappa of the LLM judge and the citation verifiers against human labels."""
    from rbirag.eval.human_review import REVIEW_FILE, agreement, load_check_rows, read_reviews

    s = get_settings()
    report = agreement(read_reviews(s.eval_dir / REVIEW_FILE), load_check_rows(s.results_dir))
    path = out or s.results_dir / "analysis" / "review_agreement.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    typer.echo(json.dumps(report, indent=1))


@eval_app.command("multi-doc")
def multi_doc(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    graph_file: Annotated[Path | None, typer.Option("--graph", help="graph.json.gz")] = None,
    n: Annotated[int, typer.Option(help="Questions to generate.")] = 40,
    seed: Annotated[int, typer.Option()] = 13,
    out_dir: Annotated[Path | None, typer.Option(help="Default: results/analysis")] = None,
    log: Annotated[bool, typer.Option(help="Append to results/experiments.jsonl.")] = True,
) -> None:
    """Multi-document questions generated from high-confidence "amends" / "supersedes"
    relations ("Which notification amended <title>, and what did it change?"; gold = both
    documents) and how often BM25 puts both documents in the context. Template-generated,
    not human-written: a stress test of multi-document retrieval, not a benchmark."""
    import random

    from rbirag.chunking import build_chunker
    from rbirag.document import read_documents_jsonl
    from rbirag.eval.ablate import settings_for
    from rbirag.eval.history import log_experiment
    from rbirag.index.build import build_index
    from rbirag.intel.cli import _graph_path, load_graph
    from rbirag.retrieval.params import RetrievalParams
    from rbirag.retrieval.pipeline import Retriever

    s = get_settings()
    out = out_dir or s.results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    graph = load_graph(graph_file or _graph_path(None))
    edges = sorted(
        (e for e in graph.edges if e.type in {"amends", "supersedes"} and e.confidence == "high"),
        key=lambda e: (e.actor, e.affected),
    )
    rng = random.Random(seed)
    picked = rng.sample(edges, min(n, len(edges)))
    verb = {"amends": "amended", "supersedes": "superseded"}
    questions = [
        {
            "question": f"Which RBI notification {verb[e.type]} the "
            f"'{graph.intel[e.affected].subject}', and what did it change?",
            "gold": [e.actor, e.affected],
            "type": e.type,
        }
        for e in picked
    ]
    documents = read_documents_jsonl(docs or s.documents_path)
    chunker = build_chunker("fixed", settings_for(s, 1024))
    bundle = build_index([c for d in documents for c in chunker.chunk(d)], embedder=None)
    retriever = Retriever(bundle.store, bm25=bundle.bm25, params=RetrievalParams(mode="bm25"))
    from rbirag.retrieval.params import RetrievalFilter

    def graph_expanded(question: str, top: list[str], extra: int = 3) -> list[str]:
        """The top passages' documents plus documents the graph says change them (or
        that they change): the best passage of each linked document is added."""
        linked: list[str] = []
        for d in dict.fromkeys(top):
            linked += [e.actor for e in graph.changed_by(d)] + [
                e.affected for e in graph.changes(d)
            ]
        linked = [d for d in dict.fromkeys(linked) if d not in top][:extra]
        if not linked:
            return top
        res = retriever.retrieve(question, top_k=len(linked) * 3,
                                 filters=RetrievalFilter(doc_ids=tuple(linked)))  # fmt: skip
        return top + list(dict.fromkeys(r.chunk.doc_id for r in res.results))[:extra]

    rows = []
    for q in questions:
        docs5 = [r.chunk.doc_id for r in retriever.retrieve(q["question"], top_k=5).results]
        docs10 = [r.chunk.doc_id for r in retriever.retrieve(q["question"], top_k=10).results]
        expanded = graph_expanded(q["question"], docs5)
        rows.append({
            **q,
            "both_in_top5": all(g in docs5 for g in q["gold"]),
            "both_in_top10": all(g in docs10 for g in q["gold"]),
            "changed_doc_in_top5": q["gold"][1] in docs5,
            "changing_doc_in_top5": q["gold"][0] in docs5,
            "both_in_top5_plus_graph": all(g in expanded for g in q["gold"]),
            "context_passages_plus_graph": len(expanded),
        })  # fmt: skip
    summary = {
        "label": "template-generated from the relation graph; not human-written or verified",
        "questions": len(rows),
        **{
            k: sum(r[k] for r in rows) / len(rows) if rows else None
            for k in ("both_in_top5", "both_in_top10", "changed_doc_in_top5",
                      "changing_doc_in_top5", "both_in_top5_plus_graph",
                      "context_passages_plus_graph")
        },
        "retriever": "BM25, fixed-1024",
    }  # fmt: skip
    (out / "multi_doc.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    if log:
        log_experiment(s.results_dir, {"experiment": "multi-document-retrieval",
                                       "command": "rbirag eval multi-doc", **summary})  # fmt: skip
    typer.echo(json.dumps(summary, indent=1))


@eval_app.command("answer-correctness")
def answer_correctness(
    configs: Annotated[str, typer.Option(help="Generation configs.")] = "fixed-1024-bm25",
    out_dir: Annotated[Path | None, typer.Option(help="Default: results/analysis")] = None,
    log: Annotated[bool, typer.Option(help="Append to results/experiments.jsonl.")] = True,
) -> None:
    """Answer correctness against the reference answers, without an LLM: token F1 (SQuAD
    style, on BM25-normalised tokens) and figure recall (share of the reference's numbers
    that the answer states). Answered, answerable test questions only."""
    import numpy as np

    from rbirag.eval.cli import dev_test, load_qa
    from rbirag.eval.history import log_experiment
    from rbirag.eval.records import load_records
    from rbirag.eval.stats import bootstrap_ci
    from rbirag.generation.citations import strip_citations
    from rbirag.generation.verify import numbers_in
    from rbirag.index.bm25 import bm25_tokenize

    s = get_settings()
    out = out_dir or s.results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    _, test = dev_test(s, load_qa(s))
    refs = {p.qa_id: p.answer for p in test if p.answerable and p.answer}

    def f1(answer: str, reference: str) -> float:
        a, r = bm25_tokenize(answer), bm25_tokenize(reference)
        common = sum((Counter(a) & Counter(r)).values())
        if not a or not r or not common:
            return 0.0
        p, rc = common / len(a), common / len(r)
        return 2 * p * rc / (p + rc)

    report: dict[str, Any] = {"metric": "token F1 and figure recall vs the reference answer",
                              "split": "test"}  # fmt: skip
    for name in _csv(configs):
        for path in sorted((s.results_dir / "generation").glob(f"records-*-{name}.jsonl")):
            recs = [r for r in load_records(path).values()
                    if r.qa_id in refs and not r.refused and not r.error]  # fmt: skip
            if not recs:
                continue
            f1s = [f1(strip_citations(r.answer), refs[r.qa_id]) for r in recs]
            fig = []
            for r in recs:
                wanted = numbers_in(refs[r.qa_id])
                if wanted:
                    fig.append(len(wanted & numbers_in(r.answer)) / len(wanted))
            key = f"{name}/{recs[0].backend}"
            report[key] = {
                "answers": len(recs),
                "token_f1": bootstrap_ci(f1s, seed=s.seed).model_dump(),
                "figure_recall": bootstrap_ci(fig, seed=s.seed).model_dump() if fig else None,
                "with_figures": len(fig),
                "judge_relevance_mean": float(np.mean([r.judge_relevance for r in recs
                                                       if r.judge_relevance is not None])),
            }  # fmt: skip
    (out / "answer_correctness.json").write_text(json.dumps(report, indent=1), "utf-8")
    if log:
        log_experiment(s.results_dir, {"experiment": "answer-correctness",
                                       "command": "rbirag eval answer-correctness",
                                       "configs": _csv(configs)})  # fmt: skip
    typer.echo(json.dumps(report, indent=1))
