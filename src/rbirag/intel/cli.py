"""``rbirag intel ...``: build the document-intelligence layer and query it."""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import typer

from rbirag.config import get_settings

intel_app = typer.Typer(
    help="Regulatory metadata, cross-references and the relation graph.", no_args_is_help=True
)

GRAPH_FILE = "graph.json.gz"


def save_graph(intel: dict[str, Any], relations: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "intel": [i.model_dump(mode="json") for i in intel.values()],
        "relations": [r.model_dump(mode="json") for r in relations],
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def load_graph(path: Path) -> Any:
    """A :class:`RelationGraph` from a file written by ``rbirag intel build``."""
    from rbirag.intel.extract import DocIntel, Relation
    from rbirag.intel.graph import RelationGraph

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    intel = {i["doc_id"]: DocIntel.model_validate(i) for i in payload["intel"]}
    relations = [Relation.model_validate(r) for r in payload["relations"]]
    return RelationGraph(intel, relations)


def _graph_path(out: Path | None) -> Path:
    s = get_settings()
    return (out or s.processed_dir) / GRAPH_FILE


@intel_app.command("build")
def build(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    out: Annotated[Path | None, typer.Option(help="Directory (default: data/processed).")] = None,
    export: Annotated[
        Path | None, typer.Option(help="Also write the compact JSON for the static site.")
    ] = None,
    summary: Annotated[Path | None, typer.Option(help="Write statistics JSON here.")] = None,
) -> None:
    """Extract metadata and references from every document and build the graph."""
    from rbirag.document import read_documents_jsonl
    from rbirag.intel.extract import build_intel
    from rbirag.intel.graph import RelationGraph

    s = get_settings()
    documents = read_documents_jsonl(docs or s.documents_path)
    intel, relations = build_intel(documents)
    path = _graph_path(out)
    save_graph(intel, relations, path)
    graph = RelationGraph(intel, relations)
    stats = graph.stats()
    if export is not None:
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(json.dumps(graph.export(), ensure_ascii=False), encoding="utf-8")
    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    typer.echo(json.dumps({"graph": str(path), **stats}, indent=1))


@intel_app.command("show")
def show(
    doc_id: str,
    graph_file: Annotated[Path | None, typer.Option("--graph", help="graph.json.gz")] = None,
) -> None:
    """Metadata, versions, amendments, the latest document in the corpus, related docs."""
    graph = load_graph(graph_file or _graph_path(None))
    if doc_id not in graph.intel:
        raise typer.BadParameter(f"unknown document {doc_id!r}")
    payload = {
        "metadata": graph.intel[doc_id].model_dump(mode="json"),
        "versions": graph.versions(doc_id),
        "original": graph.original(doc_id),
        "changed_by": [e.model_dump() for e in graph.changed_by(doc_id)],
        "changes": [e.model_dump() for e in graph.changes(doc_id)],
        "latest_in_corpus": graph.latest_in_corpus(doc_id).model_dump(),
        "related": graph.related(doc_id),
        "unresolved_references": len(graph.unresolved(doc_id)),
    }
    typer.echo(json.dumps(payload, indent=1, ensure_ascii=False))


@intel_app.command("diff")
def diff(
    a: str,
    b: str,
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
) -> None:
    """What changed between two documents (sentence-level text comparison)."""
    from rbirag.document import read_documents_jsonl
    from rbirag.intel.graph import compare_documents

    s = get_settings()
    by_id = {d.doc_id: d for d in read_documents_jsonl(docs or s.documents_path)}
    for d in (a, b):
        if d not in by_id:
            raise typer.BadParameter(f"unknown document {d!r}")
    typer.echo(compare_documents(by_id[a], by_id[b]).model_dump_json(indent=1))


@intel_app.command("sample")
def sample(
    out: Annotated[Path, typer.Option(help="JSONL to review.")] = Path(
        "results/intel/review_sample.jsonl"
    ),
    graph_file: Annotated[Path | None, typer.Option("--graph")] = None,
    per_type: Annotated[int, typer.Option(help="Resolved relations per type.")] = 12,
    effective: Annotated[int, typer.Option(help="Effective-date extractions.")] = 20,
    seed: Annotated[int, typer.Option()] = 13,
) -> None:
    """A seeded, stratified sample of extracted relations and effective dates to check
    by hand (the verdicts go to data/intel/review.jsonl)."""
    graph = load_graph(graph_file or _graph_path(None))
    rng = random.Random(seed)
    by_type: dict[str, list[Any]] = defaultdict(list)
    for r in graph.relations:
        if r.cited is not None:
            by_type[r.type].append(r)
    items: list[dict[str, Any]] = []
    for rtype in sorted(by_type):
        group = sorted(by_type[rtype], key=lambda r: (r.citing, r.start))
        for r in rng.sample(group, min(per_type, len(group))):
            ci, cd = graph.intel[r.citing], graph.intel[r.cited]
            items.append({
                "id": f"rel:{r.citing}:{r.start}", "kind": "relation", "type": r.type,
                "cue": r.cue, "resolution": r.resolution, "mention": r.mention,
                "sentence": r.sentence,
                "citing": [r.citing, ci.title, str(ci.publication_date)],
                "cited": [r.cited, cd.title, str(cd.publication_date), cd.circular_number,
                          cd.ref_number],
            })  # fmt: skip
    with_dates = sorted(d for d, i in graph.intel.items() if i.effective.kind == "explicit")
    for d in rng.sample(with_dates, min(effective, len(with_dates))):
        i = graph.intel[d]
        items.append({
            "id": f"eff:{d}", "kind": "effective_date", "title": i.title,
            "published": str(i.publication_date), "effective": str(i.effective.date),
            "sentence": i.effective.evidence,
        })  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), "utf-8")
    meta = {
        "seed": seed,
        "per_type": per_type,
        "effective": effective,
        "resolved_relations_by_type": {t: len(v) for t, v in sorted(by_type.items())},
        "effective_explicit_documents": len(with_dates),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    typer.echo(f"wrote {len(items)} items to {out}")


@intel_app.command("precision")
def precision(
    sample_file: Annotated[Path, typer.Option("--sample")] = Path(
        "results/intel/review_sample.jsonl"
    ),
    review: Annotated[Path, typer.Option(help="Verdicts JSONL.")] = Path("data/intel/review.jsonl"),
    out: Annotated[Path, typer.Option()] = Path("results/intel/precision.json"),
) -> None:
    """Precision of the extracted relations (target and type) and effective dates on the
    reviewed sample, with Wilson 95% intervals; states who reviewed."""
    from rbirag.intel.review import precision_report

    report = precision_report(sample_file, review)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    typer.echo(json.dumps(report, indent=1))


@intel_app.command("acronyms")
def acronyms(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    out: Annotated[Path | None, typer.Option(help="Default: the packaged dictionary.")] = None,
) -> None:
    """Mine the RBI acronym dictionary used by query expansion from the corpus."""
    from rbirag.document import iter_documents_jsonl
    from rbirag.retrieval.expansion import mine_acronyms, save_acronyms

    s = get_settings()
    texts = [d.text for d in iter_documents_jsonl(docs or s.documents_path)]
    entries = mine_acronyms(texts)
    target = out or Path(__file__).resolve().parent.parent / "retrieval" / "acronyms.json"
    save_acronyms(entries, target)
    typer.echo(f"wrote {len(entries)} acronyms to {target}")
