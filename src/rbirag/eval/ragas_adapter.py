"""Optional RAGAS adapter.

Converts generation records into RAGAS samples and, if the ``ragas`` package is installed,
runs its metrics with caller-supplied LLM/embedding wrappers. Nothing in CI depends on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from rbirag.generation.citations import strip_citations


class RagasSample(BaseModel):
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str | None = None


def to_ragas_samples(
    questions: Sequence[str],
    answers: Sequence[str],
    contexts: Sequence[Sequence[str]],
    references: Sequence[str | None] | None = None,
) -> list[RagasSample]:
    refs = list(references) if references is not None else [None] * len(questions)
    return [
        RagasSample(
            user_input=q,
            response=strip_citations(a),
            retrieved_contexts=list(c),
            reference=r or None,
        )
        for q, a, c, r in zip(questions, answers, contexts, refs, strict=True)
    ]


def run_ragas(samples: Sequence[RagasSample], *, llm: Any, embeddings: Any) -> dict[str, float]:
    """Run RAGAS faithfulness + answer relevancy. Requires ``pip install ragas``."""
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import Faithfulness, ResponseRelevancy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("RAGAS is not installed; pip install ragas") from exc
    dataset = EvaluationDataset.from_list([s.model_dump() for s in samples])  # pragma: no cover
    result = evaluate(  # pragma: no cover
        dataset, metrics=[Faithfulness(), ResponseRelevancy()], llm=llm, embeddings=embeddings
    )
    df = result.to_pandas()  # pragma: no cover
    return {c: float(df[c].mean()) for c in ("faithfulness", "answer_relevancy") if c in df}
