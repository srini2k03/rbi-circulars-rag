"""NLI cross-check (no API needed): DeBERTa NLI cross-encoder, plus a fake for tests.

* Faithfulness (NLI): share of answer sentences entailed by at least one of the sources they
  cite (by any source when the sentence has no citation).
* Citation accuracy: share of (sentence, cited source) pairs where the source entails the
  sentence. Citations to non-existent sources count as wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from rbirag.generation.citations import CitationReport, strip_citations
from rbirag.index.bm25 import bm25_tokenize


@runtime_checkable
class NLIScorer(Protocol):
    name: str

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """P(entailment) for each (premise, hypothesis) pair."""
        ...


class CrossEncoderNLI:
    """``cross-encoder/nli-deberta-v3-base`` (labels read from the model config)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        *,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._entail_idx: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name, device=self.device, max_length=512)
            id2label = getattr(self._model.model.config, "id2label", {}) or {}
            for idx, label in id2label.items():
                if "entail" in str(label).lower():
                    self._entail_idx = int(idx)
            if self._entail_idx is None:
                self._entail_idx = 1  # documented order: contradiction, entailment, neutral
        return self._model

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        logits = np.asarray(
            self.model.predict(list(pairs), batch_size=self.batch_size, show_progress_bar=False)
        )
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        assert self._entail_idx is not None
        return [float(p) for p in probs[:, self._entail_idx]]


class FakeNLI:
    """Entailment = share of hypothesis content tokens present in the premise."""

    name = "fake-overlap-nli"

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for premise, hypothesis in pairs:
            h = set(bm25_tokenize(hypothesis))
            p = set(bm25_tokenize(premise))
            out.append(len(h & p) / len(h) if h else 0.0)
        return out


class NLIResult(BaseModel):
    faithfulness: float | None
    citation_accuracy: float | None
    n_sentences: int
    n_citation_pairs: int


def nli_check(
    scorer: NLIScorer,
    report: CitationReport,
    sources: Sequence[str],
    *,
    threshold: float = 0.5,
) -> NLIResult:
    if report.is_refusal:
        return NLIResult(
            faithfulness=None, citation_accuracy=None, n_sentences=0, n_citation_pairs=0
        )
    sentences = [(strip_citations(s.sentence), s.citations) for s in report.sentences]
    sentences = [(h, c) for h, c in sentences if len(h.split()) >= 3]
    if not sentences:
        return NLIResult(
            faithfulness=None, citation_accuracy=None, n_sentences=0, n_citation_pairs=0
        )

    pairs: list[tuple[str, str]] = []
    owners: list[tuple[int, int | None]] = []  # (sentence index, cited id or None)
    for si, (hyp, cited) in enumerate(sentences):
        valid = [c for c in cited if 1 <= c <= len(sources)]
        targets: list[int | None] = list(valid) if valid else [None]
        for c in targets:
            premises = [sources[c - 1]] if c is not None else list(sources)
            for prem in premises:
                pairs.append((prem, hyp))
                owners.append((si, c))
    probs = scorer.entailment(pairs) if pairs else []

    best_per_sentence = [0.0] * len(sentences)
    best_per_citation: dict[tuple[int, int], float] = {}
    for (si, c), p in zip(owners, probs, strict=True):
        best_per_sentence[si] = max(best_per_sentence[si], p)
        if c is not None:
            best_per_citation[(si, c)] = max(best_per_citation.get((si, c), 0.0), p)

    faithful = sum(p >= threshold for p in best_per_sentence) / len(sentences)
    n_pairs = sum(len(c) for _, c in sentences)
    correct = sum(1 for p in best_per_citation.values() if p >= threshold)
    return NLIResult(
        faithfulness=faithful,
        citation_accuracy=correct / n_pairs if n_pairs else None,
        n_sentences=len(sentences),
        n_citation_pairs=n_pairs,
    )
