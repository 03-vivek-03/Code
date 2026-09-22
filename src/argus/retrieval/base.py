"""Retriever interface.

Retrieval scores are a first-class output, not an implementation detail. The detector in
Study B reads the score distribution and how it drifts across iterations, so every
backend must return calibrated, comparable scores alongside the documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from argus.corpus.store import Corpus, Document


@dataclass
class RetrievalResult:
    """One retrieval event."""

    query: str
    doc_ids: list[str]
    scores: list[float]
    documents: list[Document] = field(default_factory=list)
    backend: str = ""
    latency_ms: float = 0.0
    #: Rank each returned document holds in the *unfiltered* ranking for this query.
    #:
    #: Without this, ranks recorded during iterative retrieval are meaningless across
    #: rounds. Round two excludes the five documents already seen, so a document that
    #: truly sits at rank 5 is reported at rank 0. That is why the first run showed
    #: iterative retrieval with a better mean poison rank (0.28) than vanilla (0.50):
    #: the number was an artefact of re-indexing after exclusion, not a real improvement
    #: in ranking.
    global_ranks: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.doc_ids)

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0

    @property
    def score_gap(self) -> float:
        """Gap between the best and worst retrieved score.

        A poisoned document optimised for retrieval often sits far above the rest, so
        this gap is one of the cheaper signals available to the detector.
        """
        if len(self.scores) < 2:
            return 0.0
        return float(self.scores[0] - self.scores[-1])

    def poison_ranks(self, poison_ids: set[str]) -> list[int]:
        """Zero-based ranks at which poisoned documents appear."""
        return [i for i, d in enumerate(self.doc_ids) if d in poison_ids]

    def n_poison(self, poison_ids: set[str]) -> int:
        return len(self.poison_ranks(poison_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "doc_ids": list(self.doc_ids),
            "scores": [round(float(s), 6) for s in self.scores],
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 3),
        }


class Retriever(ABC):
    """Base class for all retrieval backends."""

    name: str = "base"

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._built = False

    @abstractmethod
    def build(self) -> None:
        """Index the corpus. Must be idempotent."""

    @abstractmethod
    def _search(self, query: str, top_k: int) -> tuple[list[str], list[float]]:
        """Return (doc_ids, scores) sorted by descending score."""

    def retrieve(self, query: str, top_k: int = 5, exclude: set[str] | None = None) -> RetrievalResult:
        """Retrieve documents for a query.

        Args:
            query: the search string.
            top_k: how many documents to return.
            exclude: documents already seen. Iterative retrieval passes the running set
                here so later rounds surface new evidence rather than repeating round one.
        """
        import time

        if not self._built:
            self.build()

        t0 = time.perf_counter()
        # Over-fetch so exclusion still leaves top_k results.
        fetch = top_k + (len(exclude) if exclude else 0)
        doc_ids, scores = self._search(query, fetch)

        # Position in the unfiltered ranking, captured before exclusion is applied.
        ranked = list(enumerate(zip(doc_ids, scores)))
        if exclude:
            ranked = [(g, ds) for g, ds in ranked if ds[0] not in exclude]

        ranked = ranked[:top_k]
        global_ranks = [g for g, _ in ranked]
        doc_ids = [d for _, (d, _) in ranked]
        scores = [s for _, (_, s) in ranked]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RetrievalResult(
            query=query,
            doc_ids=doc_ids,
            scores=scores,
            documents=[self.corpus.get(d) for d in doc_ids if self.corpus.get(d)],
            backend=self.name,
            latency_ms=latency_ms,
            global_ranks=global_ranks,
        )
