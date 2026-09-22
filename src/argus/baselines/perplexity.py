"""Perplexity-style input filter.

The standard first line of defence against injected text: adversarially optimised
passages tend to read unnaturally, so flag passages whose fluency looks wrong.

A real deployment would score fluency with a language model. Here the score is computed
from surface statistics, which keeps the baseline runnable offline and, more importantly,
keeps it honest: this is a cheap filter, and pretending otherwise would flatter the
comparison against trace-based detection.

Where it fails, and why that matters for the write-up: passages that are fluent but
false sail straight through. That is precisely what the black-box PoisonedRAG setting
produces.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any

from argus.baselines.base import BaselineDefense, BaselineResult
from argus.corpus.store import Corpus
from argus.retrieval.bm25 import tokenize
from argus.telemetry.spans import Trace


class PerplexityFilterDefense(BaselineDefense):
    """Surface-statistics fluency filter over the retrieved context."""

    name = "perplexity_filter"
    #: Runs on text only, so no extra generator passes.
    overhead_x = 0.0

    def __init__(self, corpus: Corpus, threshold: float = 0.5) -> None:
        super().__init__(threshold=threshold)
        self.corpus = corpus
        self._unigram: Counter[str] = Counter()
        self._total = 1
        self._fit()

    def _fit(self) -> None:
        """Fit a unigram model on clean documents, which is what a defender would have."""
        for doc in self.corpus.documents:
            if doc.is_poison:
                continue
            self._unigram.update(tokenize(doc.text))
        self._total = max(sum(self._unigram.values()), 1)

    def _passage_score(self, text: str) -> float:
        """Suspicion score in [0, 1]. Higher is more suspicious."""
        toks = tokenize(text)
        if not toks:
            return 0.0

        vocab = max(len(self._unigram), 1)
        # Add-one smoothed unigram negative log likelihood.
        nll = -sum(
            math.log((self._unigram.get(t, 0) + 1) / (self._total + vocab)) for t in toks
        ) / len(toks)

        counts = Counter(toks)
        repetition = 1.0 - len(counts) / len(toks)
        top_share = counts.most_common(1)[0][1] / len(toks)

        # Token repetition is the signature of the white-box attack, which stuffs
        # high-IDF query terms into the passage.
        norm_nll = min(nll / 15.0, 1.0)
        return min(0.55 * norm_nll + 0.30 * repetition + 0.15 * top_share, 1.0)

    def analyse(self, trace: Trace, **kwargs: Any) -> BaselineResult:
        t0 = time.perf_counter()
        texts: list[str] = []
        for doc_id in trace.context_doc_ids:
            doc = self.corpus.get(doc_id)
            if doc is not None:
                texts.append(doc.text)

        if not texts:
            return BaselineResult(score=0.0, flagged=False, latency_ms=0.0)

        scores = [self._passage_score(t) for t in texts]
        worst = max(scores)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return BaselineResult(
            score=worst,
            flagged=worst >= self.threshold,
            n_llm_calls=0,
            latency_ms=latency_ms,
            detail={"n_docs": len(texts), "mean_score": sum(scores) / len(scores)},
        )
