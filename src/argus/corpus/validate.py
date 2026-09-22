"""Corpus validation.

This module exists because of a failure that was expensive to find. An earlier build of
the Natural Questions loader created one document per question whose entire text was::

    f"{question} The answer is {gold}."

That is not a retrieval corpus. It is a lookup table keyed by the question string. BM25
matched the question verbatim, clean accuracy read 0.98, and every downstream number
described a task no real RAG system performs. 45 GPU-hours of experiments were run on it
before the problem was visible, because nothing in the pipeline ever asserted that the
corpus looked like a corpus.

So the checks below run at build time and the builder refuses to save a corpus that
fails them. A validator that only warns would have been ignored exactly as the silent
version was.

Each check states what it measures, what threshold it applies and why that threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from argus.corpus.answers import answer_type, type_match_rate
from argus.corpus.store import Corpus
from argus.retrieval.bm25 import tokenize


class CorpusValidationError(ValueError):
    """Raised when a corpus is structurally unfit for the experiment."""


@dataclass
class Check:
    name: str
    value: float
    threshold: float
    ok: bool
    direction: str  # "min" or "max"
    why: str

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        rel = ">=" if self.direction == "min" else "<="
        return f"  [{mark}] {self.name:<26s} {self.value:>8.3f}  (need {rel} {self.threshold})"


@dataclass
class ValidationReport:
    corpus_name: str
    checks: list[Check] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def text(self) -> str:
        head = f"Corpus validation: {self.corpus_name}"
        body = "\n".join(c.line() for c in self.checks)
        stats = "\n".join(f"  {k:<26s} {v}" for k, v in self.stats.items())
        verdict = "OK" if self.ok else f"FAILED ({len(self.failures)} check(s))"
        return f"{head}\n{body}\n\n  --- statistics ---\n{stats}\n\n  verdict: {verdict}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus_name,
            "ok": self.ok,
            "checks": [
                {
                    "name": c.name,
                    "value": c.value,
                    "threshold": c.threshold,
                    "direction": c.direction,
                    "ok": c.ok,
                    "why": c.why,
                }
                for c in self.checks
            ],
            "stats": self.stats,
        }


def _content_tokens(text: str) -> list[str]:
    return tokenize(text, remove_stopwords=True)


def question_echo_score(query_text: str, doc_text: str) -> float:
    """Share of the query's content tokens that appear in the document.

    A genuine supporting passage shares vocabulary with the question, so this is never
    zero. What marks a degenerate corpus is a value at or near 1.0 across the board,
    which means the document is a restatement of the question and retrieval is a string
    match rather than a search.
    """
    q = set(_content_tokens(query_text))
    if not q:
        return 0.0
    d = set(_content_tokens(doc_text))
    return len(q & d) / len(q)


def validate_corpus(
    corpus: Corpus,
    strict: bool = True,
    min_doc_tokens: float = 25.0,
    max_echo_rate: float = 0.35,
    min_answer_present: float = 0.70,
    min_type_match: float = 0.60,
    max_poison_rate: float = 0.20,
) -> ValidationReport:
    """Check that a corpus can support a meaningful poisoning experiment.

    Args:
        corpus: the corpus to inspect.
        strict: raise :class:`CorpusValidationError` on failure instead of returning.
        min_doc_tokens: mean content tokens per document. Single-sentence stub
            documents make both retrieval and generation trivial.
        max_echo_rate: share of queries whose gold document echoes almost the whole
            question. Above this the corpus is a lookup table.
        min_answer_present: share of queries whose gold document actually contains the
            gold answer. Below this the task is unanswerable and clean accuracy is noise.
        min_type_match: share of (gold, target) pairs sharing an answer type. Below this
            the attacks are being handed implausible targets.
        max_poison_rate: share of the corpus that is poison. PoisonedRAG injects five
            documents into a million; a corpus that is mostly poison is a different
            threat model and its retrieval statistics are not comparable.
    """
    docs = corpus.documents
    queries = corpus.queries
    n_docs = max(len(docs), 1)
    n_queries = max(len(queries), 1)

    clean_docs = [d for d in docs if not d.is_poison]
    mean_tokens = sum(len(_content_tokens(d.text)) for d in clean_docs) / max(
        len(clean_docs), 1
    )

    echoes = 0
    answer_present = 0
    resolvable = 0
    for q in queries:
        gold_texts = [corpus.get(d).text for d in q.gold_doc_ids if corpus.get(d)]
        if not gold_texts:
            continue
        resolvable += 1
        joined = " ".join(gold_texts)
        if max(question_echo_score(q.text, t) for t in gold_texts) >= 0.90:
            echoes += 1
        if q.gold_answer and q.gold_answer.lower() in joined.lower():
            answer_present += 1

    resolvable = max(resolvable, 1)
    echo_rate = echoes / resolvable
    answer_rate = answer_present / resolvable
    tmatch = type_match_rate(
        [(q.gold_answer, q.target_answer) for q in queries if q.target_answer]
    )
    poison_rate = corpus.n_poison / n_docs
    distinct_targets = len({q.target_answer for q in queries if q.target_answer})

    checks = [
        Check(
            "mean_doc_tokens", mean_tokens, min_doc_tokens,
            mean_tokens >= min_doc_tokens, "min",
            "Stub documents make retrieval and generation trivial.",
        ),
        Check(
            "question_echo_rate", echo_rate, max_echo_rate,
            echo_rate <= max_echo_rate, "max",
            "Gold documents that restate the question turn retrieval into a string match.",
        ),
        Check(
            "gold_answer_present", answer_rate, min_answer_present,
            answer_rate >= min_answer_present, "min",
            "If the gold document lacks the answer the task is unanswerable.",
        ),
        Check(
            "target_type_match", tmatch, min_type_match,
            tmatch >= min_type_match, "min",
            "Implausible targets let the generator dismiss poison without reasoning.",
        ),
        Check(
            "poison_rate", poison_rate, max_poison_rate,
            poison_rate <= max_poison_rate, "max",
            "A corpus that is mostly poison is not the published threat model.",
        ),
        Check(
            "queries_with_gold_docs", resolvable / n_queries, 0.95,
            resolvable / n_queries >= 0.95, "min",
            "Every query needs at least one supporting document.",
        ),
    ]

    report = ValidationReport(
        corpus_name=corpus.name,
        checks=checks,
        stats={
            "n_documents": len(docs),
            "n_clean": len(clean_docs),
            "n_poison": corpus.n_poison,
            "n_queries": len(queries),
            "n_multi_hop": sum(1 for q in queries if q.multi_hop),
            "distinct_target_answers": distinct_targets,
            "answer_types": _type_histogram(queries),
        },
    )

    if strict and not report.ok:
        raise CorpusValidationError(
            report.text()
            + "\n\nThe corpus is not fit for a poisoning experiment. "
            "See docs/RUNBOOK.md section 'Corpus validity' before running the grid."
        )
    return report


def _type_histogram(queries: list) -> dict[str, int]:
    hist: dict[str, int] = {}
    for q in queries:
        if q.gold_answer:
            key = answer_type(q.gold_answer)
            hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def validate_attack_retrievability(
    corpus: Corpus,
    poison_by_query: dict[str, list[str]],
    retriever: Any,
    top_k: int = 5,
    sample: int = 100,
    min_rate: float = 0.30,
    strict: bool = True,
) -> dict[str, Any]:
    """Check that an attack's poison is actually retrievable.

    The retrieval condition is half of every poisoning attack. An attack whose poison
    never enters the top-k contributes nothing but null rows, and if it is one of the
    leave-one-attack-out folds it drags the headline detection metric below chance.

    This is not hypothetical: the original corpus-poisoning implementation placed poison
    in the top-5 for 7 runs out of 22,000 (0.03%) and was never noticed, because nothing
    checked.
    """
    if not retriever._built:
        retriever.build()

    queries = [q for q in corpus.queries if poison_by_query.get(q.query_id)][:sample]
    if not queries:
        raise ValueError("no targeted queries to check retrievability against")

    hits = 0
    ranks: list[int] = []
    for q in queries:
        poison = set(poison_by_query.get(q.query_id, []))
        res = retriever.retrieve(q.text, top_k=top_k)
        found = [i for i, d in enumerate(res.doc_ids) if d in poison]
        if found:
            hits += 1
            ranks.append(found[0])

    rate = hits / len(queries)
    out = {
        "n_checked": len(queries),
        "poison_in_top_k_rate": rate,
        "mean_first_poison_rank": (sum(ranks) / len(ranks)) if ranks else -1.0,
        "top_k": top_k,
        "ok": rate >= min_rate,
    }

    if strict and not out["ok"]:
        raise CorpusValidationError(
            f"Attack poison reaches the top-{top_k} for only {rate:.1%} of sampled "
            f"queries (need >= {min_rate:.0%}).\n"
            "The retrieval condition of the attack is not satisfied, so this attack "
            "would contribute only null rows to the grid and would corrupt the "
            "leave-one-attack-out evaluation.\n"
            "Fix the attack's craft() method or drop it from the grid."
        )
    return out
