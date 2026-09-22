"""Agent state.

State is explicit and inspectable rather than hidden inside a framework, because the
whole project depends on knowing exactly which documents the agent consulted at each
step. That is precisely the response-level logging prior work identified as missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from argus.corpus.store import Document


@dataclass
class EvidenceItem:
    """One document in the agent's working context."""

    doc_id: str
    text: str
    title: str = ""
    score: float = 0.0
    iteration: int = 0
    rank: int = 0
    #: How the document entered context: retrieval or inspection.
    via: str = "retrieval"

    def render(self, index: int) -> str:
        head = f"[DOC {index}]"
        if self.title:
            head += f" {self.title}"
        return f"{head}\n{self.text}"


@dataclass
class AgentState:
    """Everything the agent knows during one run."""

    question: str
    query_id: str = ""

    #: Search strings actually issued, starting with the original question.
    queries: list[str] = field(default_factory=list)
    #: Documents currently in context.
    evidence: list[EvidenceItem] = field(default_factory=list)
    #: Every document id ever seen, used to avoid re-retrieving the same thing.
    seen_doc_ids: set[str] = field(default_factory=set)

    iteration: int = 0
    n_rewrites: int = 0
    n_inspections: int = 0
    n_reflections: int = 0
    #: Mechanism responses the parsers could not read and had to fall back on. A run with
    #: failures here is not exercising the mechanism it claims to; the runner aggregates
    #: this per cell so a model that phrases things differently is caught early rather
    #: than silently producing a degraded grid.
    parse_failures: int = 0
    #: Mechanism responses parsed at all, the denominator for the rate above.
    parse_attempts: int = 0

    #: Reflection verdicts in order, for the plan-delta feature.
    reflection_history: list[str] = field(default_factory=list)
    #: Intermediate answers, one per iteration when reflection is enabled. Used by the
    #: answer-stability feature and by the analysis of when an agent changed its mind.
    intermediate_answers: list[str] = field(default_factory=list)

    final_answer: str = ""
    stopped_because: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def current_query(self) -> str:
        return self.queries[-1] if self.queries else self.question

    def record_parse(self, ok: bool) -> None:
        """Note whether a mechanism response was understood or fell back."""
        self.parse_attempts += 1
        if not ok:
            self.parse_failures += 1

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.parse_attempts if self.parse_attempts else 0.0

    def add_evidence(
        self,
        docs: list[Document],
        scores: list[float],
        iteration: int,
        via: str = "retrieval",
    ) -> int:
        """Add documents to context, skipping ones already present. Returns how many were new."""
        added = 0
        for rank, (doc, score) in enumerate(zip(docs, scores)):
            if doc is None or doc.doc_id in self.seen_doc_ids:
                continue
            self.evidence.append(
                EvidenceItem(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    title=doc.title,
                    score=float(score),
                    iteration=iteration,
                    rank=rank,
                    via=via,
                )
            )
            self.seen_doc_ids.add(doc.doc_id)
            added += 1
        return added

    def context_block(self, max_docs: int | None = None) -> str:
        """Render context for the prompt, best-scoring first."""
        items = sorted(self.evidence, key=lambda e: -e.score)
        if max_docs is not None:
            items = items[:max_docs]
        return "\n\n".join(item.render(i + 1) for i, item in enumerate(items))

    def context_doc_ids(self, max_docs: int | None = None) -> list[str]:
        items = sorted(self.evidence, key=lambda e: -e.score)
        if max_docs is not None:
            items = items[:max_docs]
        return [e.doc_id for e in items]

    @property
    def n_evidence(self) -> int:
        return len(self.evidence)
