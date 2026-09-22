"""Attack interface.

Every attack takes a clean corpus and returns a poisoned copy plus a record of what it
injected. The original corpus is never mutated, so the same clean baseline can be reused
across every attack and poison ratio in the grid.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from argus.corpus.store import Corpus, Document, Query


@dataclass
class AttackResult:
    """A poisoned corpus and the ground truth about what was injected."""

    corpus: Corpus
    poison_doc_ids: list[str] = field(default_factory=list)
    #: query_id -> poisoned document ids aimed at that query
    poison_by_query: dict[str, list[str]] = field(default_factory=dict)
    attack_name: str = ""
    n_poison_per_query: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_poison(self) -> int:
        return len(self.poison_doc_ids)

    def poison_for(self, query_id: str) -> set[str]:
        return set(self.poison_by_query.get(query_id, []))

    def summary(self) -> dict[str, Any]:
        return {
            "attack": self.attack_name,
            "n_poison": self.n_poison,
            "n_poison_per_query": self.n_poison_per_query,
            "n_targeted_queries": len(self.poison_by_query),
            "corpus_size": len(self.corpus),
            "poison_rate": round(self.n_poison / max(len(self.corpus), 1), 6),
        }


class Attack(ABC):
    """Base class for poisoning attacks."""

    name: str = "base"
    #: True when the attack needs access to the retriever, i.e. the white-box setting.
    needs_retriever: bool = False

    def __init__(self, n_poison_docs: int = 5, seed: int = 42) -> None:
        self.n_poison_docs = n_poison_docs
        self.seed = seed

    @abstractmethod
    def craft(self, query: Query, index: int) -> str:
        """Produce the text of one poisoned passage for a query."""

    def target_queries(self, corpus: Corpus, fraction: float = 1.0) -> list[Query]:
        import random

        queries = list(corpus.queries)
        if fraction >= 1.0:
            return queries
        rng = random.Random(self.seed)
        k = max(1, int(len(queries) * fraction))
        return rng.sample(queries, k)

    def apply(
        self,
        corpus: Corpus,
        target_fraction: float = 1.0,
        retriever: Any = None,
    ) -> AttackResult:
        """Inject poison into a copy of the corpus."""
        if self.needs_retriever and retriever is None:
            raise ValueError(
                f"attack '{self.name}' is white-box and needs a retriever; pass retriever="
            )

        poisoned = Corpus(
            name=f"{corpus.name}__{self.name}__p{self.n_poison_docs}",
            documents=copy.deepcopy(corpus.documents),
            queries=copy.deepcopy(corpus.queries),
            meta={**corpus.meta, "attack": self.name, "n_poison_docs": self.n_poison_docs},
        )

        self._prepare(poisoned, retriever)

        poison_ids: list[str] = []
        by_query: dict[str, list[str]] = {}
        counter = 0

        for query in self.target_queries(poisoned, target_fraction):
            ids_for_query: list[str] = []
            for i in range(self.n_poison_docs):
                text = self.craft(query, i)
                doc = Document(
                    doc_id=f"poison_{self.name}_{counter:06d}",
                    text=text,
                    title=self._title_for(query, i),
                    source="poison",
                    is_poison=True,
                    attack=self.name,
                    target_query_id=query.query_id,
                    meta={"variant": i, "target_answer": query.target_answer},
                )
                poisoned.documents.append(doc)
                poison_ids.append(doc.doc_id)
                ids_for_query.append(doc.doc_id)
                counter += 1
            by_query[query.query_id] = ids_for_query

        if hasattr(poisoned, "_doc_index"):
            del poisoned._doc_index

        return AttackResult(
            corpus=poisoned,
            poison_doc_ids=poison_ids,
            poison_by_query=by_query,
            attack_name=self.name,
            n_poison_per_query=self.n_poison_docs,
            meta={"target_fraction": target_fraction, "seed": self.seed},
        )

    # ---- hooks -----------------------------------------------------------
    def _prepare(self, corpus: Corpus, retriever: Any) -> None:
        """Optional setup before crafting, e.g. fitting a white-box optimiser."""

    def _title_for(self, query: Query, index: int) -> str:
        """Poison titles must not look obviously hostile, or retrieval becomes unfair."""
        return f"{query.text[:60]}"
