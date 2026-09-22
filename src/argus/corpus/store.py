"""Document store.

A corpus is a flat collection of documents plus a set of queries with gold answers.
Poisoned documents live in the same store as clean ones and are distinguished only by
their metadata, which is exactly the situation a real deployment faces: the retriever
has no idea which documents are hostile.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""
    source: str = "corpus"
    #: True when this document was injected by an attack. Never visible to the agent.
    is_poison: bool = False
    #: Which attack injected it, if any.
    attack: str = ""
    #: Query this poison document targets, if it is targeted.
    target_query_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        return cls(**d)


@dataclass
class Query:
    query_id: str
    text: str
    #: The correct answer. Used to compute clean accuracy.
    gold_answer: str
    #: The answer an attacker wants the system to produce. Used to compute attack success.
    target_answer: str = ""
    #: Document ids that legitimately support the gold answer.
    gold_doc_ids: list[str] = field(default_factory=list)
    #: Multi-hop queries need more than one gold document to be answerable.
    multi_hop: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Query:
        return cls(**d)


@dataclass
class Corpus:
    name: str
    documents: list[Document] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- accessors
    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    @property
    def doc_index(self) -> dict[str, Document]:
        if not hasattr(self, "_doc_index") or len(self._doc_index) != len(self.documents):
            self._doc_index = {d.doc_id: d for d in self.documents}
        return self._doc_index

    def get(self, doc_id: str) -> Document | None:
        return self.doc_index.get(doc_id)

    def query_index(self) -> dict[str, Query]:
        return {q.query_id: q for q in self.queries}

    @property
    def poison_ids(self) -> set[str]:
        return {d.doc_id for d in self.documents if d.is_poison}

    @property
    def n_poison(self) -> int:
        return len(self.poison_ids)

    def poison_for(self, query_id: str) -> set[str]:
        """Poison documents targeted at one query, plus any untargeted poison."""
        return {
            d.doc_id
            for d in self.documents
            if d.is_poison and d.target_query_id in ("", query_id)
        }

    # ------------------------------------------------------------------ mutation
    def add(self, docs: Iterable[Document]) -> None:
        self.documents.extend(docs)
        if hasattr(self, "_doc_index"):
            del self._doc_index

    def without_poison(self) -> Corpus:
        """A clean copy, used to measure clean accuracy against the same queries."""
        return Corpus(
            name=f"{self.name}_clean",
            documents=[d for d in self.documents if not d.is_poison],
            queries=list(self.queries),
            meta={**self.meta, "derived": "without_poison"},
        )

    # --------------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "meta": self.meta,
            "n_documents": len(self.documents),
            "n_queries": len(self.queries),
            "n_poison": self.n_poison,
        }
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"__header__": payload}) + "\n")
            for d in self.documents:
                fh.write(json.dumps({"__doc__": d.to_dict()}) + "\n")
            for q in self.queries:
                fh.write(json.dumps({"__query__": q.to_dict()}) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Corpus:
        path = Path(path)
        docs: list[Document] = []
        queries: list[Query] = []
        name, meta = path.stem, {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "__header__" in rec:
                    name = rec["__header__"].get("name", name)
                    meta = rec["__header__"].get("meta", {})
                elif "__doc__" in rec:
                    docs.append(Document.from_dict(rec["__doc__"]))
                elif "__query__" in rec:
                    queries.append(Query.from_dict(rec["__query__"]))
        return cls(name=name, documents=docs, queries=queries, meta=meta)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_documents": len(self.documents),
            "n_clean": len(self.documents) - self.n_poison,
            "n_poison": self.n_poison,
            "n_queries": len(self.queries),
            "n_multi_hop": sum(1 for q in self.queries if q.multi_hop),
            "poison_rate": round(self.n_poison / max(len(self.documents), 1), 5),
        }
