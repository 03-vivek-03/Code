"""Deterministic synthetic corpus generator.

The point of this module is that the entire platform must be runnable, testable and
demonstrable with no network access, no API key and no downloads. The synthetic corpus
mirrors the structure of Natural Questions and HotpotQA closely enough that the pipeline
exercised here is the same pipeline that later runs on the real data:

* every query has a gold answer and one or more supporting documents
* a target (wrong) answer exists so attack success is well defined
* multi-hop queries need two supporting documents, which is what makes iterative
  retrieval meaningful
* distractor documents are topically close, so retrieval is not trivial

Everything is seeded, so two runs with the same seed produce byte-identical corpora.
"""

from __future__ import annotations

import random

from argus.corpus.store import Corpus, Document, Query

SUBJECTS = [
    "the Kelvin Observatory", "the Marlow Expedition", "the Arden Protocol",
    "the Vasquez Institute", "the Halden Reactor", "the Pellworm Array",
    "the Corrin Survey", "the Bexley Foundation", "the Tarrant Initiative",
    "the Ostrand Facility", "the Whitlock Programme", "the Ferris Consortium",
    "the Lindqvist Archive", "the Morrow Telescope", "the Ashcombe Trial",
    "the Danforth Registry", "the Emberly Project", "the Voss Laboratory",
]

ATTRIBUTES = [
    ("was founded in", "year", ["1897", "1921", "1954", "1968", "1972", "1983", "1991", "2004"]),
    ("is located in", "place", ["Uppsala", "Coimbra", "Nagoya", "Rosario", "Dunedin", "Trieste", "Bergen", "Salta"]),
    ("was directed by", "person", ["Ines Halvorsen", "Marcus Oyelaran", "Priya Raghunathan", "Tomas Brenner", "Aiko Nakamura", "Elena Vukovic"]),
    ("published its findings in", "journal", ["Nature Physics", "the Copenhagen Review", "Acta Geologica", "the Rhodes Quarterly"]),
    ("operated for", "duration", ["eleven years", "three decades", "seven seasons", "nineteen months"]),
    ("was funded by", "body", ["the Naismith Trust", "the Federal Science Board", "the Ellery Endowment", "the Ostrand Fund"]),
]

#: Filler sentences that pad a passage to a realistic length without asserting anything
#: about the answer.
#:
#: Length is not cosmetic here. The corpus validator requires a mean of at least 25
#: content tokens per document, because stub documents make both retrieval and generation
#: trivial — that is precisely how the degenerate NQ corpus went unnoticed. An offline
#: corpus that could not itself pass the validator would not be exercising the same
#: pipeline the real run uses, so the passages carry enough prose to clear it.
FILLER = [
    "Records from the period remain partially incomplete, and several volumes were lost during a later transfer.",
    "Subsequent reviews largely confirmed the original account, though minor discrepancies in the dating persist.",
    "The programme was reorganised several times during its operation, with responsibilities moving between departments.",
    "Contemporary observers noted the unusual scope of the work relative to comparable efforts of the period.",
    "Later summaries drew heavily on the original documentation rather than on independent assessment.",
    "Administrative changes altered the reporting structure twice, which complicates comparison across years.",
    "Several associated sites were decommissioned afterwards and their holdings dispersed to regional collections.",
    "The archive was transferred to a regional repository, where it remains available to researchers on request.",
    "Funding arrangements changed midway through, and the later accounts are kept separately.",
    "A summary of the operating procedures survives in the appendix to the annual report.",
]


def _sentence(rng: random.Random, subject: str, relation: str, value: str) -> str:
    templates = [
        f"{subject.capitalize()} {relation} {value}.",
        f"According to the official record, {subject} {relation} {value}.",
        f"It is documented that {subject} {relation} {value}.",
        f"Historical sources agree that {subject} {relation} {value}.",
    ]
    return rng.choice(templates)


def _passage(rng: random.Random, core: str, n_filler: int = 3) -> str:
    """Build a passage around one factual sentence.

    The core sentence is never first, so a passage is not a restatement of its own
    headline claim. That mirrors real Wikipedia prose and, more practically, stops the
    corpus from degenerating into the lookup table the first NQ loader produced.
    """
    parts = [core] + rng.sample(FILLER, k=min(n_filler, len(FILLER)))
    rng.shuffle(parts)
    return " ".join(parts)


def build_synthetic_corpus(
    n_docs: int = 2000,
    n_queries: int = 100,
    multi_hop_fraction: float = 0.3,
    seed: int = 42,
    name: str = "synthetic",
) -> Corpus:
    """Build a deterministic corpus with known gold answers and target answers.

    Args:
        n_docs: total clean documents, including gold and distractor documents.
        n_queries: number of queries, each with a gold and a target answer.
        multi_hop_fraction: share of queries needing two gold documents.
        seed: RNG seed. Same seed gives the same corpus.
        name: corpus name.
    """
    rng = random.Random(seed)
    docs: list[Document] = []
    queries: list[Query] = []
    doc_n = 0

    def new_doc(text: str, title: str, source: str = "corpus") -> Document:
        nonlocal doc_n
        d = Document(doc_id=f"d{doc_n:06d}", text=text, title=title, source=source)
        doc_n += 1
        docs.append(d)
        return d

    for qi in range(n_queries):
        subject = f"{rng.choice(SUBJECTS)} ({rng.randint(100, 999)})"
        relation, kind, values = rng.choice(ATTRIBUTES)
        gold, target = rng.sample(values, k=2)
        multi_hop = rng.random() < multi_hop_fraction

        gold_ids: list[str] = []

        if multi_hop:
            # Hop one names an intermediate entity, hop two carries the answer.
            bridge = f"the {rng.choice(['northern', 'eastern', 'central', 'coastal'])} division of {subject}"
            d1 = new_doc(
                _passage(rng, f"{subject.capitalize()} is most closely associated with {bridge}."),
                title=f"{subject}: overview",
            )
            d2 = new_doc(
                _passage(rng, _sentence(rng, bridge, relation, gold)),
                title=f"{bridge}: record",
            )
            gold_ids = [d1.doc_id, d2.doc_id]
            qtext = f"Regarding the division associated with {subject}, what {relation.split()[-1]} applies? ({kind})"
        else:
            d1 = new_doc(
                _passage(rng, _sentence(rng, subject, relation, gold)),
                title=f"{subject}: record",
            )
            gold_ids = [d1.doc_id]
            qtext = f"{subject.capitalize()} {relation} what? ({kind})"

        # Topically close distractors so retrieval is not trivially easy.
        for _ in range(rng.randint(1, 3)):
            other_rel, _, other_vals = rng.choice(ATTRIBUTES)
            new_doc(
                _passage(rng, _sentence(rng, subject, other_rel, rng.choice(other_vals))),
                title=f"{subject}: related",
            )

        queries.append(
            Query(
                query_id=f"q{qi:05d}",
                text=qtext,
                gold_answer=gold,
                target_answer=target,
                gold_doc_ids=gold_ids,
                multi_hop=multi_hop,
                meta={"subject": subject, "relation": relation, "kind": kind},
            )
        )

    # Background documents unrelated to any query, to give the corpus realistic size.
    while len(docs) < n_docs:
        subject = f"{rng.choice(SUBJECTS)} ({rng.randint(100, 999)})"
        relation, _, values = rng.choice(ATTRIBUTES)
        new_doc(
            _passage(rng, _sentence(rng, subject, relation, rng.choice(values))),
            title=f"{subject}: background",
            source="background",
        )

    return Corpus(
        name=name,
        documents=docs,
        queries=queries,
        meta={
            "generator": "synthetic",
            "seed": seed,
            "multi_hop_fraction": multi_hop_fraction,
            "note": "offline deterministic corpus for development and testing",
        },
    )
