"""Dataset loaders.

Four datasets are supported:

* ``synthetic``  offline, deterministic, no downloads. Development and tests only.
* ``squad``      SQuAD v1.1. Real Wikipedia paragraphs with short answer spans. Light
                 (about 35 MB) and the recommended primary corpus when bandwidth or
                 disk on the run host is limited.
* ``nq``         Natural Questions, joined against the BEIR NQ passage corpus so that
                 retrieval happens over real Wikipedia passages. This is the canonical
                 dataset named in the proposal.
* ``hotpotqa``   HotpotQA distractor setting, used for the multi-hop confirmation run.

What changed, and why it matters
--------------------------------

The previous implementation of :func:`_build_nq` did not use passages at all. NQ-open
ships only question and answer strings, and rather than fetching a passage corpus the
loader synthesised one document per question whose entire text was::

    f"{question} The answer is {gold}."

Every downstream number inherited that choice. Retrieval was a verbatim string match, so
clean accuracy read 0.98; the poisoned passage competed against a one-sentence stub
rather than real prose; and comparison with published attack success rates was
meaningless. The HotpotQA run, which did use real paragraphs, was the control that gave
it away: accuracy there collapsed to 0.09 on the same pipeline.

So the real loaders now retrieve real passages, target answers are type-matched (see
:mod:`argus.corpus.answers`), and every corpus is validated before it is saved
(:mod:`argus.corpus.validate`). The synthetic generator remains, clearly scoped to
development, and is never silently substituted for a real dataset.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from argus.corpus.answers import TargetAnswerPicker
from argus.corpus.store import Corpus, Document, Query
from argus.corpus.synthetic import build_synthetic_corpus
from argus.corpus.validate import validate_corpus

#: Datasets whose documents are real prose. The synthetic generator is excluded on
#: purpose: it is for tests and demos, never for a reported result.
REAL_DATASETS = ("squad", "nq", "hotpotqa")

_WS = re.compile(r"\s+")


def save_corpus(corpus: Corpus, path: str | Path) -> Path:
    return corpus.save(path)


def load_corpus(path: str | Path) -> Corpus:
    return Corpus.load(path)


def recommended_n_docs(n_queries: int, max_poison_per_query: int = 10) -> int:
    """Clean-corpus size that keeps the poison rate defensible.

    Attacks inject ``n_poison_docs`` documents for every target query, so poison volume
    scales with the query count. With 300 queries and 10 poisoned documents each the
    corpus gains 3,000 hostile documents; against a 2,000-document corpus that is a
    corpus which is 60% poison, which is not the threat model any of the cited papers
    evaluate.

    This returns a clean size that keeps poison at or below roughly 10%.
    """
    poison = n_queries * max_poison_per_query
    return max(5_000, int(poison * 9))


def build_corpus(
    dataset: str = "synthetic",
    n_docs: int = 2000,
    n_queries: int = 100,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    validate: bool = True,
) -> Corpus:
    """Build or load a corpus by dataset name.

    Args:
        dataset: one of ``synthetic``, ``squad``, ``nq``, ``hotpotqa``.
        n_docs: total clean documents, including gold passages and distractors.
        n_queries: number of queries, each with a gold and a type-matched target answer.
        seed: RNG seed. The same seed reproduces the same corpus.
        cache_dir: HuggingFace datasets cache directory.
        validate: run :func:`argus.corpus.validate.validate_corpus` and refuse to
            return a structurally unfit corpus. Only tests should disable this.
    """
    dataset = dataset.lower()
    builders = {
        "synthetic": _build_synthetic,
        "squad": _build_squad,
        "nq": _build_nq,
        "natural_questions": _build_nq,
        "hotpotqa": _build_hotpotqa,
        "hotpot": _build_hotpotqa,
    }
    if dataset not in builders:
        raise ValueError(
            f"unknown dataset '{dataset}', expected one of: "
            "synthetic, squad, nq, hotpotqa"
        )

    corpus = builders[dataset](
        n_docs=n_docs, n_queries=n_queries, seed=seed, cache_dir=cache_dir
    )

    if validate and dataset != "synthetic":
        validate_corpus(corpus, strict=True)
    return corpus


# --------------------------------------------------------------------- synthetic
def _build_synthetic(n_docs: int, n_queries: int, seed: int, cache_dir=None) -> Corpus:
    return build_synthetic_corpus(n_docs=n_docs, n_queries=n_queries, seed=seed)


# ------------------------------------------------------------------------ shared
def _require_datasets():
    import importlib.util

    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        if importlib.util.find_spec("datasets") is None:
            raise ImportError(
                "Loading real datasets needs the 'datasets' package.\n"
                "  pip install datasets\n"
                "Until then use --dataset synthetic, which runs fully offline but is "
                "for development only and must never be used for a reported result."
            ) from exc
        raise ImportError(
            "'datasets' is installed but failed to import, which normally means one of "
            "its dependencies is missing or version-incompatible.\n"
            f"  underlying error: {type(exc).__name__}: {exc}\n"
            "  try: pip install --upgrade --force-reinstall datasets huggingface_hub pyarrow fsspec"
        ) from exc
    return datasets


def _clean_text(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("​", " ")).strip()


def _norm_question(text: str) -> str:
    return _WS.sub(" ", (text or "").lower().strip().rstrip("?").strip())


def _assign_targets(queries: list[Query], seed: int) -> TargetAnswerPicker:
    """Fill in every query's target answer with a same-type plausible alternative."""
    picker = TargetAnswerPicker([q.gold_answer for q in queries], seed=seed)
    for q in queries:
        q.target_answer = picker.pick(q.gold_answer, q.query_id)
    return picker


def _pad_with_distractors(
    corpus: Corpus, n_docs: int, pool: list[Document], seed: int
) -> None:
    """Top up a corpus to ``n_docs`` using real passages held back from the dataset.

    The earlier implementation padded real corpora with synthetic filler, which meant a
    "Natural Questions" corpus was part invented prose. Distractors now come from the
    same distribution as the gold passages, which is what makes retrieval a real
    ranking problem.
    """
    need = n_docs - len(corpus.documents)
    if need <= 0:
        return
    if not pool:
        raise ValueError(
            f"corpus '{corpus.name}' needs {need} more documents but the dataset "
            "supplied no spare passages. Lower --n-docs or raise --n-queries."
        )

    rng = random.Random(seed + 1)
    chosen = pool if len(pool) <= need else rng.sample(pool, need)
    offset = len(corpus.documents)
    for j, d in enumerate(chosen):
        d.doc_id = f"bg{offset + j:07d}"
        d.source = "distractor"
    corpus.add(chosen)


# ---------------------------------------------------------------------- SQuAD
def _build_squad(
    n_docs: int, n_queries: int, seed: int, cache_dir: str | Path | None
) -> Corpus:  # pragma: no cover - requires network
    """SQuAD v1.1: real Wikipedia paragraphs with short answer spans.

    Each question's gold document is the paragraph the question was written from, so the
    answer is genuinely present but the paragraph is not a restatement of the question.
    Remaining paragraphs from the same and other articles become topically plausible
    distractors.
    """
    ds = _require_datasets()
    raw = ds.load_dataset("rajpurkar/squad", split="validation", cache_dir=cache_dir)
    raw = raw.shuffle(seed=seed)

    docs: list[Document] = []
    queries: list[Query] = []
    spare: list[Document] = []

    #: One paragraph can serve many questions; index it once and reuse the id.
    para_id: dict[str, str] = {}
    doc_n = 0

    for row in raw:
        context = _clean_text(row["context"])
        answers = row.get("answers", {}).get("text") or []
        if not context or not answers:
            continue
        gold = _clean_text(answers[0])
        if not gold:
            continue

        if context not in para_id:
            did = f"d{doc_n:07d}"
            doc_n += 1
            para_id[context] = did
            doc = Document(
                doc_id=did,
                text=context,
                title=_clean_text(row.get("title", "")).replace("_", " "),
                source="squad",
            )
            if len(queries) < n_queries:
                docs.append(doc)
            else:
                spare.append(doc)
        elif len(queries) >= n_queries:
            continue

        if len(queries) >= n_queries:
            continue

        queries.append(
            Query(
                query_id=f"q{len(queries):05d}",
                text=_clean_text(row["question"]),
                gold_answer=gold,
                target_answer="",  # filled in below
                gold_doc_ids=[para_id[context]],
                multi_hop=False,
                meta={"dataset": "squad", "title": row.get("title", "")},
            )
        )

    if len(queries) < n_queries:
        raise ValueError(
            f"SQuAD supplied only {len(queries)} usable queries, fewer than the "
            f"{n_queries} requested."
        )

    picker = _assign_targets(queries, seed)
    corpus = Corpus(
        name="squad",
        documents=docs,
        queries=queries,
        meta={"dataset": "squad", "seed": seed, "target_answers": picker.stats()},
    )
    _pad_with_distractors(corpus, n_docs, spare, seed)
    return corpus


# ------------------------------------------------------------------------- NQ
def _build_nq(
    n_docs: int, n_queries: int, seed: int, cache_dir: str | Path | None
) -> Corpus:  # pragma: no cover - requires network
    """Natural Questions over the BEIR NQ passage corpus.

    NQ-open ships question and answer strings only, with no passages. BEIR's NQ is built
    from the same questions and does ship the Wikipedia passage corpus plus relevance
    judgements, so the two are joined on the normalised question text: NQ-open supplies
    the short answer, BEIR supplies the passage that supports it and the distractor pool.

    This download is large (roughly 3 GB for the passage corpus). ``--dataset squad`` is
    a fully real-passage alternative at about 35 MB if that is impractical on the run
    host; the RUNBOOK explains when each is appropriate.
    """
    ds = _require_datasets()
    rng = random.Random(seed)

    # ---- 1. short answers, from NQ-open ---------------------------------------
    qa = ds.load_dataset(
        "google-research-datasets/nq_open", split="validation", cache_dir=cache_dir
    )
    answer_of: dict[str, str] = {}
    for row in qa:
        answers = row.get("answer") or []
        if answers:
            answer_of.setdefault(_norm_question(row["question"]), answers[0])

    # ---- 2. BEIR queries and relevance judgements -----------------------------
    try:
        beir_q = ds.load_dataset("BeIR/nq", "queries", split="queries", cache_dir=cache_dir)
        qrels = ds.load_dataset("BeIR/nq-qrels", split="test", cache_dir=cache_dir)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the BEIR NQ passage corpus, which is required for real "
            f"passage retrieval on Natural Questions.\n  underlying error: {exc}\n"
            "Use --dataset squad for a light real-passage alternative, or --dataset "
            "hotpotqa for the multi-hop corpus. Do not fall back to --dataset "
            "synthetic for a reported result."
        ) from exc

    text_of_qid = {str(r["_id"]): _clean_text(r["text"]) for r in beir_q}

    gold_docs_of: dict[str, list[str]] = {}
    for r in qrels:
        if int(r.get("score", 0)) <= 0:
            continue
        gold_docs_of.setdefault(str(r["query-id"]), []).append(str(r["corpus-id"]))

    # Keep only queries that have both a short answer and a judged passage.
    usable: list[tuple[str, str, str, list[str]]] = []
    for qid, qtext in text_of_qid.items():
        gold_ids = gold_docs_of.get(qid)
        if not gold_ids:
            continue
        answer = answer_of.get(_norm_question(qtext))
        if not answer:
            continue
        usable.append((qid, qtext, answer, gold_ids))

    if len(usable) < n_queries:
        raise ValueError(
            f"only {len(usable)} NQ queries have both a short answer and a judged "
            f"passage, fewer than the {n_queries} requested. Lower --n-queries or use "
            "--dataset squad."
        )

    rng.shuffle(usable)
    usable = usable[:n_queries]
    wanted_gold = {d for _, _, _, ids in usable for d in ids}

    # ---- 3. stream the passage corpus ------------------------------------------
    # 2.68M passages: streamed and reservoir-sampled so peak memory stays flat.
    corpus_stream = ds.load_dataset(
        "BeIR/nq", "corpus", split="corpus", streaming=True, cache_dir=cache_dir
    )
    n_distractors_wanted = max(n_docs - len(wanted_gold), 0)

    gold_by_id: dict[str, Document] = {}
    reservoir: list[Document] = []
    seen = 0
    sample_rng = random.Random(seed + 7)

    for row in corpus_stream:
        did = str(row["_id"])
        doc = Document(
            doc_id=did,
            text=_clean_text(row.get("text", "")),
            title=_clean_text(row.get("title", "")),
            source="nq",
        )
        if not doc.text:
            continue
        if did in wanted_gold:
            gold_by_id[did] = doc
            continue
        # Reservoir sampling keeps a uniform distractor sample in one pass.
        seen += 1
        if len(reservoir) < n_distractors_wanted:
            reservoir.append(doc)
        else:
            j = sample_rng.randrange(seen)
            if j < n_distractors_wanted:
                reservoir[j] = doc

    docs: list[Document] = []
    queries: list[Query] = []
    for i, (_, qtext, answer, gold_ids) in enumerate(usable):
        present = [d for d in gold_ids if d in gold_by_id]
        if not present:
            continue
        for d in present:
            if gold_by_id[d] not in docs:
                docs.append(gold_by_id[d])
        queries.append(
            Query(
                query_id=f"q{i:05d}",
                text=qtext,
                gold_answer=answer,
                target_answer="",
                gold_doc_ids=present,
                multi_hop=False,
                meta={"dataset": "nq"},
            )
        )

    picker = _assign_targets(queries, seed)
    corpus = Corpus(
        name="nq",
        documents=docs,
        queries=queries,
        meta={"dataset": "nq", "seed": seed, "target_answers": picker.stats()},
    )
    _pad_with_distractors(corpus, n_docs, reservoir, seed)
    return corpus


# ------------------------------------------------------------------- HotpotQA
def _build_hotpotqa(
    n_docs: int, n_queries: int, seed: int, cache_dir: str | Path | None
) -> Corpus:  # pragma: no cover - requires network
    """HotpotQA distractor setting: real paragraphs, multi-hop questions.

    Each question ships ten paragraphs, two of which are supporting. The eight others are
    genuine hard distractors chosen by the dataset authors, so this corpus exercises
    iterative retrieval in a way single-hop data cannot.
    """
    ds = _require_datasets()
    raw = ds.load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split="validation", cache_dir=cache_dir
    )
    raw = raw.shuffle(seed=seed).select(range(min(n_queries, len(raw))))

    docs: list[Document] = []
    queries: list[Query] = []
    spare: list[Document] = []
    doc_n = 0

    for i, row in enumerate(raw):
        gold = _clean_text(row["answer"])
        if not gold:
            continue
        ctx = row["context"]
        titles, sentences = ctx["title"], ctx["sentences"]
        support_titles = set(row["supporting_facts"]["title"])

        gold_ids: list[str] = []
        for title, sents in zip(titles, sentences):
            did = f"d{doc_n:07d}"
            doc_n += 1
            doc = Document(
                doc_id=did,
                text=_clean_text(" ".join(sents)),
                title=_clean_text(title),
                source="hotpotqa",
            )
            docs.append(doc)
            if title in support_titles:
                gold_ids.append(did)

        if not gold_ids:
            continue

        queries.append(
            Query(
                query_id=f"q{i:05d}",
                text=_clean_text(row["question"]),
                gold_answer=gold,
                target_answer="",
                gold_doc_ids=gold_ids,
                multi_hop=True,
                meta={"dataset": "hotpotqa", "level": row.get("level", "")},
            )
        )

    picker = _assign_targets(queries, seed)
    corpus = Corpus(
        name="hotpotqa",
        documents=docs,
        queries=queries,
        meta={"dataset": "hotpotqa", "seed": seed, "target_answers": picker.stats()},
    )

    # HotpotQA already supplies eight distractors per question, so padding is only
    # needed when a very large corpus is requested.
    if len(docs) < n_docs:
        extra = ds.load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split="train", cache_dir=cache_dir
        )
        extra = extra.shuffle(seed=seed + 3).select(range(min(2000, len(extra))))
        for row in extra:
            ctx = row["context"]
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                spare.append(
                    Document(
                        doc_id="tmp",
                        text=_clean_text(" ".join(sents)),
                        title=_clean_text(title),
                        source="hotpotqa",
                    )
                )
            if len(docs) + len(spare) > n_docs * 2:
                break
        _pad_with_distractors(corpus, n_docs, spare, seed)

    return corpus
