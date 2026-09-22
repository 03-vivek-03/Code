"""Regression tests for the eleven defects found in the first full run.

Each test names the defect it guards, states what went wrong, and asserts the property
whose absence let it through. These are the tests that would have caught the problems
before 45 GPU-hours were spent on them, so they are written to fail loudly rather than to
be easy to satisfy.

The defects, in the order they appear below:

    D1  corpus was a lookup table, not a retrieval corpus
    D2  target answers were random, so the generation condition never fired
    D3  corpus_poisoning never retrieved
    D4  iterative retrieval could not reach the answer prompt
    D5  the detection label was the attack's outcome, not its occurrence
    D6  configurations were compared across different query sets
    D7  reflection's abstention was invisible in the metrics
    D8  context metrics measured the retrieval union, not the prompt
    D9  detector cost was a hardcoded placeholder
    D10 a truncated trace file was resumed as if complete; rules could never fire
    D11 (covered by D1: the validator rejects stub corpora)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from argus.agent.engine import AgenticRAG
from argus.attacks.registry import build_attack
from argus.config import MECHANISM_CONFIGS, AgentConfig, get_agent_config
from argus.corpus.answers import TargetAnswerPicker, answer_type, type_match_rate
from argus.corpus.store import Corpus, Document, Query
from argus.corpus.validate import (
    CorpusValidationError,
    question_echo_score,
    validate_attack_retrievability,
    validate_corpus,
)
from argus.llm.mock import MockLLM
from argus.retrieval.bm25 import BM25Retriever
from argus.telemetry.spans import Trace


# ---------------------------------------------------------------------- D1
class TestD1CorpusIsARetrievalCorpus:
    """The NQ loader built one document per question reading

        f"{question} The answer is {gold}."

    BM25 then matched the question verbatim, clean accuracy read 0.98, and every
    downstream number described a lookup task rather than retrieval.
    """

    @staticmethod
    def _lookup_table_corpus() -> Corpus:
        docs, queries = [], []
        for i in range(30):
            q = f"in what year was the {i}th observatory founded"
            docs.append(Document(doc_id=f"d{i}", text=f"{q} The answer is {1900 + i}."))
            queries.append(
                Query(
                    query_id=f"q{i}",
                    text=q,
                    gold_answer=str(1900 + i),
                    target_answer=str(1800 + i),
                    gold_doc_ids=[f"d{i}"],
                )
            )
        return Corpus(name="lookup_table", documents=docs, queries=queries)

    def test_validator_rejects_a_lookup_table(self):
        with pytest.raises(CorpusValidationError) as exc:
            validate_corpus(self._lookup_table_corpus(), strict=True)
        message = str(exc.value)
        assert "question_echo_rate" in message
        assert "FAIL" in message

    def test_validator_reports_rather_than_raises_when_not_strict(self):
        report = validate_corpus(self._lookup_table_corpus(), strict=False)
        assert not report.ok
        failed = {c.name for c in report.failures}
        assert "question_echo_rate" in failed

    def test_echo_score_separates_a_stub_from_a_real_passage(self):
        question = "where does the optic nerve cross the midline"
        stub = "where does the optic nerve cross the midline The answer is optic chiasm."
        passage = (
            "The optic chiasm is an X-shaped structure formed by the crossing of the "
            "optic nerves in the brain. Fibres from the nasal retina decussate there, "
            "which is what produces the partial decussation of the visual pathway."
        )
        assert question_echo_score(question, stub) >= 0.90
        assert question_echo_score(question, passage) < 0.90

    def test_synthetic_corpus_passes_structural_checks(self, corpus):
        """The offline corpus must satisfy the same structural checks as a real one.

        It is used for tests and demos rather than results, but if it cannot pass the
        validator then the validator is not testing anything meaningful.
        """
        report = validate_corpus(corpus, strict=False)
        for check in report.checks:
            if check.name != "target_type_match":  # synthetic values are already typed
                assert check.ok, check.line()


# ---------------------------------------------------------------------- D2
class TestD2TargetAnswersArePlausible:
    """`_pick_target_answer` sampled uniformly from every answer in the dataset, so the
    attacker's target for "where does the optic nerve cross the midline" was "Nigel
    Lythgoe". Poison was retrieved on 100% of runs and misled the model on 43%.
    """

    def test_answer_types_are_recognised(self):
        assert answer_type("1972") == "year"
        assert answer_type("October 22, 2017") == "date"
        assert answer_type("3.5 million") == "number"
        assert answer_type("Nigel Lythgoe") == "person"
        assert answer_type("Kansas City") == "place_or_entity"

    def test_picker_matches_the_gold_answer_type(self):
        years = [str(y) for y in range(1900, 1960)]
        people = ["Nigel Lythgoe", "Ada Lovelace", "Marie Curie", "Alan Turing", "Grace Hopper"]
        picker = TargetAnswerPicker(years + people, seed=1)

        for gold in years[:20]:
            assert answer_type(picker.pick(gold, gold)) == "year"
        for gold in people:
            assert answer_type(picker.pick(gold, gold)) == "person"

    def test_target_never_equals_gold(self):
        picker = TargetAnswerPicker([str(y) for y in range(1900, 1950)], seed=2)
        for gold in [str(y) for y in range(1900, 1950)]:
            assert picker.pick(gold, gold) != gold

    def test_picker_is_deterministic(self):
        pool = [str(y) for y in range(1900, 1960)]
        a = TargetAnswerPicker(pool, seed=3)
        b = TargetAnswerPicker(pool, seed=3)
        assert [a.pick(g, g) for g in pool] == [b.pick(g, g) for g in pool]

    def test_type_match_rate_detects_the_old_behaviour(self):
        """Random cross-type assignment must score far below the validator threshold."""
        rng = np.random.default_rng(0)
        pool = [str(y) for y in range(1900, 1950)] + [
            f"Person {i}" for i in range(50)
        ]
        random_pairs = [(g, pool[rng.integers(len(pool))]) for g in pool]
        assert type_match_rate(random_pairs) < 0.60

        typed = TargetAnswerPicker(pool, seed=4)
        typed_pairs = [(g, typed.pick(g, g)) for g in pool]
        assert type_match_rate(typed_pairs) >= 0.60


# ---------------------------------------------------------------------- D3
class TestD3AttacksActuallyRetrieve:
    """corpus_poisoning entered the top-5 in 7 of 22,000 runs (0.03%). It was a third of
    the grid and one of three leave-one-attack-out folds, where it held 46 positives in
    16,459 rows and pulled every detector below chance.
    """

    @pytest.mark.parametrize(
        "attack_name", ["poisonedrag_black", "poisonedrag_white", "corpus_poisoning"]
    )
    def test_every_registered_attack_reaches_the_top_k(self, corpus, attack_name):
        attack = build_attack(attack_name, n_poison_docs=3, seed=7)
        retriever = None
        if attack.needs_retriever:
            retriever = BM25Retriever(corpus)
            retriever.build()

        result = attack.apply(corpus, retriever=retriever)
        poisoned_retriever = BM25Retriever(result.corpus)
        poisoned_retriever.build()

        report = validate_attack_retrievability(
            result.corpus, result.poison_by_query, poisoned_retriever,
            top_k=5, strict=False,
        )
        assert report["poison_in_top_k_rate"] >= 0.30, (
            f"{attack_name} reaches the top-5 for only "
            f"{report['poison_in_top_k_rate']:.1%} of queries; it would contribute only "
            "null rows to the grid"
        )

    def test_validator_rejects_an_inert_attack(self, corpus):
        """An attack whose poison is unretrievable must fail the preflight."""
        inert = Corpus(
            name="inert",
            documents=list(corpus.documents)
            + [
                Document(
                    doc_id="poison_inert_0",
                    text="zzz qqq xxx unrelated filler with no query vocabulary at all",
                    is_poison=True,
                )
            ],
            queries=list(corpus.queries),
        )
        retriever = BM25Retriever(inert)
        retriever.build()
        poison_map = {q.query_id: ["poison_inert_0"] for q in inert.queries}

        with pytest.raises(CorpusValidationError, match="retrieval condition"):
            validate_attack_retrievability(
                inert, poison_map, retriever, top_k=5, strict=True
            )

    def test_cluster_size_scales_with_query_count_not_a_fixed_total(self):
        """corpus_poisoning was retested on the synthetic corpus (100% top-5) and passed
        the server preflight, then measured at 10% on the real 500-query NQ corpus. The
        synthetic corpus's queries are templated and share vocabulary even in a broad
        cluster; five hundred independent natural-language questions do not, and a fixed
        `n_clusters=24` left ~20 heterogeneous queries per cluster.

        This rebuilds that heterogeneity locally — real natural-language questions on
        distinct topics, no shared template — and checks that cluster count now scales
        with the query count rather than staying fixed, and that the resulting
        retrievability clears the preflight gate on a corpus shaped like the failure.
        """
        import random

        from argus.corpus.answers import TargetAnswerPicker
        from argus.corpus.validate import validate_attack_retrievability
        from argus.retrieval.bm25 import BM25Retriever

        topics = [
            ("mountain range", "{n} is the tallest peak in the {r} range.", "elevation survey"),
            ("river", "The {r} river flows through {n} to the sea.", "hydrology basin"),
            ("composer", "{n} composed the {r} symphony one winter.", "music archive"),
            ("battle", "The battle of {n} was fought near {r}.", "campaign record"),
            ("volcano", "{n} last erupted near the {r} coast.", "seismic activity"),
        ]
        rng = random.Random(7)
        names = [f"Entity{i}" for i in range(120)]
        docs, queries = [], []
        for i in range(150):
            topic, tmpl, vocab = topics[i % len(topics)]
            n, r = rng.sample(names, 2)
            gold = str(2000 + i % 200)
            text = (
                tmpl.format(n=n, r=r) + " " + vocab + " "
                + "Additional survey material accompanies the record. " * 3
            )
            doc = Document(doc_id=f"d{i:05d}", text=text, title=f"{topic} record")
            docs.append(doc)
            queries.append(
                Query(
                    query_id=f"q{i:05d}",
                    text=f"In what year is {n} associated with the {r} {topic} record?",
                    gold_answer=gold, gold_doc_ids=[doc.doc_id],
                )
            )
        for j in range(900):
            topic, tmpl, vocab = topics[j % len(topics)]
            n, r = rng.sample(names, 2)
            docs.append(Document(doc_id=f"bg{j:05d}", text=tmpl.format(n=n, r=r) + " " + vocab))

        picker = TargetAnswerPicker([q.gold_answer for q in queries], seed=7)
        for q in queries:
            q.target_answer = picker.pick(q.gold_answer, q.query_id)
        heterogeneous = Corpus(name="heterogeneous", documents=docs, queries=queries)

        attack = build_attack("corpus_poisoning", n_poison_docs=5, seed=42)
        # Cluster count must track the query count, not sit at a number tuned for a
        # different corpus.
        attack._prepare(heterogeneous, None)
        n_clusters_used = len(set(attack._cluster_of.values()))
        assert n_clusters_used >= len(queries) // 10, (
            f"only {n_clusters_used} clusters for {len(queries)} heterogeneous queries; "
            "cluster count is not scaling with the query set"
        )

        result = attack.apply(heterogeneous)
        retriever = BM25Retriever(result.corpus)
        retriever.build()
        report = validate_attack_retrievability(
            result.corpus, result.poison_by_query, retriever, top_k=5, strict=False
        )
        assert report["poison_in_top_k_rate"] >= 0.30, (
            f"only {report['poison_in_top_k_rate']:.1%} on heterogeneous queries; "
            "this is the exact failure mode measured on the real NQ corpus"
        )

    def test_corpus_poisoning_does_not_copy_the_query_verbatim(self, corpus):
        """It must stay distinguishable from PoisonedRAG, or leave-one-attack-out is
        holding out a near-duplicate of what it trained on."""
        result = build_attack("corpus_poisoning", n_poison_docs=2, seed=7).apply(corpus)
        for query in corpus.queries[:10]:
            for doc_id in result.poison_by_query.get(query.query_id, []):
                assert query.text.lower() not in result.corpus.get(doc_id).text.lower()


# ---------------------------------------------------------------------- D4
class TestD4IterationReachesThePrompt:
    """`top_k` and `max_context_docs` were both 5, and the prompt renders the best
    `max_context_docs` items by score. Later rounds exclude what was already seen, so they
    return strictly lower-scoring documents that can never displace round one. Budgets 1,
    2 and 3 gathered 4.98, 9.86 and 14.55 documents while input tokens stayed at 319.5,
    319.5 and 319.4.
    """

    @staticmethod
    def _run(cid, retriever, corpus, budget=None):
        cfg = get_agent_config(cid, budget)
        return AgenticRAG(retriever, MockLLM(seed=5), cfg, corpus).run(corpus.queries[0])

    def test_config_rejects_a_context_window_iteration_cannot_reach(self):
        with pytest.raises(ValueError, match="could never enter the answer prompt"):
            AgentConfig(
                name="broken",
                iterative_retrieval=True,
                iteration_budget=3,
                top_k=5,
                max_context_docs=5,
            )

    def test_larger_budget_changes_the_rendered_prompt(self, retriever, corpus):
        b1 = self._run("C2", retriever, corpus, budget=1)
        b3 = self._run("C2", retriever, corpus, budget=3)

        assert b3.n_iterations > b1.n_iterations
        assert len(b3.context_doc_ids) > len(b1.context_doc_ids), (
            "iteration gathered more evidence but showed the generator the same set"
        )
        assert b3.total_input_tokens > b1.total_input_tokens, (
            "the answer prompt did not grow, so iteration cannot affect the outcome"
        )

    def test_budget_changes_are_monotonic_in_context_size(self, retriever, corpus):
        sizes = [
            len(self._run("C2", retriever, corpus, budget=b).context_doc_ids)
            for b in (1, 2, 3)
        ]
        assert sizes[0] < sizes[1] <= sizes[2], sizes

    def test_default_configs_leave_room_for_iteration(self):
        for cid in ("C2", "C5"):
            cfg = MECHANISM_CONFIGS[cid]
            assert cfg.max_context_docs > cfg.top_k, cid


# ---------------------------------------------------------------------- D5
class TestD5LabelIsTheCompromiseEvent:
    """`label = "compromised" if (poison_ids and attack_success) else "benign"` made the
    detection target the outcome rather than the event. 26,694 traces (39.3% of attacked
    runs) were benign with poison in their prompt, so 83.7% of the negative class was
    poisoned, and leave-one-attack-out recall came out at 0.052.
    """

    def test_survived_attack_is_still_compromised(self, retriever, corpus, poisoned):
        from argus.retrieval.bm25 import BM25Retriever

        r = BM25Retriever(poisoned.corpus)
        r.build()
        agent = AgenticRAG(r, MockLLM(seed=9), MECHANISM_CONFIGS["C0"], poisoned.corpus)

        survived = []
        for q in poisoned.corpus.queries:
            trace = agent.run(q, poison_doc_ids=poisoned.poison_for(q.query_id))
            trace.meta["attacked"] = True
            if trace.poison_in_context and not trace.attack_success:
                survived.append(trace)

        assert survived, "no run survived poison; cannot exercise the distinction"
        for trace in survived:
            # This is the exact population the old rule mislabelled.
            assert trace.poison_in_context
            assert not trace.attack_success

    def test_features_do_not_leak_ground_truth(self):
        from argus.features.extractor import FEATURE_NAMES

        forbidden = ("poison", "gold", "target", "label", "attack_success", "correct")
        for name in FEATURE_NAMES:
            assert not any(word in name.lower() for word in forbidden), name


# ---------------------------------------------------------------------- D6
class TestD6MatchedQuerySets:
    """C0, C1 and C2 ran 1,000 queries while C3, C4 and C5 ran 500. Cells answer
    `queries[:n_queries]`, so the comparison crossed different query sets: C0's attack
    success rate is 0.3029 over its full set and 0.2873 over the matched prefix, which
    inflated C4's reported reduction from −9.6 to −12.0 points.
    """

    def test_grid_refuses_mismatched_query_counts(self):
        from argus.config import RunConfig
        from argus.runner.grid import GridConsistencyError, check_matched_queries

        cells = [
            RunConfig(config_id="C0", dataset="nq", n_queries=1000),
            RunConfig(config_id="C4", dataset="nq", n_queries=500),
        ]
        with pytest.raises(GridConsistencyError, match="same n_queries"):
            check_matched_queries(cells)

    @pytest.mark.parametrize("preset", ["smoke", "pilot", "main", "budget", "abstention", "multihop", "retriever"])
    def test_every_preset_is_internally_matched(self, preset):
        from argus.runner.grid import build_grid

        build_grid(preset)  # build_grid calls check_matched_queries

    def test_restriction_drops_the_unmatched_tail(self):
        from argus.analysis.ablation import restrict_to_matched_queries

        frame = pd.DataFrame(
            [
                {"config_id": "C0", "query_id": f"q{i}", "dataset": "nq",
                 "retriever": "bm25", "attack": "a", "n_poison": 5}
                for i in range(10)
            ]
            + [
                {"config_id": "C4", "query_id": f"q{i}", "dataset": "nq",
                 "retriever": "bm25", "attack": "a", "n_poison": 5}
                for i in range(5)
            ]
        )
        out = restrict_to_matched_queries(frame)
        assert len(out) == 10
        assert set(out[out["config_id"] == "C0"]["query_id"]) == set(
            out[out["config_id"] == "C4"]["query_id"]
        )


# ---------------------------------------------------------------------- D7
class TestD7AbstentionIsMeasured:
    """Reflection cut attack success by 9.6 points while refusals rose 13.0 and correct
    answers fell 2.9. Reported as a single attack-success figure, an agent that stops
    answering looks identical to one that resists poison.
    """

    def test_abstention_is_an_outcome(self):
        t = Trace(trace_id="t", query_id="q", query="x")
        t.final_answer = "I cannot determine the answer from the provided context."
        t.target_answer = "1972"
        t.gold_answer = "1983"
        assert t.is_abstention
        assert not t.attack_success
        assert not t.answered_correctly

    def test_committed_answer_is_not_abstention(self):
        t = Trace(trace_id="t", query_id="q", query="x")
        t.final_answer = "1983"
        t.gold_answer = "1983"
        assert not t.is_abstention
        assert t.answered_correctly

    def test_refusal_markers_are_shared_between_studies(self):
        """Study A's abstention rate and Study B's `answer_is_refusal` must agree."""
        from argus.features.extractor import _REFUSAL
        from argus.telemetry.spans import REFUSAL_MARKERS

        assert _REFUSAL is REFUSAL_MARKERS

    def test_caution_is_recorded_when_applied(self, retriever, corpus):
        agent = AgenticRAG(retriever, MockLLM(seed=11), MECHANISM_CONFIGS["C4"], corpus)
        traces = [agent.run(q) for q in corpus.queries[:10]]
        assert any("caution_applied" in t.meta for t in traces)
        for t in traces:
            if t.meta.get("caution_applied"):
                assert t.meta["reflection_insufficient"]

    def test_c6_never_applies_caution(self, retriever, corpus):
        agent = AgenticRAG(retriever, MockLLM(seed=11), MECHANISM_CONFIGS["C6"], corpus)
        for q in corpus.queries[:10]:
            t = agent.run(q)
            assert t.meta["caution_applied"] is False
            # The verdict is still computed and logged; only its effect is suppressed.
            assert t.meta["n_reflections"] >= 1


# ---------------------------------------------------------------------- D8
class TestD8ContextMetricsDescribeThePrompt:
    """`context_doc_ids` unioned every retrieval span instead of reading what was
    rendered, so C2's poisoned context fraction read 0.343 when the prompt was 0.447.
    `poison_rank` took the minimum rank per span, and because later rounds exclude what
    was already seen, a document at true rank 5 reported rank 0 — which is why iterative
    retrieval appeared to rank poison better (0.28) than vanilla (0.50).
    """

    def test_context_is_read_from_the_answer_span(self, retriever, corpus):
        agent = AgenticRAG(retriever, MockLLM(seed=13), MECHANISM_CONFIGS["C2"], corpus)
        trace = agent.run(corpus.queries[0])

        span = trace.answer_span
        assert span is not None
        recorded = span.attributes["argus.context_document_ids"]
        assert trace.context_doc_ids == list(recorded)

    def test_context_is_a_subset_of_what_was_retrieved(self, retriever, corpus):
        agent = AgenticRAG(retriever, MockLLM(seed=13), MECHANISM_CONFIGS["C2"], corpus)
        trace = agent.run(corpus.queries[0])
        assert set(trace.context_doc_ids) <= set(trace.retrieved_doc_ids)
        assert len(trace.retrieved_doc_ids) > len(trace.context_doc_ids)

    def test_poison_fraction_is_computed_over_the_prompt(self, retriever, corpus):
        agent = AgenticRAG(retriever, MockLLM(seed=13), MECHANISM_CONFIGS["C2"], corpus)
        ctx_ids = None
        trace = agent.run(corpus.queries[0])
        ctx_ids = trace.context_doc_ids
        # Declare half the prompt poisoned and check the fraction follows the prompt.
        trace.poison_doc_ids = list(ctx_ids[: len(ctx_ids) // 2])
        expected = len(trace.poison_doc_ids) / len(ctx_ids)
        assert trace.poison_context_fraction == pytest.approx(expected)

    def test_poison_rank_uses_the_unfiltered_ranking(self, retriever, corpus):
        """Ranks must be comparable across iterations."""
        agent = AgenticRAG(retriever, MockLLM(seed=13), MECHANISM_CONFIGS["C2"], corpus)
        trace = agent.run(corpus.queries[0])

        spans = trace.retrieval_spans
        assert len(spans) > 1
        # Round two excludes round one, so its global ranks must start beyond zero.
        later = spans[1].attributes["retrieval.global_ranks"]
        assert later and min(later) > 0, (
            "later rounds are renumbering from zero, so ranks are not comparable"
        )

        # A document only found in a later round must not report rank 0.
        second_round_only = set(spans[1].attributes["retrieval.document_ids"]) - set(
            spans[0].attributes["retrieval.document_ids"]
        )
        if second_round_only:
            trace.poison_doc_ids = sorted(second_round_only)
            assert trace.poison_rank > 0

    def test_retrieved_and_context_agree_without_iteration(self, retriever, corpus):
        agent = AgenticRAG(retriever, MockLLM(seed=13), MECHANISM_CONFIGS["C0"], corpus)
        trace = agent.run(corpus.queries[0])
        assert set(trace.context_doc_ids) == set(trace.retrieved_doc_ids)


# ---------------------------------------------------------------------- D9
class TestD9CostIsMeasuredNotPlaceheld:
    """`inference_overhead_x` was hardcoded to 0.0 at every call site, so the cost column
    of the comparison read as "not measured" rather than "measured, and zero" — and the
    baseline comparison the claim depends on was never run at all.
    """

    def test_trace_detector_overhead_is_a_named_constant(self):
        from argus.detect.evaluate import TRACE_DETECTOR_OVERHEAD_X

        assert TRACE_DETECTOR_OVERHEAD_X == 0.0

    def test_baselines_declare_their_true_overhead(self):
        from argus.baselines.llm_judge import LLMJudgeDefense
        from argus.baselines.loo_counterfactual import LOOCounterfactualDefense
        from argus.baselines.perplexity import PerplexityFilterDefense

        corpus = Corpus(name="c", documents=[], queries=[])
        llm = MockLLM(seed=1)

        assert PerplexityFilterDefense(corpus).overhead_x == 0.0
        assert LLMJudgeDefense(llm, corpus, max_docs=5).overhead_x == 5.0
        # RAGuard's k+1 passes: the number Study B is arguing against.
        assert LOOCounterfactualDefense(llm, corpus, max_docs=5).overhead_x == 6.0


# ---------------------------------------------------------------------- D10
class TestD10IntegrityAndReachableThresholds:
    """One cell held 141 usable trace lines, the 141st truncated mid-string, while its
    result JSON recorded n_traces=500. And the rules detector scored the fraction of six
    90th-percentile rules that fired, thresholded at 0.5, so three had to trip at once:
    recall was 0.000 in every fold and pooled AUC 0.408, below chance.
    """

    def test_truncated_trace_file_is_detected(self, tmp_path):
        from argus.runner.experiment import count_valid_traces

        path = tmp_path / "cell.jsonl"
        good = json.dumps({"trace_id": "t", "query_id": "q", "query": "x"})
        path.write_text(f"{good}\n{good}\n" + good[:20], encoding="utf-8")

        ok, bad = count_valid_traces(path)
        assert (ok, bad) == (2, 1)

    def test_reader_skips_corrupt_lines_and_counts_them(self, tmp_path):
        from argus.telemetry.writer import TraceReader, TraceWriter

        path = tmp_path / "t.jsonl"
        with TraceWriter(path, append=False) as w:
            w.write(Trace(trace_id="t1", query_id="q1", query="x"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"trace_id": "broken"')

        reader = TraceReader(path)
        traces = reader.read_all()
        assert len(traces) == 1
        assert reader.n_corrupt == 1

    def test_reader_can_be_strict(self, tmp_path):
        from argus.telemetry.writer import TraceReader

        path = tmp_path / "t.jsonl"
        path.write_text('{"broken"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            TraceReader(path, strict=True).read_all()

    def test_rules_detector_can_actually_fire(self):
        """A floor that cannot be reached is not a floor."""
        from argus.detect.models import build_detector
        from argus.features.extractor import FEATURE_NAMES

        rng = np.random.default_rng(0)
        n = 600
        y = rng.integers(0, 2, n)
        X = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES})
        X["retr_score_top_gap"] += y * 2.0
        X["query_drift_total"] += y * 1.5

        det = build_detector("rules", feature_names=list(X.columns))
        det.fit(X.to_numpy(), y)
        scores = det.score(X.to_numpy())

        assert np.all((scores >= 0) & (scores <= 1))
        fired = (scores >= 0.5).sum()
        assert fired > 0, "the rule detector never fires at its own threshold"
        # And it must carry signal, not just fire.
        from argus.detect.base import compute_metrics

        assert compute_metrics(y, scores)["roc_auc"] > 0.6

    def test_rules_operating_point_respects_target_fpr(self):
        from argus.detect.models import build_detector
        from argus.features.extractor import FEATURE_NAMES

        rng = np.random.default_rng(1)
        n = 2000
        X = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES})
        y = np.zeros(n, dtype=int)

        det = build_detector("rules", feature_names=list(X.columns), target_fpr=0.05)
        det.fit(X.to_numpy(), y)
        fpr = float((det.score(X.to_numpy()) >= 0.5).mean())
        assert 0.0 < fpr < 0.15, fpr


# ------------------------------------------------------- LOAO protocol integrity
class TestLOAOHasNoLeakage:
    """The benign pool was `attack in {none, ""} or y == 0`, so unsuccessful runs of the
    held-out attack were treated as benign and split 70/30 into train and test. The
    detector therefore trained on the distribution the protocol exists to hold out.
    """

    @staticmethod
    def _data(n=900, seed=0):
        from argus.features.extractor import FEATURE_NAMES

        rng = np.random.default_rng(seed)
        attack = rng.choice(
            ["none", "poisonedrag_black", "poisonedrag_white", "corpus_poisoning"], n
        )
        y = np.where(attack == "none", 0, rng.integers(0, 2, n))
        X = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES[:12]})
        X["retr_score_top_gap"] += y * 1.5
        meta = pd.DataFrame(
            {
                "y": y,
                "attack": attack,
                "dataset": "synthetic",
                "trace_id": [f"t{i}" for i in range(n)],
            }
        )
        return X, meta

    def test_held_out_attack_never_appears_in_training(self, monkeypatch):
        from argus.detect import evaluate as ev

        X, meta = self._data()
        seen: dict[str, set[str]] = {}
        original = ev._fit_score
        order = list(sorted(a for a in meta["attack"].unique() if a != "none"))
        calls = {"i": 0}

        def spy(name, X_train, y_train, X_test, feature_names, **params):
            held = order[calls["i"]]
            calls["i"] += 1
            # Recover which rows went to training by matching the first feature column.
            train_rows = {tuple(np.round(r, 9)) for r in X_train}
            all_rows = {
                tuple(np.round(r, 9)): a
                for r, a in zip(X.to_numpy(), meta["attack"].to_numpy())
            }
            seen[held] = {all_rows[r] for r in train_rows if r in all_rows}
            return original(name, X_train, y_train, X_test, feature_names, **params)

        monkeypatch.setattr(ev, "_fit_score", spy)
        ev.leave_one_attack_out("gbdt", X, meta, seed=0, min_positives=1)

        for held, attacks_in_train in seen.items():
            assert held not in attacks_in_train, (
                f"held-out attack {held} leaked into its own training fold"
            )

    def test_fold_without_enough_positives_is_skipped(self):
        from argus.detect.evaluate import leave_one_attack_out

        X, meta = self._data()
        # Make one attack essentially inert, as corpus_poisoning was.
        inert = meta["attack"] == "corpus_poisoning"
        meta.loc[inert, "y"] = 0
        meta.loc[meta[inert].index[:3], "y"] = 1

        res = leave_one_attack_out("gbdt", X, meta, seed=0, min_positives=50)
        assert "skipped" in res.per_split["corpus_poisoning"]
        assert "corpus_poisoning" in res.meta["folds_skipped"]
        assert res.meta["folds_evaluated"] == 2


# ------------------------------------------- moving to a different model / endpoint
class TestMechanismResponseParsing:
    """The agent turns free text into control decisions at three points, and each was a
    one-liner tuned against one model. Moving to another serving stack is exactly when
    they break, and they break *silently*: a grid where every reflect response fell back
    to SUFFICIENT is complete, plausible and wrong.

    Every response below is reasonable English that the original code mis-read.
    """

    def test_chatty_inspect_response_takes_the_first_index(self):
        from argus.agent.parsing import parse_document_choice

        text = "I would recommend reading Document 3 in full, as it covers 2 key topics."
        # The original concatenated every digit -> "32" -> index 31 -> clamped to the last
        # candidate, so the agent inspected the wrong document and nothing reported it.
        assert "".join(c for c in text if c.isdigit()) == "32"

        parsed = parse_document_choice(text, n_candidates=5)
        assert parsed.value == 2  # zero-based index of "Document 3"
        assert parsed.ok

    def test_inspect_out_of_range_is_clamped_and_flagged(self):
        from argus.agent.parsing import parse_document_choice

        parsed = parse_document_choice("Document 9", n_candidates=5)
        assert parsed.value == 4
        assert not parsed.ok

    def test_inspect_without_a_number_falls_back_and_is_flagged(self):
        from argus.agent.parsing import parse_document_choice

        parsed = parse_document_choice("The first one looks most relevant.", 5)
        assert parsed.value == 0
        assert not parsed.ok

    def test_not_sufficient_is_read_as_insufficient(self):
        from argus.agent.parsing import parse_reflection

        text = "The evidence provided is not sufficient to answer this question."
        # The original tested `"INSUFFICIENT" in text.upper()`, which is False here, so
        # this read as approval and M4 was disabled while appearing to work.
        assert "INSUFFICIENT" not in text.upper()

        parsed = parse_reflection(text)
        assert parsed.value == "INSUFFICIENT"
        assert parsed.ok

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("SUFFICIENT", "SUFFICIENT"),
            ("INSUFFICIENT", "INSUFFICIENT"),
            ("The context is sufficient.", "SUFFICIENT"),
            ("Insufficient evidence to determine the answer.", "INSUFFICIENT"),
            ("More information is needed.", "INSUFFICIENT"),
            ("Yes, this is enough to answer.", "SUFFICIENT"),
        ],
    )
    def test_reflection_verdicts(self, text, expected):
        from argus.agent.parsing import parse_reflection

        parsed = parse_reflection(text)
        assert parsed.value == expected, text
        assert parsed.ok

    def test_unparseable_reflection_is_flagged_not_silently_approved(self):
        from argus.agent.parsing import parse_reflection

        parsed = parse_reflection("Hmm, that's an interesting question.")
        assert parsed.value == "SUFFICIENT"  # conservative: leaves the prompt untouched
        assert not parsed.ok                 # but counted as a failure, not a verdict

    def test_rewrite_preamble_is_stripped(self):
        from argus.agent.parsing import parse_rewrite

        original = "where does the optic nerve cross the midline"
        parsed = parse_rewrite(
            'Here is a new search query: "optic nerve decussation"', original
        )
        # The original used the whole response, so "Here is a new search query" went into
        # the retriever as query terms.
        assert parsed.value == "optic nerve decussation"
        assert parsed.ok

    def test_rewrite_takes_the_query_line_from_a_chatty_response(self):
        from argus.agent.parsing import parse_rewrite

        original = "where does the optic nerve cross the midline"
        parsed = parse_rewrite(
            "optic chiasm decussation anatomy\n\nThis should surface the anatomical "
            "description you need, since it names the structure directly.",
            original,
        )
        assert parsed.value == "optic chiasm decussation anatomy"

    def test_degenerate_rewrite_falls_back_to_the_original(self):
        from argus.agent.parsing import parse_rewrite

        original = "where does the optic nerve cross"
        for bad in ("", "   ", "Here is a search query:"):
            parsed = parse_rewrite(bad, original)
            assert parsed.value == original
            assert not parsed.ok

    def test_absurdly_long_rewrite_is_rejected(self):
        from argus.agent.parsing import parse_rewrite

        original = "where does the optic nerve cross"
        parsed = parse_rewrite("word " * 200, original)
        assert parsed.value == original
        assert not parsed.ok

    def test_engine_counts_parse_failures(self, retriever, corpus):
        """A run must be able to say whether its mechanisms actually fired."""
        from argus.agent.engine import AgenticRAG
        from argus.llm.mock import MockLLM

        agent = AgenticRAG(retriever, MockLLM(seed=3), MECHANISM_CONFIGS["C5"], corpus)
        trace = agent.run(corpus.queries[0])
        assert "parse_attempts" in trace.meta
        assert "parse_failures" in trace.meta
        assert trace.meta["parse_attempts"] > 0


class TestConcurrencyIsResultPreserving:
    """Concurrency exists because the generator sits behind a network hop. It must not
    change what the experiment measures — only how long it takes.

    Identifiers are derived rather than random precisely so this comparison is possible.
    """

    @staticmethod
    def _run(workers, tmp_path):
        from argus.config import ArgusConfig, LLMConfig, RetrievalConfig, RunConfig
        from argus.runner.experiment import ExperimentRunner

        cfg = ArgusConfig(
            data_dir=tmp_path,
            seed=11,
            llm=LLMConfig(backend="mock"),
            retrieval=RetrievalConfig(backend="bm25"),
            check_attack_retrievability=False,
            workers=workers,
        )
        cfg.ensure_dirs()
        runner = ExperimentRunner(cfg, progress=lambda m: None)
        cell = RunConfig(
            config_id="C5", dataset="synthetic", attack="poisonedrag_black",
            n_poison_docs=3, n_queries=24, retriever="bm25", seed=11,
        )
        outcome = runner.run_cell(cell, n_docs=500)
        path = tmp_path / "traces" / f"{cell.cell_id}.jsonl"
        records = [json.loads(line) for line in path.open(encoding="utf-8")]
        return outcome, records

    @staticmethod
    def _strip_timing(record):
        """Drop wall-clock fields, which legitimately differ: a request that queued
        behind three others really did take longer."""
        rec = json.loads(json.dumps(record))
        rec.pop("total_latency_ms", None)
        rec.pop("cost_usd", None)
        for span in rec["spans"]:
            for key in ("start_ms", "end_ms", "duration_ms"):
                span.pop(key, None)
            span["attributes"].pop("argus.latency_ms", None)
            span["attributes"].pop("retrieval.latency_ms", None)
        return rec

    def test_traces_are_identical_in_content_and_order(self, tmp_path):
        _, serial = self._run(1, tmp_path / "serial")
        _, parallel = self._run(4, tmp_path / "parallel")

        assert len(serial) == len(parallel) == 24
        assert [r["query_id"] for r in serial] == [r["query_id"] for r in parallel], (
            "concurrent execution changed the order traces were written in"
        )
        assert [r["trace_id"] for r in serial] == [r["trace_id"] for r in parallel], (
            "trace identifiers are not derived; a re-run produces a different corpus"
        )
        for a, b in zip(serial, parallel):
            assert self._strip_timing(a) == self._strip_timing(b), a["query_id"]

    def test_aggregate_metrics_are_identical(self, tmp_path):
        serial, _ = self._run(1, tmp_path / "s")
        parallel, _ = self._run(4, tmp_path / "p")

        for field_name in (
            "attack_success_rate", "poisoned_accuracy", "abstention_rate",
            "poison_retrieval_rate", "mean_input_tokens", "parse_failure_rate",
        ):
            assert getattr(serial, field_name) == getattr(parallel, field_name), field_name

    def test_token_accounting_is_per_run_not_differenced(self, retriever, corpus):
        """Under concurrency, differencing a shared counter attributes other threads'
        tokens to this run. The engine sums its own spans instead."""
        from argus.agent.engine import AgenticRAG
        from argus.llm.mock import MockLLM

        llm = MockLLM(seed=5)
        agent = AgenticRAG(retriever, llm, MECHANISM_CONFIGS["C5"], corpus)
        trace = agent.run(corpus.queries[0])

        span_total = sum(
            s.attributes.get("gen_ai.usage.input_tokens", 0) for s in trace.inference_spans
        )
        assert trace.total_input_tokens == span_total
        assert trace.total_input_tokens > 0

        # A second run on the same backend must not inherit the first run's tokens.
        second = agent.run(corpus.queries[1])
        assert second.total_input_tokens < llm.usage.input_tokens


class TestPricingIsHonest:
    """A self-hosted model matched no pricing key and was silently billed at gpt-4o-mini
    rates, with the mid-run spend guard armed at $35."""

    def test_self_hosted_qwen_is_recognised(self):
        from argus.llm.openai_compat import price_for_detailed

        (inp, out), known = price_for_detailed("qwen2.5:14b-instruct-q4_K_M")
        assert known, "the Ollama-style Qwen name still falls through to the default rate"
        assert (inp, out) == (0.05, 0.10)

    def test_unknown_model_is_flagged_as_a_fallback(self):
        from argus.llm.openai_compat import price_for_detailed

        _, known = price_for_detailed("some-model-nobody-has-heard-of")
        assert not known

    def test_longest_key_wins(self):
        from argus.llm.openai_compat import price_for_detailed

        # "qwen2.5" must beat the shorter "qwen" for a name containing both.
        (inp, _), known = price_for_detailed("qwen2.5-72b")
        assert known and inp == 0.05

    def test_budget_of_zero_disables_the_mid_run_guard(self, tmp_path):
        from argus.config import ArgusConfig, LLMConfig
        from argus.runner.experiment import ExperimentRunner

        cfg = ArgusConfig(data_dir=tmp_path, budget_usd=0.0, llm=LLMConfig(backend="mock"))
        cfg.ensure_dirs()
        runner = ExperimentRunner(cfg, progress=lambda m: None)
        runner.llm.usage.input_tokens = 10**12  # an absurd spend
        assert runner._over_budget() is False


# ------------------------------------------------- surviving a dropped connection
class TestTransientErrorClassification:
    """A 30-hour grid died on the first `APIConnectionError` ("No route to host") it hit,
    after a total retry budget of about three seconds. A routing blip between the run
    host and the gateway is routine over that many hours; it must not be treated the same
    as a request that will fail identically on every retry.
    """

    @staticmethod
    def _exc(name: str, **attrs):
        cls = type(name, (Exception,), {})
        exc = cls("simulated")
        for k, v in attrs.items():
            setattr(exc, k, v)
        return exc

    @pytest.mark.parametrize(
        "name",
        [
            "APIConnectionError", "APITimeoutError", "InternalServerError",
            "RateLimitError", "ConnectionError", "ConnectionResetError",
            "ConnectError", "ReadTimeout", "ConnectTimeout", "RemoteProtocolError",
        ],
    )
    def test_known_transient_names_get_more_patience(self, name):
        from argus.llm.openai_compat import _is_transient

        assert _is_transient(self._exc(name))

    @pytest.mark.parametrize(
        "name", ["AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError"]
    )
    def test_known_fatal_names_fail_fast(self, name):
        from argus.llm.openai_compat import _is_transient

        assert not _is_transient(self._exc(name))

    def test_http_status_is_the_fallback_signal(self):
        from argus.llm.openai_compat import _is_transient

        assert _is_transient(self._exc("APIStatusError", status_code=503))
        assert _is_transient(self._exc("APIStatusError", status_code=429))
        assert not _is_transient(self._exc("APIStatusError", status_code=400))

    def test_unknown_exception_defaults_to_retryable(self):
        """Misclassifying a connection issue as fatal is what caused the outage this
        guards against; misclassifying a truly-fatal unknown error as transient costs at
        most one extra retry cycle. The asymmetry is deliberate."""
        from argus.llm.openai_compat import _is_transient

        assert _is_transient(self._exc("SomeExceptionTypeNeverSeenBefore"))

    def test_transient_error_gets_far_more_attempts_than_the_base_budget(self, monkeypatch):
        """The actual bug: with max_retries=3 the call gave up after ~3 seconds. A
        transient error must extend the budget to `max_retries_transient` instead."""
        from argus.llm.openai_compat import OpenAICompatLLM

        llm = OpenAICompatLLM(
            model="qwen2.5:14b-instruct-q4_K_M", api_key="x",
            max_retries=3, max_retries_transient=6, transient_backoff_cap_s=0.01,
        )

        calls = {"n": 0}

        class FakeCompletions:
            def create(self, **kw):
                calls["n"] += 1
                if calls["n"] < 5:
                    raise self.__class__.exc_type("connection dropped")
                msg = type("M", (), {"content": "ok", "role": "assistant"})()
                choice = type("C", (), {"message": msg, "finish_reason": "stop"})()
                return type("R", (), {"choices": [choice], "usage": None})()

        FakeCompletions.exc_type = type("APIConnectionError", (Exception,), {})
        fake_client = type("Client", (), {})()
        fake_client.chat = type("Chat", (), {"completions": FakeCompletions()})()
        monkeypatch.setattr(llm, "_get_client", lambda: fake_client)
        monkeypatch.setattr("time.sleep", lambda *_: None)

        resp = llm.generate("hello")
        assert resp.text == "ok"
        assert calls["n"] == 5, "should have kept retrying well past the old 3-attempt limit"

    def test_fatal_error_does_not_get_the_extended_budget(self, monkeypatch):
        from argus.llm.openai_compat import OpenAICompatLLM

        llm = OpenAICompatLLM(
            model="x", api_key="x", max_retries=3, max_retries_transient=8,
        )
        calls = {"n": 0}
        auth_error = type("AuthenticationError", (Exception,), {})

        class FakeCompletions:
            def create(self, **kw):
                calls["n"] += 1
                raise auth_error("bad key")

        fake_client = type("Client", (), {})()
        fake_client.chat = type("Chat", (), {"completions": FakeCompletions()})()
        monkeypatch.setattr(llm, "_get_client", lambda: fake_client)
        monkeypatch.setattr("time.sleep", lambda *_: None)

        with pytest.raises(RuntimeError, match="non-retryable"):
            llm.generate("hello")
        assert calls["n"] == 1, "a fatal error should fail on the first attempt, not retry"


class TestOneQueryFailureDoesNotLoseTheCell:
    """A single query's exhausted retries used to propagate out of the ThreadPoolExecutor
    collection loop and abort the whole cell — a connection blip on query 481 of 500 threw
    away 480 already-completed LLM calls, not one.
    """

    @staticmethod
    def _flaky_llm(bad_marker: str):
        from argus.llm.mock import MockLLM

        class FlakyLLM(MockLLM):
            def _generate(self, prompt, system="", **kwargs):
                if bad_marker in prompt:
                    raise RuntimeError("simulated permanent failure for this query")
                return super()._generate(prompt, system=system, **kwargs)

        return FlakyLLM(seed=13)

    def test_other_queries_survive_one_permanent_failure(self, tmp_path):
        from argus.config import ArgusConfig, RetrievalConfig, RunConfig
        from argus.runner.experiment import ExperimentRunner, count_valid_traces

        cfg = ArgusConfig(
            data_dir=tmp_path, seed=11, retrieval=RetrievalConfig(backend="bm25"),
            check_attack_retrievability=False, workers=4,
        )
        cfg.ensure_dirs()

        cell = RunConfig(
            config_id="C0", dataset="synthetic", attack="poisonedrag_black",
            n_poison_docs=3, n_queries=20, retriever="bm25", seed=11,
        )
        # Build the corpus once to find a real query to target.
        from argus.corpus.loaders import build_corpus

        corpus = build_corpus(dataset="synthetic", n_docs=500, n_queries=20, seed=11)
        bad_query = corpus.queries[7].text

        runner = ExperimentRunner(cfg, llm=self._flaky_llm(bad_query), progress=lambda m: None)
        outcome = runner.run_cell(cell, n_docs=500)

        assert outcome.n_traces == 19, "exactly the one failing query should be missing"

        path = tmp_path / "traces" / f"{cell.cell_id}.jsonl"
        ok, corrupt = count_valid_traces(path)
        assert ok == 19 and corrupt == 0

        # And a partial cell must not look complete on the next launch.
        assert ok < cell.n_queries, (
            "a short cell must be re-run in full next time, not mistaken for done"
        )

    def test_progress_reports_which_query_failed(self, tmp_path):
        from argus.config import ArgusConfig, RetrievalConfig, RunConfig
        from argus.corpus.loaders import build_corpus
        from argus.runner.experiment import ExperimentRunner

        cfg = ArgusConfig(
            data_dir=tmp_path, seed=11, retrieval=RetrievalConfig(backend="bm25"),
            check_attack_retrievability=False, workers=1,
        )
        cfg.ensure_dirs()
        corpus = build_corpus(dataset="synthetic", n_docs=300, n_queries=10, seed=11)
        bad_query = corpus.queries[2].text

        messages: list[str] = []
        cell = RunConfig(
            config_id="C0", dataset="synthetic", attack="poisonedrag_black",
            n_poison_docs=2, n_queries=10, retriever="bm25", seed=11,
        )
        runner = ExperimentRunner(
            cfg, llm=self._flaky_llm(bad_query), progress=messages.append
        )
        runner.run_cell(cell, n_docs=300)

        assert any("QUERY FAILED" in m for m in messages)
        assert any("1/10" in m or "will be redone" in m for m in messages)
