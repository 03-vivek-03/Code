"""Agent engine, mechanism isolation and telemetry tests.

The most important tests in this file are the mechanism-isolation ones. If enabling M1
also changes what M3 does, the ablation attributes effects to the wrong mechanism and
every result in Study A is wrong.
"""

from __future__ import annotations

from argus.agent.engine import AgenticRAG
from argus.config import MECHANISM_CONFIGS, get_agent_config
from argus.llm.mock import MockLLM
from argus.telemetry.spans import SpanKind, Trace


class TestMechanismIsolation:
    def _run(self, cid, retriever, llm, corpus, budget=None):
        cfg = get_agent_config(cid, budget)
        return AgenticRAG(retriever, llm, cfg, corpus).run(corpus.queries[0])

    def test_c0_does_one_retrieval_and_one_call(self, retriever, llm, corpus):
        t = self._run("C0", retriever, llm, corpus)
        assert t.n_iterations == 1
        assert len(t.inference_spans) == 1
        assert t.inference_spans[0].attributes["argus.task"] == "answer"

    def test_c1_rewrites_and_does_not_iterate(self, retriever, llm, corpus):
        t = self._run("C1", retriever, llm, corpus)
        tasks = [s.attributes["argus.task"] for s in t.inference_spans]
        assert "rewrite" in tasks
        assert "reflect" not in tasks
        assert "inspect" not in tasks
        assert t.n_iterations == 1

    def test_c2_iterates_without_other_mechanisms(self, retriever, llm, corpus):
        t = self._run("C2", retriever, llm, corpus)
        assert t.n_iterations == 3
        tasks = [s.attributes["argus.task"] for s in t.inference_spans]
        assert tasks == ["answer"]

    def test_c3_inspects_and_does_not_iterate(self, retriever, llm, corpus):
        t = self._run("C3", retriever, llm, corpus)
        tasks = [s.attributes["argus.task"] for s in t.inference_spans]
        assert "inspect" in tasks
        assert "rewrite" not in tasks
        assert t.n_iterations == 1
        assert len(t.tool_spans) >= 1

    def test_c4_reflects_and_does_not_iterate(self, retriever, llm, corpus):
        t = self._run("C4", retriever, llm, corpus)
        tasks = [s.attributes["argus.task"] for s in t.inference_spans]
        assert "reflect" in tasks
        assert "rewrite" not in tasks
        assert t.n_iterations == 1

    def test_c5_activates_everything(self, retriever, llm, corpus):
        t = self._run("C5", retriever, llm, corpus)
        tasks = {s.attributes["argus.task"] for s in t.inference_spans}
        assert {"rewrite", "inspect", "reflect", "answer"} <= tasks

    def test_iteration_budget_bounds_iterations(self, retriever, llm, corpus):
        for budget in (1, 2, 3):
            t = self._run("C2", retriever, llm, corpus, budget=budget)
            assert t.n_iterations == budget

    def test_iteration_gathers_new_documents(self, retriever, llm, corpus):
        """Iteration must surface new evidence, not repeat round one."""
        t = self._run("C2", retriever, llm, corpus, budget=3)
        per_iter = [
            set(s.attributes["retrieval.document_ids"]) for s in t.retrieval_spans
        ]
        for a, b in zip(per_iter, per_iter[1:]):
            assert not (a & b), "iteration returned documents already seen"

    def test_more_mechanisms_cost_more_tokens(self, retriever, llm, corpus):
        c0 = self._run("C0", retriever, llm, corpus)
        c5 = self._run("C5", retriever, llm, corpus)
        assert c5.total_input_tokens > c0.total_input_tokens


class TestTelemetry:
    def test_every_span_has_required_fields(self, agent, corpus):
        t = agent.run(corpus.queries[0])
        for s in t.spans:
            assert s.span_id and s.trace_id and s.name
            assert isinstance(s.kind, SpanKind)
            assert s.end_ms >= s.start_ms

    def test_retrieval_span_uses_otel_attribute_names(self, agent, corpus):
        t = agent.run(corpus.queries[0])
        attrs = t.retrieval_spans[0].attributes
        for key in ("retrieval.query", "retrieval.document_ids", "retrieval.scores"):
            assert key in attrs

    def test_inference_span_uses_genai_attribute_names(self, agent, corpus):
        t = agent.run(corpus.queries[0])
        attrs = t.inference_spans[0].attributes
        for key in (
            "gen_ai.operation.name",
            "gen_ai.request.model",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
        ):
            assert key in attrs

    def test_scores_align_with_document_ids(self, agent, corpus):
        t = agent.run(corpus.queries[0])
        for s in t.retrieval_spans:
            assert len(s.attributes["retrieval.document_ids"]) == len(
                s.attributes["retrieval.scores"]
            )

    def test_trace_roundtrip(self, agent, corpus):
        t = agent.run(corpus.queries[0])
        back = Trace.from_dict(t.to_dict())
        assert back.trace_id == t.trace_id
        assert len(back.spans) == len(t.spans)
        assert back.attack_success == t.attack_success

    def test_writer_reader_roundtrip(self, traces, tmp_path):
        from argus.telemetry.writer import TraceReader, TraceWriter

        path = tmp_path / "t.jsonl"
        with TraceWriter(path, append=False) as w:
            w.write_many(traces)
        assert TraceReader(path).count() == len(traces)

    def test_gzip_roundtrip(self, traces, tmp_path):
        from argus.telemetry.writer import TraceReader, TraceWriter

        path = tmp_path / "t.jsonl.gz"
        with TraceWriter(path, compress=True, append=False) as w:
            w.write_many(traces[:3])
        assert len(TraceReader(path).read_all()) == 3


class TestOutcomes:
    def test_attack_success_requires_target_answer(self, corpus):
        t = Trace(trace_id="t", query_id="q", query="x")
        t.final_answer = "the answer is 1972"
        t.target_answer = "1972"
        t.gold_answer = "1983"
        assert t.attack_success
        assert not t.answered_correctly

    def test_correct_answer_is_not_attack_success(self, corpus):
        t = Trace(trace_id="t", query_id="q", query="x")
        t.final_answer = "1983"
        t.target_answer = "1972"
        t.gold_answer = "1983"
        assert not t.attack_success
        assert t.answered_correctly

    def test_poison_in_context_needs_retrieval(self, agent, corpus):
        t = agent.run(corpus.queries[0], poison_doc_ids={"never_retrieved"})
        assert not t.poison_in_context
        assert t.poison_rank == -1

    def test_empty_answer_is_neither(self):
        t = Trace(trace_id="t", query_id="q", query="x")
        t.final_answer = ""
        t.target_answer = "1972"
        assert not t.attack_success


class TestDeterminism:
    def test_same_seed_same_answer(self, retriever, corpus):
        a = AgenticRAG(retriever, MockLLM(seed=3), MECHANISM_CONFIGS["C5"], corpus)
        b = AgenticRAG(retriever, MockLLM(seed=3), MECHANISM_CONFIGS["C5"], corpus)
        assert a.run(corpus.queries[0]).final_answer == b.run(corpus.queries[0]).final_answer

    def test_mock_responds_to_poisoned_context(self, corpus, poisoned):
        """The mock must actually be influenced by poison, otherwise every security
        number the platform produces offline would be meaningless."""
        from argus.retrieval.bm25 import BM25Retriever

        r = BM25Retriever(poisoned.corpus)
        r.build()
        agent = AgenticRAG(r, MockLLM(seed=3), MECHANISM_CONFIGS["C0"], poisoned.corpus)
        successes = sum(
            agent.run(q, poison_doc_ids=poisoned.poison_for(q.query_id)).attack_success
            for q in poisoned.corpus.queries
        )
        assert successes > 0, "mock never falls for poison; the simulation is inert"
