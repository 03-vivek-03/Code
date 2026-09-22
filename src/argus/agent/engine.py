"""The agentic RAG engine.

This is the experimental apparatus for Study A. Each of the four mechanisms can be
switched independently, and every action is emitted as a span, which is what lets the
ablation attribute an effect to a mechanism rather than to an architecture.

Execution flow, with mechanisms in brackets:

    question
      -> [M1] rewrite query
      -> retrieve
      -> [M3] inspect a document in full
      -> [M4] reflect on sufficiency
      -> [M2] iterate if unsatisfied and budget remains
      -> answer

A note on isolating M2 and M4, since it is the subtlest part of the design.

M2 without M1 re-issues the same query, so to make iteration meaningful the retriever
excludes documents already seen. That models an agent searching deeper into the ranking
rather than uselessly repeating round one.

M4 without M2 cannot act on an INSUFFICIENT verdict by searching again, because there is
no iteration. It acts by answering more conservatively, which is a real behaviour and
gives reflection a distinct, measurable effect on its own. That conservative answering is
controlled by `AgentConfig.reflection_caution` and can be switched off (configuration C6)
so the analysis can tell apart reflection's two effects: judging the evidence, and
declining to answer. In the first full run they were conflated, and the outcome breakdown
showed most of reflection's apparent protection was abstention rather than resistance.

A note on what gets logged
--------------------------

The answer span records `argus.context_document_ids`: the documents actually rendered
into the answer prompt, not the union of everything retrieved. Those are different sets
whenever the agent iterates, and the analysis previously read the union while calling it
the context. That understated the poisoned share of the prompt for every multi-iteration
configuration and made `poison_rank` incomparable across configurations, because ranks
were recorded against each round's post-exclusion candidate list.
"""

from __future__ import annotations

from typing import Any

from argus.agent.parsing import (
    parse_document_choice,
    parse_reflection,
    parse_rewrite,
)
from argus.agent.prompts import (
    SYSTEM_ANSWER,
    SYSTEM_INSPECT,
    SYSTEM_REFLECT,
    SYSTEM_REWRITE,
    answer_prompt,
    inspect_prompt,
    reflect_prompt,
    rewrite_prompt,
)
from argus.agent.state import AgentState
from argus.config import AgentConfig
from argus.corpus.store import Corpus, Query
from argus.llm.base import LLMBackend
from argus.retrieval.base import Retriever
from argus.telemetry.spans import Trace, derived_id
from argus.telemetry.tracer import Tracer

CAUTION_SUFFIX = (
    "\n\nNote: the evidence was judged incomplete. Answer only if the context clearly "
    "supports it, otherwise say you cannot determine the answer."
)


class AgenticRAG:
    """Configurable agentic RAG pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMBackend,
        config: AgentConfig,
        corpus: Corpus | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config
        self.corpus = corpus or retriever.corpus

    # ------------------------------------------------------------------- run
    def run(self, query: Query, poison_doc_ids: set[str] | None = None, **meta: Any) -> Trace:
        """Answer one query and return the full execution trace."""
        cfg = self.config
        # Derived rather than random, so re-running a cell reproduces its identifiers and
        # a concurrently-executed cell is comparable field by field with a sequential one.
        tracer = Tracer(
            query_id=query.query_id,
            query=query.text,
            trace_id=derived_id("t_", str(meta.get("cell_id", "")), query.query_id, cfg.name),
        )
        state = AgentState(question=query.text, query_id=query.query_id, queries=[query.text])

        max_iters = cfg.iteration_budget if cfg.iterative_retrieval else 1

        for it in range(max_iters):
            state.iteration = it

            # ---- M1: query rewriting ---------------------------------------
            if cfg.query_rewriting and state.n_rewrites < cfg.max_rewrites:
                # Rewrite before the first retrieval, and again before each later one.
                self._rewrite(state, tracer)

            # ---- retrieval --------------------------------------------------
            result = self.retriever.retrieve(
                state.current_query,
                top_k=cfg.top_k,
                exclude=state.seen_doc_ids if cfg.iterative_retrieval else None,
            )
            tracer.record_retrieval(
                query=state.current_query,
                doc_ids=result.doc_ids,
                scores=result.scores,
                iteration=it,
                backend=result.backend,
                latency_ms=result.latency_ms,
                is_rewritten=state.n_rewrites > 0,
                global_ranks=result.global_ranks,
            )
            state.add_evidence(result.documents, result.scores, iteration=it)

            # ---- M3: document inspection ------------------------------------
            if cfg.document_inspection and state.n_inspections < cfg.max_inspections:
                self._inspect(state, tracer)

            # ---- M4: reflection ---------------------------------------------
            verdict = ""
            if cfg.reflection:
                verdict = self._reflect(state, tracer)

            # ---- M2: stopping decision --------------------------------------
            # The stopping decision belongs to iterative retrieval. Reflection informs
            # it when present; without reflection the agent searches until its budget
            # is exhausted, which is what an iterating agent with no critic does.
            last_iteration = it == max_iters - 1
            if last_iteration:
                state.stopped_because = "budget_exhausted" if max_iters > 1 else "single_pass"
                if max_iters > 1:
                    tracer.record_mechanism(
                        "M2_iterate", "stop", iteration=it, reason="budget_exhausted"
                    )
                break
            if verdict == "SUFFICIENT":
                state.stopped_because = "reflection_satisfied"
                tracer.record_mechanism(
                    "M2_iterate", "stop", iteration=it, reason=state.stopped_because
                )
                break
            tracer.record_mechanism(
                "M2_iterate",
                "continue",
                iteration=it,
                reason="reflection_insufficient" if cfg.reflection else "budget_remaining",
            )

        # ---- final answer ---------------------------------------------------
        # The caution instruction is a separate switch from reflection itself, so C4 and
        # C6 differ only in whether an INSUFFICIENT verdict is allowed to change the
        # answer prompt. Everything else about the two runs is identical.
        verdict_insufficient = bool(
            state.reflection_history and state.reflection_history[-1] == "INSUFFICIENT"
        )
        cautious = cfg.reflection and cfg.reflection_caution and verdict_insufficient
        answer = self._answer(state, tracer, cautious=cautious)
        state.final_answer = answer

        # Token accounting comes from this run's own spans, not from differencing the
        # shared UsageTracker. The tracker is global to the backend, so under concurrent
        # execution a difference would sweep in whatever other threads spent in between
        # and attribute it here. Summing the spans is exact and thread-safe.
        used_in, used_out = tracer.token_totals()
        in_price, out_price = self.llm.usage.input_price_per_m, self.llm.usage.output_price_per_m
        run_cost = used_in / 1_000_000 * in_price + used_out / 1_000_000 * out_price

        return tracer.finish(
            final_answer=answer,
            gold_answer=query.gold_answer,
            target_answer=query.target_answer,
            poison_doc_ids=sorted(poison_doc_ids or set()),
            config_id=cfg.name,
            config=cfg.to_dict(),
            llm_backend=self.llm.name,
            llm_model=self.llm.model,
            total_input_tokens=used_in,
            total_output_tokens=used_out,
            cost_usd=run_cost,
            meta={
                **meta,
                "n_rewrites": state.n_rewrites,
                "n_inspections": state.n_inspections,
                "n_reflections": state.n_reflections,
                "stopped_because": state.stopped_because,
                "reflection_history": list(state.reflection_history),
                "n_evidence": state.n_evidence,
                "queries": list(state.queries),
                # Whether the caution instruction was actually applied on this run. The
                # abstention analysis conditions on it, so it must be recorded rather
                # than inferred from the configuration.
                "caution_applied": cautious,
                "reflection_insufficient": verdict_insufficient,
                # Mechanism responses the parsers could not read. Non-zero means this run
                # is not exercising the mechanism it claims to.
                "parse_failures": state.parse_failures,
                "parse_attempts": state.parse_attempts,
            },
        )

    # ------------------------------------------------------------ mechanisms
    def _rewrite(self, state: AgentState, tracer: Tracer) -> None:
        preview = state.context_block(max_docs=2)[:600]
        prompt = rewrite_prompt(state.question, state.queries[1:], preview)
        resp = self.llm.generate(prompt, system=SYSTEM_REWRITE, task="rewrite", question=state.question)
        tracer.record_inference(
            "rewrite", prompt, resp.text, resp.input_tokens, resp.output_tokens,
            resp.latency_ms, resp.model, n_context_docs=state.n_evidence,
        )
        parsed = parse_rewrite(resp.text, state.current_query)
        state.queries.append(parsed.value)
        state.n_rewrites += 1
        state.record_parse(parsed.ok)
        tracer.record_mechanism(
            "M1_rewrite", "rewritten",
            original=state.question, rewritten=parsed.value,
            rewrite_index=state.n_rewrites,
            parse_ok=parsed.ok, parse_reason=parsed.reason,
        )

    def _inspect(self, state: AgentState, tracer: Tracer) -> None:
        if not state.evidence:
            return
        candidates = sorted(state.evidence, key=lambda e: -e.score)[: self.config.top_k]
        summaries = "\n".join(
            f"{i + 1}. {e.title or e.doc_id}: {e.text[:100]}" for i, e in enumerate(candidates)
        )
        prompt = inspect_prompt(state.question, summaries)
        resp = self.llm.generate(
            prompt, system=SYSTEM_INSPECT, task="inspect", n_docs=len(candidates)
        )
        tracer.record_inference(
            "inspect", prompt, resp.text, resp.input_tokens, resp.output_tokens,
            resp.latency_ms, resp.model, n_context_docs=len(candidates),
        )

        parsed = parse_document_choice(resp.text, len(candidates))
        idx = parsed.value
        state.record_parse(parsed.ok)
        chosen = candidates[idx]

        full = self.corpus.get(chosen.doc_id)
        if full is not None:
            # Inspection promotes the document: full text and a score boost, which is
            # what "reading it properly" means for downstream ranking.
            for item in state.evidence:
                if item.doc_id == chosen.doc_id:
                    item.text = full.text
                    item.via = "inspection"
                    item.score = max(item.score, max((e.score for e in state.evidence), default=0.0) * 1.05)

        state.n_inspections += 1
        tracer.record_tool(
            "get_document_by_id",
            {"document_id": chosen.doc_id, "rank": idx},
            result_preview=(full.text if full else "")[:200],
        )
        tracer.record_mechanism(
            "M3_inspect", "inspected", document_id=chosen.doc_id,
            inspect_index=state.n_inspections,
            parse_ok=parsed.ok, parse_reason=parsed.reason,
        )

    def _reflect(self, state: AgentState, tracer: Tracer) -> str:
        context = state.context_block(max_docs=self.config.max_context_docs)
        prompt = reflect_prompt(state.question, context)
        resp = self.llm.generate(
            prompt, system=SYSTEM_REFLECT, task="reflect", n_docs=state.n_evidence
        )
        tracer.record_inference(
            "reflect", prompt, resp.text, resp.input_tokens, resp.output_tokens,
            resp.latency_ms, resp.model, n_context_docs=state.n_evidence,
        )
        parsed = parse_reflection(resp.text)
        verdict = parsed.value
        state.reflection_history.append(verdict)
        state.n_reflections += 1
        state.record_parse(parsed.ok)
        tracer.record_mechanism(
            "M4_reflect", verdict, iteration=state.iteration, n_evidence=state.n_evidence,
            parse_ok=parsed.ok, parse_reason=parsed.reason,
        )
        return verdict

    def _answer(self, state: AgentState, tracer: Tracer, cautious: bool = False) -> str:
        max_docs = self.config.max_context_docs
        context = state.context_block(max_docs=max_docs)
        # The exact set rendered into the prompt. Everything downstream that talks about
        # "the context" must read this, not the union of all retrieval spans.
        context_ids = state.context_doc_ids(max_docs=max_docs)

        prompt = answer_prompt(state.question, context)
        if cautious:
            prompt += CAUTION_SUFFIX
        resp = self.llm.generate(
            prompt, system=SYSTEM_ANSWER, task="answer", question=state.question
        )
        tracer.record_inference(
            "answer", prompt, resp.text, resp.input_tokens, resp.output_tokens,
            resp.latency_ms, resp.model,
            n_context_docs=len(context_ids),
            context_doc_ids=context_ids,
            cautious=cautious,
        )
        return resp.text.strip()
