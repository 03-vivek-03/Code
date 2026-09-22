"""Tracer: builds a Trace while the agent runs.

Usage mirrors OpenTelemetry closely enough to be familiar:

    tracer = Tracer(trace_id, query_id, query)
    with tracer.span("retrieve", SpanKind.RETRIEVAL) as span:
        span.set("retrieval.query", q)
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from argus.telemetry.spans import Span, SpanKind, Trace, derived_id, new_id


class Tracer:
    """Builds a Trace while the agent runs.

    Identifiers are derived, not random. `trace_id` is a hash of the cell, the query and
    the configuration, and span ids are a hash of the trace id and the span's position.
    Two reasons: the trace corpus is a released research artefact and its identifiers
    should be stable across re-runs so a result can be traced to a specific run rather
    than to a UUID that changes every time; and it means a cell executed concurrently
    produces the same identifiers as one executed sequentially, so the two are comparable
    field by field.
    """

    def __init__(self, query_id: str, query: str, trace_id: str | None = None) -> None:
        self.trace = Trace(
            trace_id=trace_id or new_id("t_"),
            query_id=query_id,
            query=query,
        )
        self._stack: list[str] = []
        self._n_spans = 0
        self._t0 = time.perf_counter()

    def _next_span_id(self) -> str:
        self._n_spans += 1
        return derived_id("s_", self.trace.trace_id, str(self._n_spans))

    # ------------------------------------------------------------------ spans
    @contextmanager
    def span(self, name: str, kind: SpanKind, **attributes: Any) -> Iterator[Span]:
        sp = Span(
            span_id=self._next_span_id(),
            trace_id=self.trace.trace_id,
            name=name,
            kind=kind,
            parent_id=self._stack[-1] if self._stack else None,
            start_ms=(time.perf_counter() - self._t0) * 1000.0,
            attributes=dict(attributes),
        )
        self._stack.append(sp.span_id)
        try:
            yield sp
        except Exception as exc:
            sp.status = "error"
            sp.set("error.type", type(exc).__name__)
            sp.set("error.message", str(exc)[:500])
            raise
        finally:
            sp.end_ms = (time.perf_counter() - self._t0) * 1000.0
            self._stack.pop()
            self.trace.spans.append(sp)

    # ------------------------------------------------- convenience recorders
    def record_retrieval(
        self,
        query: str,
        doc_ids: list[str],
        scores: list[float],
        iteration: int,
        backend: str,
        latency_ms: float,
        is_rewritten: bool = False,
        global_ranks: list[int] | None = None,
    ) -> Span:
        with self.span(f"retrieve[{iteration}]", SpanKind.RETRIEVAL) as sp:
            sp.set("retrieval.query", query)
            sp.set("retrieval.document_ids", list(doc_ids))
            sp.set("retrieval.scores", [round(float(s), 6) for s in scores])
            sp.set("retrieval.top_k", len(doc_ids))
            sp.set("retrieval.iteration", iteration)
            sp.set("retrieval.backend", backend)
            sp.set("retrieval.latency_ms", round(latency_ms, 3))
            sp.set("argus.query_is_rewritten", is_rewritten)
            # Rank in the unfiltered ranking. Iterative retrieval excludes documents
            # already seen, so the position within this span's own list is not
            # comparable across rounds; this attribute is.
            sp.set(
                "retrieval.global_ranks",
                list(global_ranks) if global_ranks is not None else list(range(len(doc_ids))),
            )
        return self.trace.spans[-1]

    def record_inference(
        self,
        task: str,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model: str,
        n_context_docs: int = 0,
        context_doc_ids: list[str] | None = None,
        cautious: bool = False,
    ) -> Span:
        with self.span(f"llm[{task}]", SpanKind.INFERENCE) as sp:
            sp.set("gen_ai.operation.name", "chat")
            sp.set("gen_ai.request.model", model)
            sp.set("gen_ai.usage.input_tokens", input_tokens)
            sp.set("gen_ai.usage.output_tokens", output_tokens)
            sp.set("gen_ai.response.text_length", len(response))
            sp.set("argus.task", task)
            sp.set("argus.prompt_length", len(prompt))
            sp.set("argus.n_context_docs", n_context_docs)
            sp.set("argus.latency_ms", round(latency_ms, 3))
            # Truncated so trace files stay a manageable size on disk.
            sp.set("argus.response_preview", response[:200])
            if context_doc_ids is not None:
                # The documents actually rendered into this prompt. Analysis of "what
                # was in the context" reads this and nothing else.
                sp.set("argus.context_document_ids", list(context_doc_ids))
            if task == "answer":
                sp.set("argus.caution_applied", bool(cautious))
        return self.trace.spans[-1]

    def record_tool(self, tool: str, arguments: dict[str, Any], result_preview: str) -> Span:
        with self.span(f"tool[{tool}]", SpanKind.TOOL) as sp:
            sp.set("gen_ai.tool.name", tool)
            for key, val in arguments.items():
                sp.set(f"tool.{key}", val)
            sp.set("tool.result_preview", str(result_preview)[:200])
        return self.trace.spans[-1]

    def record_mechanism(self, mechanism: str, decision: str, **attrs: Any) -> Span:
        with self.span(f"mechanism[{mechanism}]", SpanKind.MECHANISM) as sp:
            sp.set("argus.mechanism", mechanism)
            sp.set("argus.decision", decision)
            for key, val in attrs.items():
                sp.set(f"argus.{key}", val)
        return self.trace.spans[-1]

    # ----------------------------------------------------------------- finish
    def token_totals(self) -> tuple[int, int]:
        """(input, output) tokens across this trace's own inference spans.

        The engine uses this instead of differencing the backend's shared UsageTracker.
        That tracker is global, so under concurrent execution a before/after difference
        would sweep in whatever other threads spent in between. Summing this trace's own
        spans is exact regardless of how many runs are in flight.
        """
        tin = tout = 0
        for span in self.trace.spans:
            if span.kind is SpanKind.INFERENCE:
                tin += int(span.attributes.get("gen_ai.usage.input_tokens", 0) or 0)
                tout += int(span.attributes.get("gen_ai.usage.output_tokens", 0) or 0)
        return tin, tout

    def finish(self, **fields: Any) -> Trace:
        for key, val in fields.items():
            if hasattr(self.trace, key):
                setattr(self.trace, key, val)
        self.trace.total_latency_ms = (time.perf_counter() - self._t0) * 1000.0
        return self.trace
