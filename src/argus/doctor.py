"""Environment and model health check.

Run this before anything long. It answers a question the rest of the pipeline assumes:
*does this model, on this endpoint, behave the way the engine needs it to?*

Reachability is the easy half. The half that matters is behaviour. The agent turns free
text from the generator into control decisions at three points — which document to
inspect, whether evidence is sufficient, what to search for next — and a model that
phrases its answers differently does not crash, it quietly degrades. A run where every
reflect response fell back to SUFFICIENT still produces a complete grid, plausible
numbers and a wrong conclusion. That is the failure mode this whole project exists to
have caught, so it gets a preflight rather than a footnote.

Each probe therefore *asserts* rather than prints, and the exit code reflects the worst
result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
from argus.config import MECHANISM_CONFIGS, ArgusConfig
from argus.llm.base import LLMBackend

#: A short passage of roughly the length a real corpus supplies, used to build prompts
#: that exercise the model at the size it will actually see.
#:
#: The subject is invented on purpose. A probe built on real-world facts cannot tell
#: whether the model read the context or answered from memory, and grounding is precisely
#: what the answer probe needs to establish. The question also has exactly one defensible
#: answer: an earlier version asked "where does the optic nerve cross the midline" over a
#: passage that named both "the optic chiasm" and "beneath the hypothalamus", and a model
#: answering the second — a correct answer to a "where" question — was reported as a
#: failure. A probe that fails a well-behaved model is worse than no probe.
_PASSAGE = (
    "The Kelvin Observatory was founded in 1897 by the Naismith Trust, which had been "
    "established a decade earlier to support astronomical work in the region. The site "
    "operated for eleven years before its principal instruments were transferred to a "
    "regional repository. Administrative responsibility changed twice during that "
    "period, and the surviving records are held in the Lindqvist Archive. A summary of "
    "the observing programme accompanies the final annual report."
)

_QUESTION = "In what year was the Kelvin Observatory founded?"
_GOLD = "1897"


@dataclass
class Probe:
    name: str
    ok: bool
    detail: str = ""
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    fatal: bool = True
    response: str = ""

    def line(self) -> str:
        mark = "PASS" if self.ok else ("FAIL" if self.fatal else "WARN")
        timing = f"{self.latency_ms:>7.0f} ms" if self.latency_ms else " " * 10
        return f"  [{mark}] {self.name:<28s}{timing}  {self.detail}"


@dataclass
class DoctorReport:
    probes: list[Probe] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(p.ok for p in self.probes if p.fatal)

    def add(self, probe: Probe) -> Probe:
        self.probes.append(probe)
        return probe


def _context_block(n_docs: int, salt: str = "") -> str:
    """Render n documents the way `AgentState.context_block` does.

    `salt` varies the filler text between calls. Gateways and inference servers commonly
    cache identical prompts, so timing the same request five times measures the cache
    rather than the model and produces a wall-clock projection that is too optimistic to
    plan against.
    """
    tail = f" Reference {salt}." if salt else ""
    return "\n\n".join(
        f"[DOC {i + 1}] Kelvin Observatory\n{_PASSAGE}{tail}" for i in range(n_docs)
    )


def _timed(llm: LLMBackend, prompt: str, system: str, task: str) -> tuple[Any, float]:
    t0 = time.perf_counter()
    resp = llm.generate(prompt, system=system, task=task)
    return resp, (time.perf_counter() - t0) * 1000.0


def run_doctor(
    cfg: ArgusConfig,
    llm: LLMBackend,
    throughput_samples: int = 5,
    total_runs: int = 51_400,
) -> DoctorReport:
    """Exercise the endpoint and every model behaviour the engine depends on."""
    report = DoctorReport()
    agent_cfg = MECHANISM_CONFIGS["C5"]

    report.info = {
        "backend": cfg.llm.backend,
        "model": cfg.llm.model,
        "base_url": cfg.llm.base_url or "(provider default)",
        "api_key": f"set ({len(cfg.llm.api_key)} chars)" if cfg.llm.api_key else "NOT SET",
        "timeout_s": cfg.llm.timeout_s,
        "max_retries": cfg.llm.max_retries,
        "max_tokens": cfg.llm.max_tokens,
        "temperature": cfg.llm.temperature,
        "workers": cfg.workers,
        "seed": cfg.seed,
        "data_dir": str(cfg.data_dir),
        "budget_usd": cfg.budget_usd,
        "retriever": cfg.retrieval.backend,
    }

    # ---- 0. the backend must not be the simulator ---------------------------
    if cfg.llm.backend == "mock":
        report.add(
            Probe(
                "backend is real",
                ok=False,
                detail=(
                    "backend is 'mock'. It exercises the pipeline but its output is not a "
                    "research result. Set ARGUS_LLM_BACKEND=openai in .env."
                ),
            )
        )
        return report
    report.add(Probe("backend is real", True, cfg.llm.backend))

    # ---- 1. reachability ----------------------------------------------------
    try:
        resp, ms = _timed(llm, "Reply with the single word OK.", "Be terse.", "answer")
        report.add(
            Probe(
                "endpoint reachable", True,
                f"replied {resp.text.strip()[:40]!r}",
                ms, resp.input_tokens, resp.output_tokens,
            )
        )
    except Exception as exc:
        cause = getattr(exc, "__cause__", None)
        report.add(
            Probe(
                "endpoint reachable", False,
                f"{type(exc).__name__}: {exc}" + (f"  <- {cause!r}" if cause else ""),
            )
        )
        return report  # nothing else can be checked

    # ---- 2. the four behaviours the engine parses ---------------------------
    # answer: must return a short span, not an essay. Verbosity does not break scoring
    # (matching is a substring test) but it inflates tokens and signals a prompt mismatch.
    try:
        resp, ms = _timed(
            llm, answer_prompt(_QUESTION, _context_block(3)), SYSTEM_ANSWER, "answer"
        )
        text = resp.text.strip()
        correct = _GOLD.lower() in text.lower()
        terse = len(text) <= 120
        report.add(
            Probe(
                "answer: extracts the span", correct,
                f"{text[:70]!r}" + ("" if correct else "  <- expected 'optic chiasm'"),
                ms, resp.input_tokens, resp.output_tokens, response=text,
            )
        )
        report.add(
            Probe(
                "answer: stays terse", terse,
                f"{len(text)} chars" + ("" if terse else "  <- model is explaining itself; "
                                        "token costs and latency will be higher than planned"),
                fatal=False, response=text,
            )
        )
    except Exception as exc:
        report.add(Probe("answer: extracts the span", False, f"{type(exc).__name__}: {exc}"))

    # reflect: the verdict gates the whole of M4. A model that never says INSUFFICIENT
    # turns C4 into C0 while still costing an extra call per iteration.
    try:
        resp, ms = _timed(
            llm, reflect_prompt(_QUESTION, _context_block(3)), SYSTEM_REFLECT, "reflect"
        )
        parsed = parse_reflection(resp.text)
        report.add(
            Probe(
                "reflect: verdict parses", parsed.ok,
                f"{resp.text.strip()[:50]!r} -> {parsed.value}"
                + ("" if parsed.ok else f"  <- {parsed.reason}; reflection would silently "
                   "default to SUFFICIENT and M4 would measure nothing"),
                ms, resp.input_tokens, resp.output_tokens, response=resp.text,
            )
        )
    except Exception as exc:
        report.add(Probe("reflect: verdict parses", False, f"{type(exc).__name__}: {exc}"))

    # inspect: a chatty response used to be read as a concatenation of every digit.
    try:
        summaries = "\n".join(f"{i + 1}. Optic chiasm: {_PASSAGE[:80]}" for i in range(5))
        resp, ms = _timed(llm, inspect_prompt(_QUESTION, summaries), SYSTEM_INSPECT, "inspect")
        parsed = parse_document_choice(resp.text, 5)
        report.add(
            Probe(
                "inspect: index parses", parsed.ok,
                f"{resp.text.strip()[:50]!r} -> doc {parsed.value + 1}"
                + ("" if parsed.ok else f"  <- {parsed.reason}"),
                ms, resp.input_tokens, resp.output_tokens, response=resp.text,
            )
        )
    except Exception as exc:
        report.add(Probe("inspect: index parses", False, f"{type(exc).__name__}: {exc}"))

    # rewrite: the response goes straight into the retriever, so a preamble becomes
    # query terms and silently degrades retrieval for every M1 run.
    try:
        resp, ms = _timed(
            llm, rewrite_prompt(_QUESTION, [], _PASSAGE[:200]), SYSTEM_REWRITE, "rewrite"
        )
        parsed = parse_rewrite(resp.text, _QUESTION)
        report.add(
            Probe(
                "rewrite: query parses", parsed.ok,
                f"{resp.text.strip()[:50]!r} -> {parsed.value[:50]!r}"
                + ("" if parsed.ok else f"  <- {parsed.reason}"),
                ms, resp.input_tokens, resp.output_tokens, response=resp.text,
            )
        )
    except Exception as exc:
        report.add(Probe("rewrite: query parses", False, f"{type(exc).__name__}: {exc}"))

    # ---- 3. full-size prompt ------------------------------------------------
    # A toy prompt hides context-length limits and timeouts. C5 renders max_context_docs
    # documents, which is the largest prompt the grid will send.
    try:
        big = answer_prompt(_QUESTION, _context_block(agent_cfg.max_context_docs))
        resp, ms = _timed(llm, big, SYSTEM_ANSWER, "answer")
        report.add(
            Probe(
                f"full-size prompt ({agent_cfg.max_context_docs} docs)", True,
                f"{resp.input_tokens} in / {resp.output_tokens} out",
                ms, resp.input_tokens, resp.output_tokens,
            )
        )
        if ms > cfg.llm.timeout_s * 1000 * 0.5:
            report.warnings.append(
                f"A full-size prompt took {ms / 1000:.1f}s against a {cfg.llm.timeout_s:.0f}s "
                "timeout. Raise ARGUS_LLM_TIMEOUT_S; a timeout mid-grid is indistinguishable "
                "from a model failure and costs the retry budget."
            )
    except Exception as exc:
        report.add(
            Probe(
                f"full-size prompt ({agent_cfg.max_context_docs} docs)", False,
                f"{type(exc).__name__}: {exc}  <- the grid's largest prompts will fail",
            )
        )

    # ---- 4. throughput ------------------------------------------------------
    if throughput_samples > 0:
        times: list[float] = []
        for i in range(throughput_samples):
            # Each sample carries a different salt, so a caching gateway cannot serve the
            # second and later calls from the first and make the endpoint look faster
            # than it is.
            prompt = answer_prompt(_QUESTION, _context_block(5, salt=f"{i:04d}"))
            try:
                _, ms = _timed(llm, prompt, SYSTEM_ANSWER, "answer")
                times.append(ms)
            except Exception:
                break
        if times:
            mean_ms = sum(times) / len(times)
            report.info["mean_call_ms"] = round(mean_ms, 1)
            # About 2.3 LLM calls per agent run averaged across the seven configurations:
            # C0 and C6 make one or two, C5 makes five or more.
            per_run_s = mean_ms * 2.3 / 1000.0
            report.info["projected_hours"] = {
                f"{w} worker{'s' if w > 1 else ''}": round(total_runs * per_run_s / 3600 / w, 1)
                for w in (1, 2, 4, 8)
            }

    # ---- 5. cost and budget sanity -----------------------------------------
    if llm.usage.price_is_notional:
        report.warnings.append(
            f"Cost is priced notionally against '{llm.usage.reference_model}'. Tokens and "
            "latency are the real cost of a self-hosted run; label the dollar column as "
            "notional in the write-up."
        )
    if cfg.budget_usd > 0:
        report.warnings.append(
            f"ARGUS_BUDGET_USD={cfg.budget_usd:g} arms the mid-run spend guard. On a "
            "self-hosted endpoint those dollars are fictional and the guard can truncate a "
            "multi-hour grid partway through. Set ARGUS_BUDGET_USD=0 to disable it."
        )
    if cfg.workers > 1 and cfg.retrieval.backend != "bm25":
        report.warnings.append(
            f"workers={cfg.workers} with the {cfg.retrieval.backend} retriever: those cells "
            "will drop to a single worker, because the embedding model is shared."
        )

    return report


def format_report(report: DoctorReport) -> str:
    """Render the report for a terminal."""
    lines = ["=" * 74, "ARGUS doctor", "=" * 74, "", "Configuration"]
    for key, value in report.info.items():
        if key == "projected_hours":
            continue
        lines.append(f"  {key:<16s} {value}")

    lines += ["", "Checks"]
    lines += [p.line() for p in report.probes]

    if "projected_hours" in report.info:
        lines += ["", "Projected wall-clock for the full programme (51,400 runs)"]
        lines.append(f"  measured mean call: {report.info['mean_call_ms']:.0f} ms")
        for label, hours in report.info["projected_hours"].items():
            lines.append(f"  {label:<12s} {hours:>7.1f} h")
        lines.append(
            "  These assume the endpoint scales with concurrency. If it serialises "
            "internally,\n  more workers will not help; compare 1 and 4 in practice."
        )

    if report.warnings:
        lines += ["", "Warnings"]
        lines += [f"  ! {w}" for w in report.warnings]

    lines += ["", "=" * 74]
    lines.append("DOCTOR PASSED" if report.ok else "DOCTOR FAILED - do not launch the grid")
    lines.append("=" * 74)
    return "\n".join(lines)
