"""Feature extraction from execution traces.

This is the input side of Study B. Everything here is computed from spans alone: no
model internals, no logits, no re-running the generator. That constraint is the
contribution, so it is enforced rather than assumed. Nothing in this module has access
to the corpus, the poison labels, or the answer key.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from argus.features.schema import feature_names
from argus.telemetry.spans import REFUSAL_MARKERS, SpanKind, Trace

FEATURE_NAMES = feature_names()

#: Imported from the span model rather than redefined here. The abstention analysis in
#: Study A and the `answer_is_refusal` feature in Study B must agree on what a refusal
#: is; two independent lists silently drift and the two studies stop describing the same
#: runs.
_REFUSAL = REFUSAL_MARKERS


def _tokens(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t}


def _jaccard_distance(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return 1.0 - len(ta & tb) / len(union)


def _entropy(counts: Iterable[float]) -> float:
    vals = [c for c in counts if c > 0]
    total = sum(vals)
    if total <= 0 or len(vals) <= 1:
        return 0.0
    probs = [v / total for v in vals]
    return float(-sum(p * math.log2(p) for p in probs))


def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    return _entropy(Counter(text).values())


def _skew(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    arr = np.asarray(values, dtype=float)
    sd = arr.std()
    if sd < 1e-9:
        return 0.0
    return float(((arr - arr.mean()) ** 3).mean() / sd**3)


class FeatureExtractor:
    """Turns a Trace into a flat numeric feature vector.

    Corpus-level statistics, used by the z-score and bigram-novelty features, are fitted
    on training traces only and then frozen, so no test-set information leaks into the
    features.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._answer_len_mean = 0.0
        self._answer_len_std = 1.0
        self._bigram_freq: Counter[str] = Counter()
        self._n_fit_traces = 0

    # ------------------------------------------------------------------- fit
    def fit(self, traces: Iterable[Trace]) -> FeatureExtractor:
        lengths: list[int] = []
        bigrams: Counter[str] = Counter()
        n = 0
        for trace in traces:
            lengths.append(len(trace.final_answer))
            for bg in self._tool_bigrams(trace):
                bigrams[bg] += 1
            n += 1
        self._answer_len_mean = float(np.mean(lengths)) if lengths else 0.0
        self._answer_len_std = float(np.std(lengths)) or 1.0
        self._bigram_freq = bigrams
        self._n_fit_traces = max(n, 1)
        self._fitted = True
        return self

    # --------------------------------------------------------------- extract
    def extract(self, trace: Trace) -> dict[str, float]:
        f: dict[str, float] = {}
        f.update(self._retrieval_dynamics(trace))
        f.update(self._query_trajectory(trace))
        f.update(self._evidence_provenance(trace))
        f.update(self._control_flow(trace))
        f.update(self._output_behaviour(trace))
        f.update(self._resource(trace))
        # Guarantee a stable, complete vector even for degenerate traces.
        return {name: float(f.get(name, 0.0)) for name in FEATURE_NAMES}

    def extract_many(self, traces: Iterable[Trace]) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
        """Returns (feature dicts, metadata dicts) in matching order."""
        rows, meta = [], []
        for trace in traces:
            rows.append(self.extract(trace))
            meta.append(
                {
                    "trace_id": trace.trace_id,
                    "query_id": trace.query_id,
                    "label": trace.label,
                    "y": 1 if trace.is_compromised else 0,
                    "attack": trace.attack,
                    "config_id": trace.config_id,
                    "dataset": trace.dataset,
                    "retriever": trace.retriever,
                    "n_poison_in_corpus": trace.n_poison_in_corpus,
                    "attack_success": trace.attack_success,
                    "answered_correctly": trace.answered_correctly,
                    "poison_in_context": trace.poison_in_context,
                }
            )
        return rows, meta

    # ------------------------------------------------------------- families
    @staticmethod
    def _retrieval_dynamics(trace: Trace) -> dict[str, float]:
        spans = trace.retrieval_spans
        if not spans:
            return {}
        per_iter = [list(s.attributes.get("retrieval.scores", [])) for s in spans]
        flat = [s for scores in per_iter for s in scores]
        if not flat:
            return {}

        first = per_iter[0]
        out = {
            "retr_score_mean": float(np.mean(flat)),
            "retr_score_std": float(np.std(flat)),
            "retr_score_max": float(np.max(flat)),
            "retr_score_top_gap": float(first[0] - first[1]) if len(first) > 1 else 0.0,
            "retr_score_range": float(max(first) - min(first)) if first else 0.0,
            "retr_score_skew": _skew(flat),
        }

        means = [float(np.mean(s)) for s in per_iter if s]
        out["retr_score_drift"] = float(means[-1] - means[0]) if len(means) > 1 else 0.0

        churn = []
        id_lists = [list(s.attributes.get("retrieval.document_ids", [])) for s in spans]
        for prev, curr in zip(id_lists, id_lists[1:]):
            if not curr:
                continue
            churn.append(len(set(curr) - set(prev)) / len(curr))
        out["retr_rank_churn"] = float(np.mean(churn)) if churn else 0.0
        return out

    @staticmethod
    def _query_trajectory(trace: Trace) -> dict[str, float]:
        queries = [
            s.attributes.get("retrieval.query", "") for s in trace.retrieval_spans
        ]
        queries = [q for q in queries if q]
        rewrites = [
            s for s in trace.mechanism_spans if s.attributes.get("argus.mechanism") == "M1_rewrite"
        ]
        original = trace.query
        if not queries:
            return {"n_rewrites": float(len(rewrites))}

        steps = [
            _jaccard_distance(a, b) for a, b in zip(queries, queries[1:])
        ]
        return {
            "n_rewrites": float(len(rewrites)),
            "query_drift_first": _jaccard_distance(original, queries[0]),
            "query_drift_total": _jaccard_distance(original, queries[-1]),
            "query_drift_max_step": float(max(steps)) if steps else 0.0,
            "query_len_ratio": len(queries[-1]) / max(len(original), 1),
            "query_novel_token_rate": (
                len(_tokens(queries[-1]) - _tokens(original)) / max(len(_tokens(queries[-1])), 1)
            ),
        }

    @staticmethod
    def _evidence_provenance(trace: Trace) -> dict[str, float]:
        spans = trace.retrieval_spans
        id_lists = [list(s.attributes.get("retrieval.document_ids", [])) for s in spans]
        seen: set[str] = set()
        per_iter_new: list[int] = []
        repeats = 0
        total = 0
        for ids in id_lists:
            new = [d for d in ids if d not in seen]
            per_iter_new.append(len(new))
            repeats += len(ids) - len(new)
            total += len(ids)
            seen.update(ids)

        inspect_spans = [s for s in trace.tool_spans if s.attributes.get("gen_ai.tool.name") == "get_document_by_id"]
        ranks = [float(s.attributes.get("tool.rank", 0)) for s in inspect_spans]

        # What the agent gathered against what it actually showed the generator. These
        # differ only when the agent iterated, and the gap is a behavioural signal in its
        # own right: a run that searched three times and then used one round's worth of
        # evidence looks different from one that searched once.
        n_retrieved = len(seen)
        context = trace.context_doc_ids
        n_context = len(context)
        shown_late = 0
        if per_iter_new and len(id_lists) > 1 and context:
            first_round = set(id_lists[0])
            shown_late = sum(1 for d in context if d not in first_round)

        return {
            "n_context_docs": float(n_context),
            "n_retrieved_docs": float(n_retrieved),
            "context_to_retrieved_ratio": n_context / max(n_retrieved, 1),
            "context_from_later_rounds": shown_late / max(n_context, 1),
            "doc_source_entropy": _entropy(per_iter_new),
            "repeat_doc_ratio": repeats / max(total, 1),
            "late_context_fraction": (per_iter_new[-1] / max(n_retrieved, 1)) if per_iter_new else 0.0,
            "n_inspections": float(len(inspect_spans)),
            "inspect_rank_mean": float(np.mean(ranks)) if ranks else 0.0,
        }

    def _control_flow(self, trace: Trace) -> dict[str, float]:
        tool_names = [s.attributes.get("gen_ai.tool.name", "") for s in trace.tool_spans]
        verdicts = [
            s.attributes.get("argus.decision", "")
            for s in trace.mechanism_spans
            if s.attributes.get("argus.mechanism") == "M4_reflect"
        ]
        insufficient = sum(1 for v in verdicts if v == "INSUFFICIENT")
        plan_delta = sum(1 for a, b in zip(verdicts, verdicts[1:]) if a != b)

        stops = [
            s for s in trace.mechanism_spans
            if s.attributes.get("argus.mechanism") == "M2_iterate"
            and s.attributes.get("argus.decision") == "stop"
        ]
        stopped_early = any(
            s.attributes.get("argus.reason") == "reflection_satisfied" for s in stops
        )

        bigrams = self._tool_bigrams(trace)
        novelty = 0.0
        if bigrams and self._fitted:
            rare = sum(
                1 for bg in bigrams if self._bigram_freq.get(bg, 0) / self._n_fit_traces < 0.05
            )
            novelty = rare / len(bigrams)

        return {
            "n_iterations": float(trace.n_iterations),
            "trajectory_length": float(len(trace.spans)),
            "n_tool_calls": float(len(tool_names)),
            "tool_seq_entropy": _entropy(Counter(tool_names).values()),
            "tool_bigram_novelty": novelty,
            "n_reflections": float(len(verdicts)),
            "reflect_insufficient_rate": insufficient / max(len(verdicts), 1),
            "plan_delta": float(plan_delta),
            "stopped_early": 1.0 if stopped_early else 0.0,
        }

    def _output_behaviour(self, trace: Trace) -> dict[str, float]:
        answer = trace.final_answer or ""
        low = answer.lower()
        return {
            "answer_len": float(len(answer)),
            "answer_len_zscore": (len(answer) - self._answer_len_mean) / self._answer_len_std,
            "answer_token_entropy": _char_entropy(answer),
            "answer_is_refusal": 1.0 if any(r in low for r in _REFUSAL) else 0.0,
            "n_llm_calls": float(len(trace.inference_spans)),
        }

    @staticmethod
    def _resource(trace: Trace) -> dict[str, float]:
        n_calls = max(len(trace.inference_spans), 1)
        latencies = [s.duration_ms for s in trace.spans]
        return {
            "total_input_tokens": float(trace.total_input_tokens),
            "total_output_tokens": float(trace.total_output_tokens),
            "tokens_per_step": trace.total_input_tokens / n_calls,
            "total_latency_ms": float(trace.total_latency_ms),
            "latency_per_step": trace.total_latency_ms / max(len(trace.spans), 1),
            "latency_std": float(np.std(latencies)) if latencies else 0.0,
        }

    @staticmethod
    def _tool_bigrams(trace: Trace) -> list[str]:
        seq = [
            s.name.split("[")[0] if "[" in s.name else s.name
            for s in trace.spans
            if s.kind in (SpanKind.TOOL, SpanKind.RETRIEVAL, SpanKind.INFERENCE)
        ]
        return [f"{a}>{b}" for a, b in zip(seq, seq[1:])]


def extract_features(traces: Iterable[Trace], fit: bool = True) -> tuple[Any, Any]:
    """Convenience wrapper returning (features DataFrame, metadata DataFrame)."""
    import pandas as pd

    traces = list(traces)
    extractor = FeatureExtractor()
    if fit:
        extractor.fit(traces)
    rows, meta = extractor.extract_many(traces)
    return pd.DataFrame(rows), pd.DataFrame(meta)
