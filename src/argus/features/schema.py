"""Feature schema.

Every feature is documented here with its family and rationale, for two reasons. The
write-up needs to justify each one, and the feature-ablation experiment needs to group
them by family to show that no single family carries the whole detector.

The hard constraint on this whole module: nothing may read model weights, activations or
logits, and nothing may require re-invoking the generator. Every feature must be
computable from the OpenTelemetry spans a normal deployment already emits.
"""

from __future__ import annotations

FEATURE_FAMILIES: dict[str, dict[str, str]] = {
    "retrieval_dynamics": {
        "retr_score_mean": "Mean retrieval score across all retrieved documents",
        "retr_score_std": "Standard deviation of retrieval scores",
        "retr_score_max": "Highest retrieval score seen",
        "retr_score_top_gap": "Gap between the best and second-best score in the first retrieval",
        "retr_score_range": "Gap between best and worst score in the first retrieval",
        "retr_score_drift": "Change in mean score from the first to the last iteration",
        "retr_score_skew": "Skew of the score distribution; a single optimised passage lifts one tail",
        "retr_rank_churn": "Fraction of top-k documents that changed between successive iterations",
    },
    "query_trajectory": {
        "n_rewrites": "How many times the query was rewritten",
        "query_drift_first": "Token Jaccard distance between the original and the first rewrite",
        "query_drift_total": "Distance between the original query and the final query",
        "query_drift_max_step": "Largest single-step drift between consecutive queries",
        "query_len_ratio": "Length of the final query relative to the original",
        "query_novel_token_rate": "Share of tokens in the final query absent from the original",
    },
    "evidence_provenance": {
        "n_context_docs": "Documents actually rendered into the answer prompt",
        "n_retrieved_docs": "Distinct documents returned by any retrieval, shown or not",
        "context_to_retrieved_ratio": "Share of gathered evidence that reached the prompt; below 1 only when the agent iterated",
        "context_from_later_rounds": "Share of the prompt contributed by rounds after the first",
        "doc_source_entropy": "Entropy over the retrieval iteration each document came from",
        "repeat_doc_ratio": "Share of retrieved documents already seen in an earlier iteration",
        "late_context_fraction": "Share of retrieved evidence first introduced in the final iteration",
        "n_inspections": "How many documents were opened in full",
        "inspect_rank_mean": "Mean rank of inspected documents; inspecting low-ranked documents is unusual",
    },
    "control_flow": {
        "n_iterations": "Number of retrieval rounds",
        "trajectory_length": "Total number of spans in the trace",
        "n_tool_calls": "Number of tool invocations",
        "tool_seq_entropy": "Entropy of the tool-call sequence",
        "tool_bigram_novelty": "Share of tool-call bigrams that are rare across the corpus of traces",
        "n_reflections": "How many reflection steps ran",
        "reflect_insufficient_rate": "Share of reflections that judged the evidence insufficient",
        "plan_delta": "Number of changes in the reflection verdict sequence",
        "stopped_early": "Whether the agent stopped before exhausting its budget",
    },
    "output_behaviour": {
        "answer_len": "Length of the final answer in characters",
        "answer_len_zscore": "Answer length relative to the corpus mean, in standard deviations",
        "answer_token_entropy": "Character-level entropy of the answer",
        "answer_is_refusal": "Whether the model declined to answer",
        "n_llm_calls": "Number of generator invocations",
    },
    "resource": {
        "total_input_tokens": "Total prompt tokens across the run",
        "total_output_tokens": "Total completion tokens across the run",
        "tokens_per_step": "Mean prompt tokens per generator call",
        "total_latency_ms": "End to end wall-clock time",
        "latency_per_step": "Mean latency per span",
        "latency_std": "Variability of per-span latency",
    },
}


def describe_features() -> list[dict[str, str]]:
    return [
        {"family": family, "feature": name, "description": desc}
        for family, feats in FEATURE_FAMILIES.items()
        for name, desc in feats.items()
    ]


def feature_names() -> list[str]:
    return [name for feats in FEATURE_FAMILIES.values() for name in feats]


def family_of(feature: str) -> str:
    for family, feats in FEATURE_FAMILIES.items():
        if feature in feats:
            return family
    return "unknown"
