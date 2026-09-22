"""Argus dashboard.

An interactive explorer for experiment results and execution traces.

    streamlit run dashboard/app.py

Four views:

* Overview          what has been run, and the headline numbers
* Study A           mechanism ablation and the stage decomposition
* Study B           detector performance and feature importance
* Trace explorer    step through a single agent run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argus.analysis.ablation import (  # noqa: E402
    budget_curve,
    cost_table,
    load_traces_frame,
    mechanism_effects,
    poison_ratio_curve,
    results_table,
)
from argus.analysis.stage import stage_attribution, stage_decomposition, verdict  # noqa: E402
from argus.config import load_config  # noqa: E402
from argus.features.extractor import FeatureExtractor  # noqa: E402
from argus.features.schema import describe_features  # noqa: E402
from argus.telemetry.writer import TraceReader  # noqa: E402

st.set_page_config(page_title="Argus | Agentic RAG Security", layout="wide")


@st.cache_data(show_spinner=False)
def _load_traces_frame(traces_dir: str, _stamp: float) -> pd.DataFrame:
    return load_traces_frame(traces_dir)


@st.cache_resource(show_spinner=False)
def _load_traces(traces_dir: str, _stamp: float):
    return list(TraceReader.read_dir(traces_dir))


def _stamp(path: Path) -> float:
    files = list(path.glob("*.jsonl*"))
    return max((f.stat().st_mtime for f in files), default=0.0)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Argus")
st.sidebar.caption("Agentic RAG security research platform")

data_dir = st.sidebar.text_input("Data directory", value="data")
cfg = load_config()
cfg.data_dir = Path(data_dir)

if not cfg.traces_dir.exists() or not list(cfg.traces_dir.glob("*.jsonl*")):
    st.title("Argus")
    st.warning(f"No traces found under `{cfg.traces_dir}`.")
    st.code("python scripts/demo.py            # offline demo, ~30 seconds\n"
            "argus run grid --preset pilot     # larger offline rehearsal", language="bash")
    st.stop()

stamp = _stamp(cfg.traces_dir)
traces_df = _load_traces_frame(str(cfg.traces_dir), stamp)

view = st.sidebar.radio(
    "View", ["Overview", "Study A: mechanisms", "Study B: detection", "Trace explorer"]
)

backends = set(traces_df["llm_backend"].unique())
if backends <= {"mock"}:
    st.sidebar.warning(
        "These traces come from the mock backend. They exercise the pipeline "
        "but are not research findings."
    )

st.sidebar.divider()
st.sidebar.metric("Agent runs", len(traces_df))
st.sidebar.metric("Configurations", traces_df["config_id"].nunique())
st.sidebar.metric("Attacks", traces_df[traces_df["attack"] != "none"]["attack"].nunique())


# ----------------------------------------------------------------- overview
if view == "Overview":
    st.title("Overview")

    attacked = traces_df[traces_df["attacked"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agent runs", len(traces_df))
    c2.metric("Attacked runs", len(attacked))
    c3.metric(
        "Attack success",
        f"{attacked['attack_success'].mean():.1%}" if len(attacked) else "n/a",
    )
    c4.metric("Total cost", f"${traces_df['cost_usd'].sum():.4f}")

    st.subheader("What has been run")
    st.dataframe(
        traces_df.groupby(["dataset", "retriever", "config_id", "attack"])
        .size()
        .reset_index(name="runs"),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Attack success by configuration")
    table = results_table(traces_df)
    if not table.empty:
        st.bar_chart(table.set_index("label")[["asr", "poison_retrieval_rate"]])

    st.subheader("Security against cost")
    costs = cost_table(traces_df)
    cols = ["label", "asr", "clean_accuracy", "mean_input_tokens",
            "token_overhead_x", "asr_reduction", "mean_latency_ms"]
    st.dataframe(
        costs[[c for c in cols if c in costs.columns]].round(4),
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------------------------------ study A
elif view == "Study A: mechanisms":
    st.title("Study A: which mechanism matters, and at which stage")

    st.header("RQ1: per-mechanism effect")
    st.caption("Every configuration compared against C0 vanilla on the same queries.")
    effects = mechanism_effects(traces_df)
    st.dataframe(
        effects[
            ["label", "n", "attack_success", "diff", "diff_ci_low", "diff_ci_high",
             "cohens_h", "effect", "p_value", "significant_fdr"]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.header("RQ2: retrieval stage or reasoning stage")
    st.latex(
        r"P(\text{success}) = P(\text{poison in context}) \times "
        r"P(\text{misled} \mid \text{poison in context})"
    )
    decomp = stage_decomposition(traces_df)
    st.dataframe(
        decomp[
            ["label", "n", "p_retrieval_stage", "p_reasoning_stage",
             "p_predicted", "p_observed", "decomposition_residual",
             "asr_without_poison_in_context"]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )

    if not decomp.empty:
        st.bar_chart(decomp.set_index("label")[["p_retrieval_stage", "p_reasoning_stage"]])

    attrib = stage_attribution(traces_df)
    st.dataframe(
        attrib[
            ["label", "delta_retrieval_stage", "delta_reasoning_stage",
             "share_from_reasoning", "attribution"]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )
    st.success(verdict(attrib))

    st.header("RQ3: iteration budget")
    budget = budget_curve(traces_df)
    if len(budget) > 1:
        st.line_chart(
            budget.pivot(index="iteration_budget", columns="config_id", values="asr")
        )
    st.dataframe(budget.round(4), use_container_width=True, hide_index=True)

    st.header("Poison ratio")
    ratios = poison_ratio_curve(traces_df)
    if len(ratios) > 1:
        st.line_chart(ratios.pivot(index="n_poison", columns="config_id", values="asr"))


# ------------------------------------------------------------------ study B
elif view == "Study B: detection":
    st.title("Study B: detecting compromise from execution traces")
    st.caption("No model internals. No generator re-execution.")

    traces = _load_traces(str(cfg.traces_dir), stamp)
    rows, meta = FeatureExtractor().fit(traces).extract_many(traces)
    X, M = pd.DataFrame(rows), pd.DataFrame(meta)

    c1, c2, c3 = st.columns(3)
    c1.metric("Traces", len(X))
    c2.metric("Features", X.shape[1])
    c3.metric("Compromised", f"{int(M['y'].sum())} / {len(M)}")

    st.header("Detection, held out on an unseen attack")
    st.caption(
        "Leave-one-attack-out is the headline metric. Train on two attacks, test on a "
        "third the detector has never seen."
    )

    if M["y"].nunique() < 2 or M["attack"].nunique() < 3:
        st.warning("Need at least two attacks and both classes present. Run a larger grid.")
    else:
        from argus.detect.evaluate import feature_ablation, leave_one_attack_out

        results = []
        for name in ("rules", "iforest", "gbdt"):
            try:
                res = leave_one_attack_out(name, X, M, seed=cfg.seed)
            except ValueError:
                continue
            results.append(
                {
                    "detector": name,
                    "roc_auc": res.roc_auc,
                    "f1": res.f1,
                    "precision": res.precision,
                    "recall": res.recall,
                    "fpr_at_95_tpr": res.fpr_at_95_tpr,
                    "overhead_x": res.inference_overhead_x,
                    "_full": res,
                }
            )

        if results:
            st.dataframe(
                pd.DataFrame([{k: v for k, v in r.items() if k != "_full"} for r in results]).round(4),
                use_container_width=True,
                hide_index=True,
            )

            best = results[-1]["_full"]
            st.subheader("Per held-out attack")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"held_out_attack": a, **{k: v for k, v in m.items() if isinstance(v, (int, float))}}
                        for a, m in best.per_split.items()
                    ]
                ).round(4),
                use_container_width=True,
                hide_index=True,
            )

            if best.feature_importance:
                st.subheader("Feature importance")
                imp = pd.DataFrame(
                    list(best.feature_importance.items())[:15],
                    columns=["feature", "importance"],
                )
                st.bar_chart(imp.set_index("feature"))

            if st.checkbox("Run feature family ablation (slower)"):
                st.dataframe(
                    feature_ablation("gbdt", X, M, seed=cfg.seed).round(4),
                    use_container_width=True,
                    hide_index=True,
                )

    with st.expander("Feature dictionary"):
        st.dataframe(pd.DataFrame(describe_features()), use_container_width=True, hide_index=True)


# ----------------------------------------------------------- trace explorer
else:
    st.title("Trace explorer")

    traces = _load_traces(str(cfg.traces_dir), stamp)

    col1, col2, col3 = st.columns(3)
    cfg_filter = col1.selectbox("Configuration", ["all"] + sorted({t.config_id for t in traces}))
    atk_filter = col2.selectbox("Attack", ["all"] + sorted({t.attack for t in traces}))
    lbl_filter = col3.selectbox("Label", ["all", "compromised", "benign"])

    subset = [
        t for t in traces
        if (cfg_filter == "all" or t.config_id == cfg_filter)
        and (atk_filter == "all" or t.attack == atk_filter)
        and (lbl_filter == "all" or t.label == lbl_filter)
    ]

    if not subset:
        st.info("No traces match those filters.")
        st.stop()

    st.caption(f"{len(subset)} matching traces")
    idx = st.slider("Trace", 0, len(subset) - 1, 0) if len(subset) > 1 else 0
    trace = subset[idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Label", trace.label)
    c2.metric("Attack success", "yes" if trace.attack_success else "no")
    c3.metric("Correct", "yes" if trace.answered_correctly else "no")
    c4.metric("Iterations", trace.n_iterations)

    st.subheader("Question and answers")
    st.write(f"**Question:** {trace.query}")
    a, b, c = st.columns(3)
    a.info(f"**Model answer**\n\n{trace.final_answer}")
    b.success(f"**Gold**\n\n{trace.gold_answer}")
    c.error(f"**Attacker target**\n\n{trace.target_answer}")

    st.subheader("Execution trace")
    poison = set(trace.poison_doc_ids)
    rows = []
    for i, span in enumerate(trace.spans):
        detail = ""
        if span.kind.value == "retrieval.query":
            ids = span.attributes.get("retrieval.document_ids", [])
            n_poison = sum(1 for d in ids if d in poison)
            detail = f"query={span.attributes.get('retrieval.query', '')[:60]!r}  docs={len(ids)}"
            if n_poison:
                detail += f"   POISONED x{n_poison}"
        elif span.kind.value == "gen_ai.chat":
            detail = (
                f"task={span.attributes.get('argus.task')}  "
                f"-> {span.attributes.get('argus.response_preview', '')[:60]!r}"
            )
        elif span.kind.value == "argus.mechanism":
            detail = f"{span.attributes.get('argus.mechanism')} = {span.attributes.get('argus.decision')}"
        elif span.kind.value == "gen_ai.execute_tool":
            doc_id = span.attributes.get("tool.document_id", "")
            detail = f"{span.attributes.get('gen_ai.tool.name')}({doc_id})"
            if doc_id in poison:
                detail += "   POISONED"
        rows.append(
            {"#": i, "span": span.name, "kind": span.kind.value,
             "ms": round(span.duration_ms, 1), "detail": detail}
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Extracted features")
    feats = FeatureExtractor().fit(traces).extract(trace)
    st.dataframe(
        pd.DataFrame(sorted(feats.items()), columns=["feature", "value"]).round(4),
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    with st.expander("Raw trace JSON"):
        st.json(json.loads(trace.to_json()))
