"""Command line interface.

    argus corpus build --dataset synthetic
    argus attack run --attack poisonedrag_black
    argus run single --config C5
    argus run grid --preset pilot
    argus analyse ablation
    argus analyse stage
    argus features build
    argus detect train --model all
    argus detect evaluate --protocol loao
    argus baselines compare
    argus cost estimate --preset main
    argus dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from argus import __version__


def _cfg(args: argparse.Namespace):
    from dataclasses import replace

    from argus.config import load_config

    cfg = load_config(getattr(args, "config_file", None))
    if getattr(args, "data_dir", None):
        cfg.data_dir = Path(args.data_dir)
    # LLMConfig and RetrievalConfig are frozen dataclasses, so these overrides have to
    # rebuild them. Assigning through raised FrozenInstanceError, which meant --backend
    # and --retriever had never worked from the command line.
    if getattr(args, "backend", None):
        cfg.llm = replace(cfg.llm, backend=args.backend)
    if getattr(args, "retriever", None):
        cfg.retrieval = replace(cfg.retrieval, backend=args.retriever)
    cfg.ensure_dirs()
    return cfg


def _print(obj: Any) -> None:
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            print("(no rows)")
            return
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(obj.round(4).to_string(index=False))
    elif isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(obj)


def _warn_if_mock(cfg) -> None:
    if cfg.llm.backend == "mock":
        print(
            "\n  NOTE: running on the mock backend. Results exercise the pipeline but "
            "are not research findings.\n  Set ARGUS_LLM_BACKEND=openai in .env for "
            "real runs.\n",
            file=sys.stderr,
        )


# --------------------------------------------------------------------- corpus
def cmd_corpus_build(args: argparse.Namespace) -> int:
    from argus.corpus.loaders import build_corpus, recommended_n_docs
    from argus.corpus.validate import validate_corpus

    cfg = _cfg(args)

    if args.dataset == "synthetic":
        print(
            "\n  NOTE: the synthetic corpus is for development and tests. It is not a "
            "research corpus.\n  Use --dataset nq, squad or hotpotqa for anything you "
            "intend to report.\n",
            file=sys.stderr,
        )

    suggested = recommended_n_docs(args.n_queries)
    if args.dataset != "synthetic" and args.n_docs < suggested:
        print(
            f"\n  NOTE: --n-docs {args.n_docs} is small for {args.n_queries} queries.\n"
            f"  Attacks inject up to 10 documents per query, so poison would be a large "
            f"share of the corpus.\n  Recommended: --n-docs {suggested}\n",
            file=sys.stderr,
        )

    corpus = build_corpus(
        dataset=args.dataset,
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        seed=cfg.seed,
        cache_dir=cfg.data_dir / "cache",
        validate=not args.no_validate,
    )
    path = cfg.corpora_dir / f"{args.dataset}_{args.n_docs}_{args.n_queries}_{cfg.seed}.jsonl"
    corpus.save(path)
    _print({**corpus.summary(), "path": str(path)})

    # Always print the validation report, even when validation was skipped, so the state
    # of a corpus is visible at the moment it is created rather than discovered later.
    print()
    print(validate_corpus(corpus, strict=False).text())
    return 0


def cmd_corpus_info(args: argparse.Namespace) -> int:
    from argus.corpus.store import Corpus

    cfg = _cfg(args)
    paths = sorted(cfg.corpora_dir.glob("*.jsonl"))
    if not paths:
        print("No corpora yet. Run: argus corpus build")
        return 1
    for path in paths:
        _print({**Corpus.load(path).summary(), "path": path.name})
    return 0


# --------------------------------------------------------------------- attack
def cmd_attack_run(args: argparse.Namespace) -> int:
    from argus.attacks.registry import build_attack
    from argus.corpus.store import Corpus
    from argus.retrieval.factory import build_retriever

    cfg = _cfg(args)
    paths = sorted(cfg.corpora_dir.glob(f"{args.dataset}_*.jsonl"))
    if not paths:
        print(f"No corpus for '{args.dataset}'. Run: argus corpus build --dataset {args.dataset}")
        return 1

    corpus = Corpus.load(paths[-1])
    attack = build_attack(args.attack, n_poison_docs=args.ratio, seed=cfg.seed)
    retriever = None
    if attack.needs_retriever:
        retriever = build_retriever(corpus, cfg.retrieval, cfg.data_dir / "cache")
        retriever.build()

    result = attack.apply(corpus, retriever=retriever)
    out = cfg.corpora_dir / f"{result.corpus.name}.jsonl"
    result.corpus.save(out)
    _print({**result.summary(), "path": str(out)})
    return 0


def cmd_attack_list(args: argparse.Namespace) -> int:
    from argus.attacks.registry import ATTACKS, list_attacks

    for name in list_attacks():
        cls = ATTACKS[name]
        setting = "white box" if cls.needs_retriever else "black box"
        print(f"  {name:<22s} {setting:<11s} {(cls.__doc__ or '').strip().splitlines()[0]}")
    return 0


# ------------------------------------------------------------------------ run
def cmd_run_single(args: argparse.Namespace) -> int:
    from argus.config import RunConfig
    from argus.runner.experiment import ExperimentRunner

    cfg = _cfg(args)
    _warn_if_mock(cfg)
    runner = ExperimentRunner(
        cfg, progress=lambda m: print(m, flush=True), workers=args.workers
    )
    run = RunConfig(
        config_id=args.config,
        dataset=args.dataset,
        attack=args.attack,
        n_poison_docs=args.ratio,
        n_queries=args.n_queries,
        retriever=cfg.retrieval.backend,
        iteration_budget=args.budget,
        seed=cfg.seed,
    )
    outcome = runner.run_cell(run, n_docs=args.n_docs, overwrite=args.overwrite)
    _print(
        {
            k: v
            for k, v in outcome.to_dict().items()
            if k not in ("run_config", "meta")
        }
    )
    return 0


def cmd_run_grid(args: argparse.Namespace) -> int:
    from argus.runner.cost import estimate_grid_cost
    from argus.runner.experiment import ExperimentRunner
    from argus.runner.grid import build_grid, grid_summary

    cfg = _cfg(args)
    _warn_if_mock(cfg)
    cells = build_grid(args.preset, seed=cfg.seed)
    if args.limit:
        cells = cells[: args.limit]

    summary = grid_summary(cells)
    print("\nGrid:")
    _print(summary)

    if cfg.llm.backend != "mock":
        est = estimate_grid_cost(cells, cfg.llm.model)
        print("\nCost estimate:")
        _print(est)
        # Cost is a reported result, not a design constraint: reporting robustness and
        # its inference bill together is one of the gaps this work identifies. So an
        # estimate over budget is a warning, not a refusal — the previous hard stop could
        # silently prevent a scientifically necessary run.
        if est["estimated_cost_usd"] > cfg.budget_usd:
            print(
                f"\n  NOTE: estimate ${est['estimated_cost_usd']:.2f} exceeds the "
                f"configured ceiling of ${cfg.budget_usd:.2f}.\n"
                "  The run will proceed. Raise ARGUS_BUDGET_USD to silence this, or set "
                "it to 0 to disable the mid-run stop entirely.\n"
                "  On a local endpoint (Ollama, vLLM) the dollar figure is notional.\n"
            )
        if not args.yes:
            reply = input(f"\nProceed with ~${est['estimated_cost_usd']:.2f}? [y/N] ")
            if reply.strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return 1

    runner = ExperimentRunner(
        cfg, progress=lambda m: print(m, flush=True), workers=args.workers
    )
    if runner.workers > 1:
        print(f"Running {runner.workers} queries concurrently per cell.")
    n_docs = args.n_docs or _preset_n_docs(args.preset)
    t0 = time.time()
    for i, cell in enumerate(cells, 1):
        elapsed = time.time() - t0
        rate = elapsed / max(i - 1, 1)
        eta = rate * (len(cells) - i + 1)
        print(
            f"[{i}/{len(cells)}] {cell.cell_id}"
            + (f"   (elapsed {elapsed / 3600:.1f}h, eta {eta / 3600:.1f}h)" if i > 1 else "")
        )
        runner.run_cell(cell, n_docs=n_docs, overwrite=args.overwrite)

    print(f"\nDone in {(time.time() - t0) / 3600:.2f}h.")
    print(f"Traces in {cfg.traces_dir}, results in {cfg.results_dir}")
    print(f"Actual spend: ${runner.llm.usage.cost_usd:.4f}")
    print("\nNext: argus validate data   (confirms every cell wrote what its result claims)")
    return 0


def _preset_n_docs(preset: str) -> int:
    """Corpus size a preset expects, so --n-docs need not be repeated on every command."""
    from argus.runner.grid import GRID_PRESETS

    return int(GRID_PRESETS.get(preset, {}).get("n_docs", 2000))


# -------------------------------------------------------------------- analyse
def _load_traces(cfg):
    from argus.analysis.ablation import load_traces_frame

    frame = load_traces_frame(cfg.traces_dir)
    if frame.empty:
        print("No traces yet. Run: argus run grid --preset pilot")
        raise SystemExit(1)
    return frame


def cmd_analyse_ablation(args: argparse.Namespace) -> int:
    from argus.analysis.ablation import (
        abstention_table,
        budget_curve,
        cost_table,
        mechanism_effects,
        poison_ratio_curve,
        restrict_to_matched_queries,
        results_table,
    )

    cfg = _cfg(args)
    traces = _load_traces(cfg)

    if traces.attrs.get("corrupt_lines"):
        print(
            f"\n  WARNING: {traces.attrs['corrupt_lines']} corrupt trace line(s) were "
            "skipped. Re-run the affected cells before quoting these numbers.\n",
            file=sys.stderr,
        )

    if not args.no_matched:
        before = len(traces)
        traces = restrict_to_matched_queries(traces)
        dropped = before - len(traces)
        if dropped:
            print(
                f"\nRestricted to matched query sets: dropped {dropped} of {before} runs "
                "so every configuration is compared on identical queries.\n"
                "(Pass --no-matched to analyse the raw, unmatched data.)"
            )

    print("\n=== Per-configuration results ===")
    _print(results_table(traces))

    print("\n=== Mechanism effects versus C0 vanilla ===")
    effects = mechanism_effects(traces)
    _print(effects)

    print("\n=== Abstention: protection or avoidance? ===")
    print("A mechanism that stops answering scores a low attack success rate too.\n")
    abst = abstention_table(traces)
    _print(abst)

    print("\n=== Iteration budget curve (RQ3) ===")
    _print(budget_curve(traces))

    print("\n=== Poison ratio curve ===")
    _print(poison_ratio_curve(traces))

    print("\n=== Security against cost ===")
    _print(cost_table(traces))

    if args.save:
        out = cfg.results_dir / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        results_table(traces).to_csv(out / "results_table.csv", index=False)
        effects.to_csv(out / "mechanism_effects.csv", index=False)
        abst.to_csv(out / "abstention_table.csv", index=False)
        budget_curve(traces).to_csv(out / "budget_curve.csv", index=False)
        poison_ratio_curve(traces).to_csv(out / "poison_ratio_curve.csv", index=False)
        cost_table(traces).to_csv(out / "cost_table.csv", index=False)
        print(f"\nSaved to {out}")
    return 0


def cmd_analyse_stage(args: argparse.Namespace) -> int:
    from argus.analysis.ablation import restrict_to_matched_queries
    from argus.analysis.stage import stage_attribution, stage_decomposition, verdict

    cfg = _cfg(args)
    traces = _load_traces(cfg)
    if not args.no_matched:
        traces = restrict_to_matched_queries(traces)

    print("\n=== Stage decomposition ===")
    print("P(success) = P(poison in context) x P(misled | poison in context)\n")
    decomp = stage_decomposition(traces)
    _print(decomp)

    print("\n=== Stage attribution versus baseline ===")
    attrib = stage_attribution(traces)
    _print(attrib)

    print("\n=== Verdict ===")
    print(verdict(attrib))

    if args.save:
        out = cfg.results_dir / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        decomp.to_csv(out / "stage_decomposition.csv", index=False)
        attrib.to_csv(out / "stage_attribution.csv", index=False)
        (out / "stage_verdict.txt").write_text(verdict(attrib))
        print(f"\nSaved to {out}")
    return 0


# ------------------------------------------------------------------- features
def cmd_features_build(args: argparse.Namespace) -> int:
    import pandas as pd

    from argus.features.extractor import FeatureExtractor
    from argus.telemetry.writer import TraceReader

    cfg = _cfg(args)
    traces = list(TraceReader.read_dir(cfg.traces_dir))
    if not traces:
        print("No traces yet. Run: argus run grid --preset pilot")
        return 1

    extractor = FeatureExtractor().fit(traces)
    rows, meta = extractor.extract_many(traces)
    X, M = pd.DataFrame(rows), pd.DataFrame(meta)

    X.to_parquet(cfg.features_dir / "features.parquet") if args.parquet else X.to_csv(
        cfg.features_dir / "features.csv", index=False
    )
    M.to_csv(cfg.features_dir / "meta.csv", index=False)

    print(f"Extracted {len(X)} traces x {X.shape[1]} features -> {cfg.features_dir}")
    print(f"  compromised: {int(M['y'].sum())}   benign: {int((M['y'] == 0).sum())}")
    print(f"  attacks: {sorted(M['attack'].unique())}")
    return 0


def cmd_features_describe(args: argparse.Namespace) -> int:
    import pandas as pd

    from argus.features.schema import describe_features

    _print(pd.DataFrame(describe_features()))
    return 0


# --------------------------------------------------------------------- detect
def _load_features(cfg):
    import pandas as pd

    fx = cfg.features_dir / "features.csv"
    fp = cfg.features_dir / "features.parquet"
    if fp.exists():
        X = pd.read_parquet(fp)
    elif fx.exists():
        X = pd.read_csv(fx)
    else:
        print("No features yet. Run: argus features build")
        raise SystemExit(1)
    meta = pd.read_csv(cfg.features_dir / "meta.csv")

    # Development-only datasets are not evidence. The same smoke-test cell that gave C5 a
    # different query composition from every other configuration in the ablation also
    # sits in the feature matrix, so it is dropped here for the same reason.
    from argus.analysis.ablation import DEV_DATASETS

    if "dataset" in meta.columns:
        keep = ~meta["dataset"].isin(DEV_DATASETS)
        if not keep.all():
            print(
                f"  dropped {int((~keep).sum())} development-dataset rows "
                f"({sorted(set(meta.loc[~keep, 'dataset']))}) before evaluation"
            )
            X, meta = X[keep.to_numpy()].reset_index(drop=True), meta[keep].reset_index(drop=True)
    return X, meta


def cmd_detect_train(args: argparse.Namespace) -> int:
    from argus.detect.models import build_detector

    cfg = _cfg(args)
    X, meta = _load_features(cfg)
    models = ["rules", "iforest", "gbdt"] if args.model == "all" else [args.model]

    for name in models:
        det = build_detector(name, feature_names=list(X.columns), seed=cfg.seed)
        det.fit(X.to_numpy(), meta["y"].to_numpy())
        path = cfg.models_dir / f"{name}.pkl"
        det.save(path)
        print(f"trained {name:<10s} -> {path}")
    return 0


def cmd_detect_evaluate(args: argparse.Namespace) -> int:
    import pandas as pd

    from argus.detect.evaluate import (
        cross_dataset_evaluation,
        evaluate_detector,
        feature_ablation,
        leave_one_attack_out,
    )

    cfg = _cfg(args)
    X, meta = _load_features(cfg)
    models = ["rules", "iforest", "gbdt"] if args.model == "all" else [args.model]

    rows = []
    for name in models:
        try:
            if args.protocol == "loao":
                res = leave_one_attack_out(name, X, meta, seed=cfg.seed)
            elif args.protocol == "cross_dataset":
                res = cross_dataset_evaluation(
                    name, X, meta, args.train_dataset, args.test_dataset
                )
            else:
                res = evaluate_detector(name, X, meta, seed=cfg.seed)
        except ValueError as exc:
            print(f"  {name}: skipped ({exc})")
            continue
        print(res.summary_line())
        rows.append(res.to_dict())

    if not rows:
        return 1

    if args.protocol == "loao":
        print("\nPer held-out attack:")
        for row in rows:
            print(f"\n  {row['detector']}:")
            for attack, m in row["per_split"].items():
                if "roc_auc" in m:
                    print(
                        f"    {attack:<22s} AUC={m['roc_auc']:.3f}  F1={m['f1']:.3f}  "
                        f"FPR@95TPR={m['fpr_at_95_tpr']:.3f}"
                    )

    top = rows[-1]
    if top.get("feature_importance"):
        print("\nTop features:")
        for feat, imp in list(top["feature_importance"].items())[:12]:
            print(f"    {feat:<28s} {imp:.4f}")

    if args.ablation:
        print("\n=== Feature family ablation ===")
        _print(feature_ablation(models[-1], X, meta, seed=cfg.seed))

    if args.save:
        out = cfg.results_dir / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows]
        ).to_csv(out / f"detect_{args.protocol}.csv", index=False)
        (out / f"detect_{args.protocol}_full.json").write_text(json.dumps(rows, indent=2))
        print(f"\nSaved to {out}")
    return 0


# ------------------------------------------------------------------ baselines
def cmd_baselines_compare(args: argparse.Namespace) -> int:
    """Compare the trace detector against published content-based defences.

    This command used to require a poisoned corpus already on disk, and nothing in the
    pipeline ever wrote one: the grid runner builds poisoned corpora in memory and
    caches them there. So `argus baselines compare` failed with "No poisoned corpus
    found" every time it was run, `baseline_comparison.csv` was never produced, and the
    central claim of Study B — comparable detection at a fraction of the inference cost —
    shipped with no supporting evidence.

    It now reconstructs the poisoned corpus the traces were produced against, from the
    clean corpus on disk plus the attack recorded in the traces themselves, so it works
    directly after a grid run with no extra step.
    """
    import collections

    import pandas as pd

    from argus.attacks.registry import build_attack
    from argus.baselines.llm_judge import LLMJudgeDefense
    from argus.baselines.loo_counterfactual import LOOCounterfactualDefense
    from argus.baselines.perplexity import PerplexityFilterDefense
    from argus.corpus.store import Corpus
    from argus.detect.evaluate import leave_one_attack_out
    from argus.llm.factory import build_llm
    from argus.retrieval.factory import build_retriever
    from argus.telemetry.writer import TraceReader

    cfg = _cfg(args)
    _warn_if_mock(cfg)

    all_traces = [t for t in TraceReader.read_dir(cfg.traces_dir) if t.attack != "none"]
    if not all_traces:
        print("No attacked traces. Run: argus run grid --preset pilot")
        return 1

    # Baselines score retrieved passages, so every trace in one comparison must come from
    # the same poisoned corpus. Pick the largest such group unless one is named.
    groups: dict[tuple, list] = collections.defaultdict(list)
    for t in all_traces:
        groups[(t.dataset, t.retriever, t.attack, t.n_poison_in_corpus)].append(t)

    if args.attack:
        groups = {k: v for k, v in groups.items() if k[2] == args.attack}
        if not groups:
            print(f"No traces for attack '{args.attack}'.")
            return 1

    # Pick a group that can actually pose the question. Taking simply the largest group
    # picked nq/bm25/poisonedrag_white/p10, where the attack is so saturated that 5,381
    # of 5,500 runs are compromised: the first 300 taken were compromised without
    # exception, every negative then came from a clean run, and "compromised or not"
    # became "poisoned corpus or not". A detector that only knows whether the corpus was
    # poisoned scores a perfect AUC on that sample, and the trace detector duly did.
    #
    # A group carrying attacked-but-uncompromised runs of its own asks the real question:
    # among runs of the same attack, did the poison reach this one's prompt?
    want_neg = max(args.limit // 2, 10)

    def _negatives_in(items):
        return [t for t in items if not t.is_compromised]

    # Prefer a group that can fill the whole negative quota from its own attacked runs,
    # so no negative is merely "a run against a clean corpus". Size breaks the tie.
    usable = {k: v for k, v in groups.items() if len(_negatives_in(v)) >= want_neg}
    degenerate = not usable
    key = max(usable or groups, key=lambda k: len(groups[k]))
    dataset, retriever_name, attack_name, n_poison = key

    group = groups[key]
    positives = [t for t in group if t.is_compromised][: args.limit]

    # Negatives first from runs of this same attack whose poison never reached the
    # prompt, topped up from clean runs. Both are genuine negatives; only the first kind
    # forces a defence to look at the run rather than at the corpus.
    attacked_neg = _negatives_in(group)[:want_neg]
    clean_traces = [
        t
        for t in TraceReader.read_dir(cfg.traces_dir, pattern="clean__*.jsonl*")
        if t.dataset == dataset and t.retriever == retriever_name
    ][: want_neg - len(attacked_neg)]
    traces = positives + attacked_neg + clean_traces

    n_pos = len(positives)
    print(
        f"\nComparing on {len(traces)} traces from "
        f"{dataset}/{retriever_name}/{attack_name}/p{n_poison}\n"
        f"  {n_pos} compromised, {len(traces) - n_pos} benign "
        f"({len(attacked_neg)} attacked but uncompromised, {len(clean_traces)} clean)"
    )
    if degenerate:
        print(
            f"  WARNING: no attack group has {want_neg} attacked-but-uncompromised runs, "
            "so the negatives here are mostly clean runs and the task slides towards "
            "'was this corpus poisoned' rather than 'did poison reach this prompt'. "
            "Read every number in this table as an upper bound."
        )
    if n_pos == 0 or n_pos == len(traces):
        print(
            "  WARNING: only one class present; ROC-AUC will be undefined. "
            "Run a grid that produces both compromised and clean traces."
        )

    # Rebuild the exact corpus those traces ran against.
    candidates = sorted(cfg.corpora_dir.glob(f"{dataset}_*.jsonl"))
    if not candidates:
        print(
            f"No clean corpus for '{dataset}' in {cfg.corpora_dir}.\n"
            f"Run: argus corpus build --dataset {dataset}"
        )
        return 1
    clean = Corpus.load(candidates[-1])

    attack = build_attack(attack_name, n_poison_docs=n_poison, seed=cfg.seed)
    retr = None
    if attack.needs_retriever:
        retr = build_retriever(clean, cfg.retrieval, cfg.data_dir / "cache")
        retr.build()
    corpus = attack.apply(clean, retriever=retr).corpus

    llm = build_llm(cfg.llm, seed=cfg.seed)
    rows = [
        PerplexityFilterDefense(corpus).evaluate(traces),
        LLMJudgeDefense(llm, corpus).evaluate(traces),
        LOOCounterfactualDefense(llm, corpus).evaluate(traces),
    ]

    METRIC_KEYS = ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "fpr_at_95_tpr")
    try:
        from argus.detect.evaluate import (
            TRACE_DETECTOR_OVERHEAD_X,
            score_trace_subset_loao,
        )

        X, meta = _load_features(cfg)

        # The row that belongs beside the baselines: same traces, same labels, attack
        # family still held out of training. Without it the table compares 450 traces of
        # one attack against 50,449 of three under a different protocol.
        try:
            matched = score_trace_subset_loao(
                "gbdt", X, meta,
                trace_ids=[t.trace_id for t in traces],
                held_out_attack=attack_name,
                seed=cfg.seed,
            )
            rows.append(
                {
                    "defense": "argus_trace_detector (same sample, loao)",
                    **{k: getattr(matched, k) for k in METRIC_KEYS},
                    "n": matched.n_test,
                    "mean_llm_calls": 0.0,
                    "mean_input_tokens": 0.0,
                    "mean_latency_ms": matched.detect_latency_ms,
                    "inference_overhead_x": TRACE_DETECTOR_OVERHEAD_X,
                }
            )
        except ValueError as exc:
            print(f"(matched-sample detector row not included: {exc})")

        # Kept as well, because it is the headline number, but it is a different
        # population and the protocol column now says so.
        res = leave_one_attack_out("gbdt", X, meta, seed=cfg.seed)
        rows.append(
            {
                "defense": "argus_trace_detector (full loao, all attacks)",
                **{k: getattr(res, k) for k in METRIC_KEYS},
                "n": res.n_test,
                # Zero because the detector reads spans the run already emitted. This is
                # the comparison the whole of Study B rests on, so it is reported from
                # the same named constant the evaluator uses rather than a literal.
                "mean_llm_calls": 0.0,
                "mean_input_tokens": 0.0,
                "mean_latency_ms": res.detect_latency_ms,
                "inference_overhead_x": TRACE_DETECTOR_OVERHEAD_X,
            }
        )
    except (SystemExit, ValueError) as exc:
        print(f"(trace detector not included: {exc})")

    frame = pd.DataFrame(rows)
    # Composition travels with the numbers. Without it the table is four AUCs over three
    # different populations, and the reader has no way to tell.
    frame["sample"] = [
        "full loao (all attacks, all cells)"
        if "full loao" in str(d)
        else f"{dataset}/{retriever_name}/{attack_name}/p{n_poison}"
        for d in frame["defense"]
    ]
    for col, val in (
        ("n_positive", n_pos),
        ("n_attacked_negative", len(attacked_neg)),
        ("n_clean_negative", len(clean_traces)),
    ):
        frame[col] = [
            float("nan") if "full loao" in str(d) else val for d in frame["defense"]
        ]

    cols = [
        "defense", "roc_auc", "f1", "recall", "fpr_at_95_tpr",
        "mean_llm_calls", "inference_overhead_x", "mean_latency_ms",
        "n", "n_positive", "n_attacked_negative", "n_clean_negative", "sample",
    ]
    print("\n=== Defence comparison: quality and cost ===")
    _print(frame[[c for c in cols if c in frame.columns]])

    if args.save:
        out = cfg.results_dir / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "baseline_comparison.csv", index=False)
        print(f"\nSaved to {out}")
    return 0


# --------------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the endpoint and the model before committing to a long run.

    Reachability is the easy half. The half that matters is whether this model phrases
    its answers the way the engine parses them: the agent turns free text into control
    decisions at three points, and a model that answers differently does not crash, it
    silently degrades. A grid where every reflect response fell back to SUFFICIENT still
    produces complete, plausible, wrong numbers.
    """
    from argus.doctor import format_report, run_doctor
    from argus.llm.factory import build_llm

    cfg = _cfg(args)
    llm = build_llm(cfg.llm, seed=cfg.seed)
    report = run_doctor(
        cfg, llm,
        throughput_samples=args.samples,
        total_runs=args.total_runs,
    )
    print(format_report(report))

    if args.save:
        out = cfg.results_dir / "doctor.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "ok": report.ok,
                    "info": report.info,
                    "warnings": report.warnings,
                    "probes": [
                        {
                            "name": p.name, "ok": p.ok, "detail": p.detail,
                            "latency_ms": p.latency_ms, "fatal": p.fatal,
                            "tokens_in": p.tokens_in, "tokens_out": p.tokens_out,
                        }
                        for p in report.probes
                    ],
                },
                indent=2,
            )
        )
        print(f"\nSaved to {out}")
    return 0 if report.ok else 1


# ------------------------------------------------------------------- validate
def cmd_validate_corpus(args: argparse.Namespace) -> int:
    """Structural checks on every built corpus.

    This is the gate that would have caught the degenerate NQ corpus before 45 GPU-hours
    were spent on it. Run it after `corpus build` and before `run grid`.
    """
    from argus.corpus.store import Corpus
    from argus.corpus.validate import validate_corpus

    cfg = _cfg(args)
    paths = sorted(cfg.corpora_dir.glob("*.jsonl"))
    if not paths:
        print("No corpora yet. Run: argus corpus build --dataset nq")
        return 1

    failures = 0
    for path in paths:
        if args.dataset and not path.name.startswith(args.dataset):
            continue
        report = validate_corpus(Corpus.load(path), strict=False)
        print(f"\n{path.name}")
        print(report.text())
        if not report.ok:
            failures += 1

    if failures:
        print(
            f"\n{failures} corpus/corpora failed validation. "
            "Do not run the grid on these; see docs/RUNBOOK.md 'Corpus validity'."
        )
    return 1 if failures else 0


def cmd_validate_attacks(args: argparse.Namespace) -> int:
    """Check that every attack's poison is actually retrievable.

    The retrieval condition is half of every poisoning attack. corpus_poisoning satisfied
    it for 0.03% of runs and nobody noticed for a full grid.
    """
    from argus.attacks.registry import list_attacks
    from argus.corpus.loaders import build_corpus
    from argus.corpus.store import Corpus
    from argus.corpus.validate import validate_attack_retrievability
    from argus.retrieval.factory import build_retriever

    cfg = _cfg(args)
    paths = sorted(cfg.corpora_dir.glob(f"{args.dataset}_*.jsonl"))
    if paths:
        corpus = Corpus.load(paths[-1])
    else:
        print(f"No corpus for '{args.dataset}'; building a small one to check against.")
        corpus = build_corpus(
            dataset=args.dataset, n_docs=args.n_docs, n_queries=args.n_queries,
            seed=cfg.seed, validate=args.dataset != "synthetic",
        )

    from argus.attacks.registry import build_attack

    rows, failures = [], 0
    for name in list_attacks():
        attack = build_attack(name, n_poison_docs=args.ratio, seed=cfg.seed)
        retriever = None
        if attack.needs_retriever:
            retriever = build_retriever(corpus, cfg.retrieval, cfg.data_dir / "cache")
            retriever.build()
        result = attack.apply(corpus, retriever=retriever)

        poisoned_retriever = build_retriever(
            result.corpus, cfg.retrieval, cfg.data_dir / "cache"
        )
        poisoned_retriever.build()
        report = validate_attack_retrievability(
            result.corpus, result.poison_by_query, poisoned_retriever,
            top_k=cfg.retrieval.top_k, sample=args.sample, strict=False,
        )
        rows.append({"attack": name, **report,
                     "poison_rate": round(result.corpus.n_poison / len(result.corpus), 4)})
        if not report["ok"]:
            failures += 1

    import pandas as pd

    print("\n=== Attack retrievability ===")
    _print(pd.DataFrame(rows))
    if failures:
        print(
            f"\n{failures} attack(s) fail the retrieval condition. They would contribute "
            "only null rows to the grid and would corrupt leave-one-attack-out."
        )
    return 1 if failures else 0


def cmd_validate_data(args: argparse.Namespace) -> int:
    """Integrity check across written traces and result files.

    Catches the mismatch that let one cell report 500 runs in its result JSON while its
    trace file held 141 usable lines.
    """
    from argus.runner.experiment import count_valid_traces

    cfg = _cfg(args)
    problems: list[str] = []
    checked = 0

    for result_path in sorted(cfg.results_dir.glob("*.json")):
        if result_path.name.startswith("clean__"):
            continue
        try:
            data = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            problems.append(f"{result_path.name}: not valid JSON")
            continue
        cell_id = data.get("cell_id")
        if not cell_id:
            continue
        checked += 1
        trace_path = cfg.traces_dir / f"{cell_id}.jsonl"
        if not trace_path.exists():
            problems.append(f"{cell_id}: result exists but trace file is missing")
            continue
        ok, bad = count_valid_traces(trace_path)
        if bad:
            problems.append(f"{cell_id}: {bad} corrupt trace line(s)")
        if ok != data.get("n_traces"):
            problems.append(
                f"{cell_id}: result says n_traces={data.get('n_traces')} "
                f"but trace file holds {ok}"
            )

    print(f"\nChecked {checked} cell(s).")
    if problems:
        print(f"\n{len(problems)} integrity problem(s):")
        for p in problems:
            print(f"  {p}")
        print("\nRe-run the affected cells: argus run grid --preset <p>")
        print("(the runner now re-executes any cell that is not complete)")
        return 1
    print("All cells consistent: every result JSON matches its trace file.")
    return 0


def cmd_validate_all(args: argparse.Namespace) -> int:
    """Full preflight. Run this before launching a long grid."""
    print("=" * 70)
    print("ARGUS preflight")
    print("=" * 70)
    rc = 0
    for name, fn in (
        ("corpora", cmd_validate_corpus),
        ("attacks", cmd_validate_attacks),
        ("data integrity", cmd_validate_data),
    ):
        print(f"\n--- {name} ---")
        try:
            rc |= fn(args)
        except Exception as exc:  # surfaced, not swallowed
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            rc = 1
    print("\n" + "=" * 70)
    print("PREFLIGHT PASSED" if rc == 0 else "PREFLIGHT FAILED - do not launch the grid")
    print("=" * 70)
    return rc


# ----------------------------------------------------------------------- cost
def cmd_cost_estimate(args: argparse.Namespace) -> int:
    from argus.runner.cost import estimate_grid_cost
    from argus.runner.grid import build_grid, grid_summary

    cfg = _cfg(args)
    cells = build_grid(args.preset, seed=cfg.seed)
    print("\nGrid:")
    _print(grid_summary(cells))
    print("\nEstimate:")
    _print(estimate_grid_cost(cells, args.model or cfg.llm.model))
    print(f"\nBudget ceiling: ${cfg.budget_usd:.2f}")
    return 0


# ------------------------------------------------------------------ dashboard
def cmd_dashboard(args: argparse.Namespace) -> int:
    import subprocess

    app = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    if not app.exists():
        print(f"Dashboard not found at {app}")
        return 1
    try:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    except FileNotFoundError:
        print('Streamlit not installed. Run: pip install -e ".[dash]"')
        return 1


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="argus",
        description="Agentic RAG security research platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version", version=f"argus {__version__}")
    p.add_argument("--data-dir", help="override the data directory")
    p.add_argument("--config-file", help="YAML config file")
    p.add_argument("--backend", choices=["mock", "openai", "local"], help="LLM backend")
    p.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], help="retriever")

    sub = p.add_subparsers(dest="command", required=True)

    # corpus
    c = sub.add_parser("corpus", help="build and inspect corpora").add_subparsers(
        dest="sub", required=True
    )
    cb = c.add_parser("build")
    cb.add_argument(
        "--dataset", default="synthetic",
        choices=["synthetic", "squad", "nq", "hotpotqa"],
        help="synthetic is for development only and must not be used for results",
    )
    cb.add_argument("--n-docs", type=int, default=2000)
    cb.add_argument("--n-queries", type=int, default=100)
    cb.add_argument(
        "--no-validate", action="store_true",
        help="skip structural validation (tests only; never for a real corpus)",
    )
    cb.set_defaults(func=cmd_corpus_build)
    c.add_parser("info").set_defaults(func=cmd_corpus_info)

    # doctor
    doc = sub.add_parser(
        "doctor",
        help="check the LLM endpoint and that this model behaves as the engine expects",
    )
    doc.add_argument(
        "--samples", type=int, default=5,
        help="calls used to measure throughput; 0 skips the projection",
    )
    doc.add_argument(
        "--total-runs", type=int, default=51_400,
        help="programme size the wall-clock projection is scaled to",
    )
    doc.add_argument("--save", action="store_true", help="write data/results/doctor.json")
    doc.set_defaults(func=cmd_doctor)

    # validate
    v = sub.add_parser(
        "validate", help="preflight checks; run before launching a long grid"
    ).add_subparsers(dest="sub", required=True)
    for sub_name, fn, helptext in (
        ("corpus", cmd_validate_corpus, "structural checks on built corpora"),
        ("attacks", cmd_validate_attacks, "check each attack's poison is retrievable"),
        ("data", cmd_validate_data, "trace files against result files"),
        ("all", cmd_validate_all, "run every check"),
    ):
        vp = v.add_parser(sub_name, help=helptext)
        vp.add_argument("--dataset", default="synthetic")
        vp.add_argument("--n-docs", type=int, default=2000)
        vp.add_argument("--n-queries", type=int, default=50)
        vp.add_argument("--ratio", type=int, default=5)
        vp.add_argument("--sample", type=int, default=50)
        vp.set_defaults(func=fn)

    # attack
    a = sub.add_parser("attack", help="poison a corpus").add_subparsers(dest="sub", required=True)
    ar = a.add_parser("run")
    ar.add_argument("--attack", default="poisonedrag_black")
    ar.add_argument("--dataset", default="synthetic")
    ar.add_argument("--ratio", type=int, default=5, help="poisoned documents per query")
    ar.set_defaults(func=cmd_attack_run)
    a.add_parser("list").set_defaults(func=cmd_attack_list)

    # run
    r = sub.add_parser("run", help="execute experiments").add_subparsers(dest="sub", required=True)
    rs = r.add_parser("single")
    rs.add_argument("--config", default="C0", choices=["C0", "C1", "C2", "C3", "C4", "C5"])
    rs.add_argument("--dataset", default="synthetic")
    rs.add_argument("--attack", default="poisonedrag_black")
    rs.add_argument("--ratio", type=int, default=5)
    rs.add_argument("--n-queries", type=int, default=20)
    rs.add_argument("--n-docs", type=int, default=1000)
    rs.add_argument("--budget", type=int, default=None, help="iteration budget override")
    rs.add_argument("--overwrite", action="store_true")
    rs.add_argument("--workers", type=int, default=None)
    rs.set_defaults(func=cmd_run_single)

    rg = r.add_parser("grid")
    rg.add_argument(
        "--preset", default="pilot",
        choices=["smoke", "pilot", "main", "budget", "abstention", "multihop", "retriever"],
    )
    rg.add_argument("--limit", type=int, default=0, help="run only the first N cells")
    rg.add_argument("--n-docs", type=int, default=0)
    rg.add_argument("--overwrite", action="store_true")
    rg.add_argument("--yes", "-y", action="store_true", help="skip the cost confirmation")
    rg.add_argument(
        "--workers", type=int, default=None,
        help="queries executed concurrently per cell (default ARGUS_WORKERS, else 1). "
             "The agent mostly waits on the generator, so this is close to a linear "
             "speed-up until the endpoint saturates. Output is unchanged at any value.",
    )
    rg.set_defaults(func=cmd_run_grid)

    # analyse
    an = sub.add_parser("analyse", help="analyse results").add_subparsers(dest="sub", required=True)
    aa = an.add_parser("ablation")
    aa.add_argument("--save", action="store_true")
    aa.add_argument(
        "--no-matched", action="store_true",
        help="analyse raw data instead of restricting to the shared query set; the "
             "restriction is on by default because comparing configurations across "
             "different query sets biases every effect size",
    )
    aa.set_defaults(func=cmd_analyse_ablation)
    ast = an.add_parser("stage")
    ast.add_argument("--save", action="store_true")
    ast.add_argument("--no-matched", action="store_true")
    ast.set_defaults(func=cmd_analyse_stage)

    # features
    f = sub.add_parser("features", help="trace features").add_subparsers(dest="sub", required=True)
    fb = f.add_parser("build")
    fb.add_argument("--parquet", action="store_true")
    fb.set_defaults(func=cmd_features_build)
    f.add_parser("describe").set_defaults(func=cmd_features_describe)

    # detect
    d = sub.add_parser("detect", help="train and evaluate detectors").add_subparsers(
        dest="sub", required=True
    )
    dt = d.add_parser("train")
    dt.add_argument("--model", default="all", choices=["all", "rules", "iforest", "gbdt"])
    dt.set_defaults(func=cmd_detect_train)
    de = d.add_parser("evaluate")
    de.add_argument("--model", default="all", choices=["all", "rules", "iforest", "gbdt"])
    de.add_argument("--protocol", default="loao", choices=["loao", "random", "cross_dataset"])
    de.add_argument("--train-dataset", default="synthetic")
    de.add_argument("--test-dataset", default="hotpotqa")
    de.add_argument("--ablation", action="store_true", help="also run the feature family ablation")
    de.add_argument("--save", action="store_true")
    de.set_defaults(func=cmd_detect_evaluate)

    # baselines
    b = sub.add_parser("baselines", help="compare against published defences").add_subparsers(
        dest="sub", required=True
    )
    bc = b.add_parser("compare")
    bc.add_argument("--limit", type=int, default=100)
    bc.add_argument(
        "--attack", default="",
        help="restrict the comparison to one attack; default picks the largest group",
    )
    bc.add_argument("--save", action="store_true")
    bc.set_defaults(func=cmd_baselines_compare)

    # cost
    co = sub.add_parser("cost", help="estimate cost").add_subparsers(dest="sub", required=True)
    ce = co.add_parser("estimate")
    ce.add_argument("--preset", default="main")
    ce.add_argument("--model", default="")
    ce.set_defaults(func=cmd_cost_estimate)

    # dashboard
    sub.add_parser("dashboard", help="launch the Streamlit dashboard").set_defaults(
        func=cmd_dashboard
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed work is saved and the run is resumable.")
        return 130
    # except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
    #     print(f"Error: {exc}", file=sys.stderr)
    #     return 1
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"Caused by: {type(exc.__cause__).__name__}: {exc.__cause__}", file=sys.stderr)
        if os.getenv("ARGUS_TRACEBACK"):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
