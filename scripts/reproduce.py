#!/usr/bin/env python3
"""Reproduce the full experimental programme.

    python scripts/reproduce.py --phase all
    python scripts/reproduce.py --phase pilot     # offline rehearsal, free
    python scripts/reproduce.py --phase main      # the real grid, costs money

Runs everything in order and writes a manifest recording exactly what was executed, with
which versions and which seeds, so a result can be traced back to the run that produced
it. See docs/RUNBOOK.md for the reasoning behind each phase.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: The dataset for the main grid. `nq` is canonical but needs the BEIR passage corpus
#: (about 3 GB); `squad` is a real-passage alternative at about 35 MB. Override with
#: --dataset when bandwidth or disk on the run host makes the download impractical.
DEFAULT_DATASET = "nq"

#: Corpus size for the main grid. Attacks inject up to 10 documents per target query, so
#: with 500 queries the corpus gains 5,000 hostile documents. 60,000 clean documents keeps
#: the poison rate near 8%, inside the threat model the cited papers evaluate.
#:
#: The first run used 2,000 documents against 1,000 queries. With five poisoned documents
#: per query that is 5,000 poison in a 7,000-document corpus — 71% hostile — which is a
#: different threat model from PoisonedRAG's five-in-a-million and makes every retrieval
#: statistic incomparable with published work.
MAIN_N_DOCS = "60000"
MAIN_N_QUERIES = "500"
CONFIRM_N_DOCS = "40000"
CONFIRM_N_QUERIES = "300"


def _phases(dataset: str = DEFAULT_DATASET) -> dict[str, list[list[str]]]:
    """Build the phase plan.

    Validation runs before the grid in every phase that costs money. The preflight is
    cheap and the failure it guards against is not: the first full run spent 45 GPU-hours
    on a corpus whose documents were one-sentence restatements of their own questions,
    and on an attack whose poison reached the top-5 for 0.03% of queries.
    """
    return {
        "pilot": [
            # Offline rehearsal on the synthetic corpus. Free, and it exercises exactly
            # the code path the paid run uses.
            ["validate", "attacks", "--dataset", "synthetic", "--n-queries", "30"],
            ["run", "grid", "--preset", "pilot", "--yes"],
            ["validate", "data"],
            ["analyse", "ablation", "--save"],
            ["analyse", "stage", "--save"],
            ["features", "build"],
            ["detect", "train", "--model", "all"],
            ["detect", "evaluate", "--protocol", "loao", "--ablation", "--save"],
            ["baselines", "compare", "--save"],
        ],
        "main": [
            ["corpus", "build", "--dataset", dataset,
             "--n-docs", MAIN_N_DOCS, "--n-queries", MAIN_N_QUERIES],
            ["validate", "corpus", "--dataset", dataset],
            ["validate", "attacks", "--dataset", dataset,
             "--n-docs", MAIN_N_DOCS, "--n-queries", MAIN_N_QUERIES],
            ["run", "grid", "--preset", "main", "--yes"],
            ["validate", "data"],
            ["analyse", "ablation", "--save"],
            ["analyse", "stage", "--save"],
            ["features", "build"],
            ["detect", "train", "--model", "all"],
            ["detect", "evaluate", "--protocol", "loao", "--ablation", "--save"],
            ["baselines", "compare", "--save"],
        ],
        # RQ3: does agency keep protecting as the budget grows? One attack, one ratio.
        "budget": [
            ["run", "grid", "--preset", "budget", "--yes"],
            ["validate", "data"],
            ["analyse", "ablation", "--save"],
        ],
        # C4 against C6: is reflection resisting poison, or declining to answer?
        "abstention": [
            ["run", "grid", "--preset", "abstention", "--yes"],
            ["validate", "data"],
            ["analyse", "ablation", "--save"],
        ],
        "confirm": [
            ["corpus", "build", "--dataset", "hotpotqa",
             "--n-docs", CONFIRM_N_DOCS, "--n-queries", CONFIRM_N_QUERIES],
            ["validate", "corpus", "--dataset", "hotpotqa"],
            ["run", "grid", "--preset", "multihop", "--yes"],
            ["run", "grid", "--preset", "retriever", "--yes"],
            ["validate", "data"],
            ["analyse", "ablation", "--save"],
            ["detect", "evaluate", "--protocol", "cross_dataset",
             "--train-dataset", dataset, "--test-dataset", "hotpotqa", "--save"],
        ],
    }


#: Order the phases run in when --phase all is given. `main` before the confirmation
#: runs, because the confirmation runs are only worth doing if the main grid held up.
PHASE_ORDER = ["pilot", "main", "budget", "abstention", "confirm"]

PHASES = _phases()


def run_step(argv: list[str], data_dir: str) -> dict:
    """Run one CLI step and record what happened."""
    cmd = [sys.executable, "-m", "argus.cli", "--data-dir", data_dir, *argv]
    print(f"\n$ argus {' '.join(argv)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    return {
        "command": " ".join(argv),
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 1),
    }


def environment() -> dict:
    def _v(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:  # noqa: BLE001
            return "not installed"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {m: _v(m) for m in ("numpy", "pandas", "sklearn", "xgboost")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce the experimental programme")
    ap.add_argument("--phase", default="pilot", choices=[*PHASE_ORDER, "all"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument(
        "--dataset", default=DEFAULT_DATASET, choices=["nq", "squad", "hotpotqa"],
        help="corpus for the main grid; squad is the light real-passage alternative",
    )
    args = ap.parse_args()

    plan = _phases(args.dataset)
    phases = list(PHASE_ORDER) if args.phase == "all" else [args.phase]

    print("=" * 74)
    print("ARGUS REPRODUCTION")
    print("=" * 74)
    print(f"phases   : {', '.join(phases)}")
    print(f"data dir : {args.data_dir}")

    if any(p != "pilot" for p in phases):
        from argus.config import load_config

        cfg = load_config()
        if cfg.llm.backend == "mock":
            print(
                "\nWARNING: the LLM backend is 'mock'. The main phase will run but its "
                "output will not be research results.\nSet ARGUS_LLM_BACKEND in .env "
                "first. See docs/RUNBOOK.md phase 2."
            )
            if input("Continue anyway? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1

    manifest: dict = {"environment": environment(), "phases": {}}
    t0 = time.time()

    for phase in phases:
        print(f"\n{'=' * 74}\nPHASE: {phase}\n{'=' * 74}")
        steps = []
        for argv in plan[phase]:
            step = run_step(argv, args.data_dir)
            steps.append(step)
            if step["returncode"] != 0:
                print(f"\nStep failed: argus {step['command']}")
                print("Stopping. Completed work is saved and the run is resumable.")
                manifest["phases"][phase] = steps
                _write(manifest, args.data_dir)
                return step["returncode"]
        manifest["phases"][phase] = steps

    manifest["total_seconds"] = round(time.time() - t0, 1)
    path = _write(manifest, args.data_dir)

    print(f"\n{'=' * 74}")
    print(f"Done in {manifest['total_seconds']:.0f}s. Manifest: {path}")
    print("Results under", Path(args.data_dir) / "results" / "analysis")
    print("Figures    : python scripts/make_figures.py")
    return 0


def _write(manifest: dict, data_dir: str) -> Path:
    out = ROOT / data_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "reproduction_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


if __name__ == "__main__":
    raise SystemExit(main())
