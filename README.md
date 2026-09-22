# Argus: Agentic RAG Security Research Platform

Research platform for the M.Tech project **"Security of Agentic Retrieval-Augmented
Generation: Mechanism-Level Vulnerability Analysis and Runtime Trace-Based Compromise
Detection."**

---

## What this does

Two studies, one platform.

**Study A** runs a controlled ablation over the four agentic mechanisms (query rewriting,
iterative retrieval, document inspection, reflection) under knowledge-poisoning attacks,
and decomposes attack success into a retrieval-stage term and a reasoning-stage term.

**Study B** reuses the execution traces emitted by Study A, which are already labelled
benign or compromised by construction, and trains a detector that spots a compromised run
from telemetry alone. No model internals, no re-running the generator.

---

## The one thing to know before you start

**The whole platform runs offline, with no API key and no GPU.**

There is a deterministic mock LLM backend that simulates poisoning susceptibility. It is
not a toy: it responds to poisoned context, respects mechanism configuration, and produces
realistic traces. That means you can build, test, debug and demo the entire pipeline for
free, and only spend money when you run the real grid.

```bash
make setup          # install, no torch, no API key needed
make demo           # full pipeline end to end, offline, ~30 seconds
make test           # test suite, offline
```

If `make demo` prints a results table, everything works.

---

## Install

```bash
cd Code
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras, only when you need them:

```bash
pip install -e ".[dense]"    # sentence-transformers, for BGE-base dense retrieval
pip install -e ".[api]"      # openai client, for real LLM backends
pip install -e ".[dash]"     # streamlit dashboard
```

---

## Quick start

```bash
# 1. Build a corpus. Real datasets fetch real passages and are validated on the way in;
#    a corpus that fails the structural checks is refused rather than saved.
argus corpus build --dataset nq --n-docs 60000 --n-queries 500
#    (--dataset squad is a light real-passage alternative; synthetic is dev-only)

# 2. Preflight. Do this before any long run.
argus validate all --dataset nq --n-docs 60000 --n-queries 500

# 3. Run one agent configuration
argus run single --config C5 --n-queries 20

# 4. Run the grids
argus run grid --preset pilot        # small, free, offline rehearsal
argus run grid --preset main         # 7 configs x 3 attacks x 3 ratios
argus run grid --preset budget       # RQ3: iteration budget against poison ratio

# 5. Confirm every cell wrote what its result file claims
argus validate data

# 6. Analyse
argus analyse ablation               # includes the abstention decomposition
argus analyse stage                  # retrieval vs reasoning decomposition

# 7. Detection
argus features build
argus detect train --model all
argus detect evaluate --protocol loao --ablation --save
argus baselines compare --save       # detection quality against inference cost

# 8. Dashboard
argus dashboard
```

The whole programme, with preflight gates and resumable cells:

```bash
tmux new -s argus
bash scripts/server_run.sh nq        # ~51,400 runs, 30-40 h; Ctrl-b d to detach
```

See `docs/RUNBOOK.md` for the full server procedure and the sanity checklist to work
through before quoting any number.

---

## Layout

```
Code/
  src/argus/
    config.py         Config objects and YAML loading
    corpus/           Document store, dataset loaders, synthetic generator
    retrieval/        BM25 (pure python) and dense retrievers
    llm/              Mock, OpenAI-compatible and local HF backends
    agent/            The agentic RAG engine and its four mechanisms
    attacks/          PoisonedRAG black/white box, Zhong-style corpus poisoning
    telemetry/        OpenTelemetry GenAI-shaped spans, JSONL trace writer
    features/         Trace feature extraction, six families
    detect/           Rules, Isolation Forest, gradient boosting, evaluation
    analysis/         Ablation stats, stage decomposition, bootstrap CIs
    baselines/        Perplexity filter, LLM judge, RAGuard-style LOO
    runner/           Single run, grid runner, cost estimation, checkpointing
    cli.py            Command line entry point
  tests/              Test suite, runs fully offline
  configs/            Mechanism and experiment configuration
  dashboard/          Streamlit results explorer
  scripts/            Reproduction and utility scripts
  data/               Corpora, traces, results (gitignored except samples)
  docs/               Architecture, runbook, decisions
```

---

## Mechanism configurations

| Config | Query rewriting | Iterative retrieval | Document inspection | Reflection | Caution on INSUFFICIENT |
|--------|-----------------|---------------------|---------------------|------------|------|
| C0 Vanilla | no | no | no | no | — |
| C1 | yes | no | no | no | — |
| C2 | no | yes | no | no | — |
| C3 | no | no | yes | no | — |
| C4 | no | no | no | yes | yes |
| C5 Full agentic | yes | yes | yes | yes | yes |
| C6 Abstention control | no | no | no | yes | **no** |

There is no separate configuration for the stopping policy. A stopping decision cannot
exist without iteration, so it is a parameter of C2, varied through the iteration budget.

C6 is not part of the ablation; it is the control on C4. Reflection does two things —
judge whether the evidence is sufficient, and decline to answer when it is not — and a
single attack-success figure cannot tell them apart. C6 computes and logs the verdict but
never lets it change the answer prompt, so the gap between C4 and C6 is exactly the effect
of abstention. This matters: in the first full run reflection appeared to cut attack
success by 12 points, and the outcome breakdown showed refusals rising 13 points while
correct answers *fell* 2.9.

---

## Cost reporting

```bash
argus cost estimate --preset main
```

Prints token and dollar estimates before you run anything, and the runner checkpoints
after every run so an interrupted grid resumes instead of restarting.

Cost is reported as a **result**, not enforced as a constraint: reporting robustness and
its inference bill together is one of the gaps this work identifies, so an estimate above
the configured ceiling warns rather than refuses. Set `ARGUS_BUDGET_USD=0` to disable the
mid-run spend guard entirely — on a local endpoint the dollar figure is notional, and a
guard that silently truncates a 34-hour grid costs more than it saves.

---

## Validation

Four things the pipeline now refuses to do, each because it did them silently once and
cost 45 GPU-hours:

- **Save a corpus that is not a retrieval corpus.** Documents that restate their own
  question turn retrieval into a string match. `argus corpus build` validates and refuses.
- **Run an attack whose poison is not retrievable.** An attack that never reaches the
  top-k contributes null rows and corrupts leave-one-attack-out.
- **Build a configuration where iteration cannot reach the answer prompt.** With
  `max_context_docs == top_k`, later rounds are gathered and then discarded.
- **Compare configurations across different query sets.** Cells answer `queries[:n]`, so
  differing `n` means differing queries.

```bash
argus validate all      # corpus, attacks, data integrity
argus validate data     # every result file against its trace file
```

`tests/test_regressions.py` carries one test per defect found in the first run, so none
of them can return unnoticed.

---

## Documentation

- `docs/ARCHITECTURE.md` how the pieces fit together
- `docs/RUNBOOK.md` step by step, from empty directory to results, including the server procedure
- `docs/DECISIONS.md` design decisions and why, plus the revisions made after the first run
- `docs/METRICS.md` every metric defined precisely

---

## Ethics

No new attacks are implemented. Every attack here is a reproduction of published work,
run only against corpora built locally from open datasets. The novel contribution is
defensive.
