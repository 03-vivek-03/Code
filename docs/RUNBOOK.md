# Runbook

From an empty directory to results you can put in the write-up.

The full programme is about **51,400 agent runs** and takes **30 to 40 hours** on a
single machine with a local 14B model. It is designed to be run inside `tmux`, to survive
a dropped SSH session, and to resume from wherever it stopped.

---

## Read this first: why the pipeline now refuses things

The first full run produced 73,400 traces over 45 GPU-hours and most of the headline
numbers were not measuring what they were named after. Four defects mattered:

| What happened | How it is now prevented |
|---|---|
| The NQ corpus was one stub document per question, `"{question} The answer is {gold}."` — a lookup table, not a retrieval corpus | `argus corpus build` runs a structural validator and **refuses to save** a corpus that fails it |
| `corpus_poisoning` reached the top-5 for 0.03% of queries, contributing 22,000 null rows and wrecking the detection headline | `argus validate attacks` and a runner preflight **refuse to run** an attack whose poison is not retrievable |
| Iterative retrieval could not reach the answer prompt, because `max_context_docs == top_k` | `AgentConfig.__post_init__` **raises** on that combination |
| Configurations were compared across different query sets (1,000 queries for C0–C2, 500 for C3–C5) | `check_matched_queries` **refuses** a grid whose cells disagree, and the analysis restricts to the shared query set by default |

These are hard failures on purpose. Every one of them was silent the first time and cost
45 hours. If a command refuses, read what it says rather than working around it.

---

## Phase 0: install and verify (15 minutes, free)

```bash
cd Code
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[dev,gbdt]"

make test                            # ~2 minutes, fully offline
make demo                            # end-to-end pipeline on the mock backend
```

`make test` must be green before anything else. It includes `tests/test_regressions.py`,
which has one test per defect above; if any of those fail, the corresponding bug is back.

Nothing so far needs an API key, a GPU or a network connection.

---

## Phase 1: rehearse offline (30 minutes, free)

Run the real experimental design at small scale on the mock backend.

```bash
python scripts/reproduce.py --phase pilot
```

**Check these before spending anything:**

- `argus validate data` reports every cell consistent.
- The ablation table shows all seven configurations (C0–C6) with equal `n`.
- `mean_context_docs` for C2 is larger than for C0. If they are equal, iteration is not
  reaching the prompt and RQ3 cannot be answered.
- `mean_input_tokens` for C2 is larger than for C0, for the same reason.
- `decomposition_residual` in the stage table is near zero.
- `asr_without_poison_in_context` is near zero. If it is high, the target answers are
  being produced for some other reason and the decomposition is contaminated.
- Leave-one-attack-out produces a number for **all three** attacks with none skipped.
- `baseline_comparison.csv` exists in `data/results/analysis/`.

Mock numbers are not findings. This phase checks wiring, not results.

---

## Phase 2: prepare the run host

### 2.1 Copy the project across

```bash
# from your laptop
rsync -av --exclude '.venv' --exclude 'data/traces' --exclude 'data/results' \
      --exclude 'data/features' --exclude 'data/cache' --exclude '__pycache__' \
      "Code/" user@server:~/argus/
```

Keep `data/samples/` — it is a tracked artefact. Everything else under `data/` is
regenerated.

### 2.2 Install on the server

```bash
ssh user@server
cd ~/argus
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gbdt,api,dense]"
pip install datasets                 # required for real corpora
python -m pytest -q                  # must pass here too
```

### 2.3 Configure the backend

Create `.env`:

```bash
ARGUS_LLM_BACKEND=openai
ARGUS_LLM_BASE_URL=https://your-gateway/v1
ARGUS_LLM_MODEL=qwen2.5:14b-instruct-q4_K_M
ARGUS_LLM_API_KEY=...

ARGUS_RETRIEVER=bm25
ARGUS_DATA_DIR=data
ARGUS_SEED=42

# 0 disables the mid-run spend guard. On self-hosted hardware the dollar figure is
# notional, and the guard silently truncating a 34-hour grid is worse than the cost it
# saves. Leave it armed only when you are paying a real per-token bill.
ARGUS_BUDGET_USD=0

# A 14B model answering a full C5 prompt over HTTPS can exceed the 60s default, and a
# timeout mid-grid is indistinguishable from a model failure.
ARGUS_LLM_TIMEOUT_S=120

# Queries run concurrently per cell. The agent mostly waits on the generator, so on a
# remote endpoint this is close to a linear speed-up. Results and ordering are unchanged
# at any value; `argus doctor` projects the wall-clock at 1/2/4/8.
ARGUS_WORKERS=4
```

### 2.4 Check the endpoint *and the model's behaviour*

Reachability is the easy half:

```bash
curl -sS "$ARGUS_LLM_BASE_URL/models" -H "Authorization: Bearer $ARGUS_LLM_API_KEY" | head

curl -sS "$ARGUS_LLM_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $ARGUS_LLM_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"'"$ARGUS_LLM_MODEL"'",
       "messages":[{"role":"user","content":"Reply with the single word OK."}],
       "max_tokens":5,"temperature":0}'
```

The half that matters is whether *this* model behaves the way the engine parses. The
agent turns free text into control decisions at three points, and a model that phrases
its answers differently does not crash — it silently degrades, and the grid comes out
complete, plausible and wrong.

```bash
argus doctor            # asserts all four task behaviours, probes a full-size prompt,
                        # measures throughput, projects wall-clock at 1/2/4/8 workers
argus doctor --save     # also writes data/results/doctor.json
```

**Do not proceed unless it prints `DOCTOR PASSED`.** In particular:

| Probe | If it fails |
|---|---|
| `reflect: verdict parses` | M4 would default to SUFFICIENT on every run — C4 becomes C0 while still paying for the extra call |
| `inspect: index parses` | M3 would inspect an arbitrary document |
| `rewrite: query parses` | M1 would feed the model's preamble to the retriever as query terms |
| `answer: stays terse` | Not fatal, but tokens and latency will exceed the projection |

During the run, every cell reports `parse_failure_rate`; anything above 5% prints a
warning naming the cell.

### 2.4 Build and validate the corpus

This is the step that would have caught the biggest defect. It downloads real passages,
so it needs network and disk.

```bash
source .venv/bin/activate

# Canonical: Natural Questions over the BEIR passage corpus (~3 GB download)
argus corpus build --dataset nq --n-docs 60000 --n-queries 500

# If the BEIR download is impractical on this host, SQuAD is a real-passage
# alternative at about 35 MB and passes the same validator:
# argus corpus build --dataset squad --n-docs 60000 --n-queries 500
```

The command prints a validation report and **exits non-zero if the corpus fails**. Every
check must read `PASS`:

```
  [PASS] mean_doc_tokens              98.4    (need >= 25.0)
  [PASS] question_echo_rate           0.06    (need <= 0.35)
  [PASS] gold_answer_present          0.91    (need >= 0.7)
  [PASS] target_type_match            0.87    (need >= 0.6)
  [PASS] poison_rate                  0.00    (need <= 0.2)
  [PASS] queries_with_gold_docs       1.00    (need >= 0.95)
```

Then check the attacks:

```bash
argus validate attacks --dataset nq --n-docs 60000 --n-queries 500
```

All three must show `poison_in_top_k_rate` well above 0.30. `corpus_poisoning` was at
0.0003 in the first run.

Finally, the full preflight:

```bash
argus validate all --dataset nq --n-docs 60000 --n-queries 500
```

**Do not start the grid until this prints `PREFLIGHT PASSED`.**

---

## Phase 3: run the grid under tmux (30–40 hours)

### 3.1 Start a session

```bash
tmux new -s argus
```

If you are reconnecting later: `tmux attach -t argus`. To detach without stopping
anything: **Ctrl-b** then **d**.

### 3.2 Launch

```bash
cd ~/argus
source .venv/bin/activate

# Everything, in order, with the doctor and validation gates between phases.
# The script logs to data/run_<timestamp>.log itself.
bash scripts/server_run.sh nq --workers 4
```

Detach with **Ctrl-b d**. The run continues.

To run the phases individually instead — useful if you want to inspect between them:

```bash
python scripts/reproduce.py --phase main        # ~21 h, 31,500 runs
python scripts/reproduce.py --phase budget      # ~4 h,   6,000 runs   (RQ3)
python scripts/reproduce.py --phase confirm     # ~3 h,   4,200 runs   (multi-hop + dense)
```

### 3.3 Monitor

Open a second window inside the same session with **Ctrl-b c**, then:

```bash
# live log
tail -f ~/argus/data/run_*.log

# progress: cells finished out of 63 in the main grid
ls ~/argus/data/results/*.json | wc -l

# integrity, safe to run at any time
cd ~/argus && source .venv/bin/activate && argus validate data

# GPU and memory
watch -n 5 nvidia-smi
```

Switch windows with **Ctrl-b 0** / **Ctrl-b 1**. List sessions with `tmux ls`.

### 3.4 If it stops

The grid is resumable and now verifies completeness before skipping a cell. A cell whose
trace file holds fewer valid records than `n_queries`, or that contains a truncated line,
is re-run rather than trusted. So the recovery is simply to relaunch the same command:

```bash
tmux attach -t argus
python scripts/reproduce.py --phase main
```

Finished cells are skipped, incomplete ones are redone. Nothing is lost.

If the machine rebooted, re-create the tmux session and relaunch — same command.

---

## Phase 4: analysis (minutes)

```bash
cd ~/argus && source .venv/bin/activate

argus validate data                                  # must be clean before quoting anything
argus analyse ablation --save
argus analyse stage --save
argus features build
argus detect train --model all
argus detect evaluate --protocol loao --ablation --save
argus detect evaluate --protocol cross_dataset --train-dataset nq --test-dataset hotpotqa --save
argus baselines compare --limit 300 --save
python scripts/make_figures.py
```

Outputs land in `data/results/analysis/`:

| File | What it answers |
|---|---|
| `results_table.csv` | Per-configuration attack success, accuracy, **abstention**, clean accuracy, cost |
| `mechanism_effects.csv` | RQ1: per-mechanism effect sizes with CIs and FDR correction |
| `abstention_table.csv` | Whether a mechanism resists poison or just stops answering |
| `stage_decomposition.csv`, `stage_attribution.csv` | RQ2: retrieval stage against reasoning stage |
| `budget_curve.csv` | RQ3: iteration budget against poison ratio |
| `detect_loao.csv` | RQ4/RQ5: the headline detection metric |
| `baseline_comparison.csv` | O6: detection quality against inference cost |
| `cost_table.csv` | Robustness and its bill, reported together |

---

## Phase 5: sanity checks before writing anything up

Run through this list against the real results. Every item corresponds to a way the first
run went wrong.

**Corpus and attacks**
- [ ] `argus validate corpus` passes on every corpus used.
- [ ] All three attacks show `poison_in_top_k_rate` above 0.30.
- [ ] `poison_rate` on the working corpus is below 0.20.
- [ ] `distinct_target_answers` is large; targets are not all the same string.

**Study A**
- [ ] Every configuration has the same `n` in `results_table.csv`.
- [ ] `mean_context_docs` and `mean_input_tokens` for C2 exceed C0.
- [ ] Attack success on C0 under PoisonedRAG at 5 documents is in the region published
      by the paper. Well below it means the generation condition is still not firing.
- [ ] `decomposition_residual` is near zero for every configuration.
- [ ] The poison-ratio curve is monotone: 1 → 5 → 10 documents should not go down.
- [ ] `abstention_table.csv` is read alongside `mechanism_effects.csv`. A mechanism with a
      high `share_explained_by_abstention` is declining to answer, not resisting poison.
- [ ] C4 against C6 quantifies exactly that.

**Study B**
- [ ] Leave-one-attack-out reports all three folds, none skipped for want of positives.
- [ ] Per-fold AUCs are within a reasonable band of each other. One fold far below the
      others means that attack is not producing compromised traces.
- [ ] Cross-dataset AUC is not *higher* than cross-attack AUC. If it is, the detector is
      keying on attack-specific artefacts rather than on compromise.
- [ ] Feature ablation shows no single family carrying the whole detector.
- [ ] `answer_is_refusal` is not the dominant feature. It was 0.354 in the first run,
      which meant the detector was reading the outcome off the answer string.

**Integrity**
- [ ] `argus validate data` reports every cell consistent.
- [ ] Every trace records the real model name, not `mock-deterministic`.

---

## Corpus validity

The validator enforces six properties. If one fails, the fix is the corpus, never the
threshold.

| Check | Threshold | Why |
|---|---|---|
| `mean_doc_tokens` | ≥ 25 | Stub documents make retrieval and generation trivial |
| `question_echo_rate` | ≤ 0.35 | A gold document restating its question turns retrieval into a string match |
| `gold_answer_present` | ≥ 0.70 | If the gold document lacks the answer the task is unanswerable |
| `target_type_match` | ≥ 0.60 | Implausible targets let the generator dismiss poison without reasoning |
| `poison_rate` | ≤ 0.20 | A corpus that is mostly poison is not the published threat model |
| `queries_with_gold_docs` | ≥ 0.95 | Every query needs a supporting document |

---

## Troubleshooting

**`CorpusValidationError` on build.** Read the report. `question_echo_rate` high means the
loader is producing stub documents. Do not pass `--no-validate`; that flag exists for
tests only.

**`Attack poison reaches the top-5 for only X%`.** The attack's retrieval condition is not
satisfied on this corpus. Do not lower the threshold — a non-retrieving attack contributes
only null rows and corrupts leave-one-attack-out.

**`max_context_docs does not exceed top_k`.** A configuration was constructed where
iteration cannot influence the answer. Raise `max_context_docs`.

**`cells within a dataset must all use the same n_queries`.** Two cells would answer
different query sets. Use one preset, or pass the same `--n-queries` throughout.

**`every leave-one-attack-out fold was skipped`.** No attack produced enough compromised
traces. Run `argus validate attacks`.

**Corrupt trace lines reported by `validate data`.** A run was killed mid-write. Re-run
the affected cells; the runner detects and redoes incomplete cells automatically.

**Ollama slows down or stalls.** Check `nvidia-smi` for memory pressure and
`journalctl -u ollama -f` for the server log. The runner retries with exponential backoff,
so brief stalls are absorbed.

**Out of disk.** Traces are the bulk. `data/traces/` runs to roughly 20–30 GB for the full
programme. Compress finished cells with `gzip data/traces/*.jsonl` — the reader handles
`.gz` transparently.
