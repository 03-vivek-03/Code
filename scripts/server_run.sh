#!/usr/bin/env bash
#
# Full experimental programme, for a long-running server session.
#
#   tmux new -s argus
#   cd ~/argus && bash scripts/server_run.sh nq
#   # Ctrl-b d to detach; tmux attach -t argus to come back
#
# Roughly 51,400 agent runs and 30-40 hours with a local 14B model.
#
# Everything is resumable: finished cells are skipped, incomplete ones are re-run. If this
# stops for any reason, run the same command again.
#
# The preflight gates are not optional. The first full programme spent 45 GPU-hours on a
# corpus whose documents were one-sentence restatements of their own questions and on an
# attack whose poison reached the top-5 for 0.03% of queries. Both are now hard failures,
# and both are checked here before the grid starts.
#
# Each grid step also has its own retry loop (see run_grid below). A run once died on the
# first dropped connection to the gateway ("No route to host") after only ~3 seconds of
# total retry patience — a routing blip that is entirely normal over 30+ hours, treated as
# though it were permanent. Three layers now absorb that: the LLM backend itself gives a
# transient error a couple of minutes before giving up (openai_compat.py), a single
# query's exhausted retries no longer discard every other query already completed in the
# same cell (experiment.py's `_execute`), and if a grid step still exits non-zero this
# script retries the whole step — safe because finished cells are skipped on retry, so
# this only redoes whatever cell was genuinely in flight.

set -euo pipefail

DATASET="${1:-nq}"              # nq | squad
shift || true

# Queries executed concurrently per cell. The agent spends most of its wall-clock waiting
# on the generator, so on a remote endpoint this is close to a linear speed-up. Trace
# contents and ordering are unchanged at any value; only timings differ.
WORKERS="${ARGUS_WORKERS:-4}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --workers=*) WORKERS="${1#*=}"; shift ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

N_QUERIES="${N_QUERIES:-500}"
N_DOCS="${N_DOCS:-60000}"
CONFIRM_QUERIES="${CONFIRM_QUERIES:-300}"
CONFIRM_DOCS="${CONFIRM_DOCS:-40000}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="data"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# Unbuffered output so `tail -f` is live rather than arriving in 4 KB blocks.
export PYTHONUNBUFFERED=1

log() { printf '\n\033[1m== %s ==\033[0m %s\n' "$1" "$(date '+%H:%M:%S')" | tee -a "$LOG"; }
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

# Retries one grid preset if the process itself dies (as opposed to the query-level and
# LLM-call-level retries inside it, which handle most transient failures without ever
# reaching here). 5 attempts with a 2-minute pause is about 10 minutes of extra patience
# for a preset that keeps failing outright; if it still hasn't succeeded after that, the
# problem is probably not a transient network blip, and `set -e` below stops the script
# with a clear message rather than silently burning another 10 minutes on each remaining
# preset in turn.
GRID_MAX_ATTEMPTS="${GRID_MAX_ATTEMPTS:-5}"
GRID_RETRY_DELAY_S="${GRID_RETRY_DELAY_S:-120}"

run_grid() {
  local preset="$1" attempt=1
  while (( attempt <= GRID_MAX_ATTEMPTS )); do
    echo "\$ argus run grid --preset $preset --yes --workers $WORKERS  (attempt $attempt/$GRID_MAX_ATTEMPTS)" \
      | tee -a "$LOG"
    if argus run grid --preset "$preset" --yes --workers "$WORKERS" 2>&1 | tee -a "$LOG"; then
      return 0
    fi
    echo "  grid step '$preset' failed on attempt $attempt/$GRID_MAX_ATTEMPTS." | tee -a "$LOG"
    if (( attempt == GRID_MAX_ATTEMPTS )); then
      echo "  giving up on '$preset' after $GRID_MAX_ATTEMPTS attempts over roughly" \
           "$(( GRID_RETRY_DELAY_S * (GRID_MAX_ATTEMPTS - 1) / 60 )) minutes." | tee -a "$LOG"
      echo "  Completed cells are safe on disk. Investigate the failure above, then" \
           "re-run this script — finished cells are skipped and it resumes cleanly." \
        | tee -a "$LOG"
      return 1
    fi
    echo "  retrying in ${GRID_RETRY_DELAY_S}s (finished cells will be skipped)..." \
      | tee -a "$LOG"
    sleep "$GRID_RETRY_DELAY_S"
    attempt=$((attempt + 1))
  done
}

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  else
    echo "No virtualenv. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev,gbdt,api,dense]'"
    exit 1
  fi
fi

{
  echo "======================================================================"
  echo "ARGUS full programme"
  echo "======================================================================"
  echo "started   : $(date -Is)"
  echo "host      : $(hostname)"
  echo "dataset   : $DATASET"
  echo "queries   : $N_QUERIES per cell"
  echo "documents : $N_DOCS"
  echo "workers   : $WORKERS"
  echo "python    : $(python -V 2>&1)"
  echo "log       : $LOG"
  echo "======================================================================"
} | tee -a "$LOG"

# --- 0. the suite must pass on this host, not just on the laptop ------------
log "Test suite"
run python -m pytest -q

# --- 1. endpoint and model behaviour ----------------------------------------
# Not just reachability. The agent turns free text into control decisions at three
# points, and a model that phrases its answers differently does not crash — it silently
# degrades, producing a complete grid of plausible, wrong numbers. `doctor` asserts each
# of those behaviours and exits non-zero if any fails.
log "Doctor: endpoint and model behaviour"
run argus doctor --save

# --- 2. corpora, with validation --------------------------------------------
log "Building corpora"
run argus corpus build --dataset "$DATASET" --n-docs "$N_DOCS" --n-queries "$N_QUERIES"
run argus corpus build --dataset hotpotqa --n-docs "$CONFIRM_DOCS" --n-queries "$CONFIRM_QUERIES"

log "Preflight"
run argus validate corpus
run argus validate attacks --dataset "$DATASET" --n-docs "$N_DOCS" --n-queries "$N_QUERIES"

echo "" | tee -a "$LOG"
echo "Preflight passed. Starting the grid." | tee -a "$LOG"

# --- 3. the grids -----------------------------------------------------------
log "Main grid  (7 configs x 3 attacks x 3 ratios, ~31,500 runs)"
run_grid main

log "Budget sweep  (RQ3: budget x poison ratio, ~6,000 runs)"
run_grid budget

log "Multi-hop confirmation  (HotpotQA, ~2,100 runs)"
run_grid multihop

log "Retriever confirmation  (dense, ~2,100 runs)"
run_grid retriever

# --- 4. integrity before any analysis ---------------------------------------
log "Data integrity"
run argus validate data

# --- 5. analysis ------------------------------------------------------------
log "Study A: mechanism ablation and stage decomposition"
run argus analyse ablation --save
run argus analyse stage --save

log "Study B: features, detectors, baselines"
run argus features build
run argus detect train --model all
run argus detect evaluate --protocol loao --ablation --save
run argus detect evaluate --protocol cross_dataset \
    --train-dataset "$DATASET" --test-dataset hotpotqa --save
run argus baselines compare --limit 300 --save

log "Figures"
run python scripts/make_figures.py || echo "(figures failed; results are still written)"

{
  echo ""
  echo "======================================================================"
  echo "COMPLETE  $(date -Is)"
  echo "======================================================================"
  echo "results : data/results/analysis/"
  echo "figures : data/results/figures/"
  echo "log     : $LOG"
  echo ""
  echo "Before quoting any number, work through Phase 5 of docs/RUNBOOK.md."
  echo "======================================================================"
} | tee -a "$LOG"
