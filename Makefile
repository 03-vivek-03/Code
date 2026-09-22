.PHONY: help setup demo test lint clean corpus validate pilot main budget confirm \
        analyse detect baselines figures server dashboard

DATASET ?= nq
N_QUERIES ?= 500
N_DOCS ?= 60000

help:
	@echo "Argus: Agentic RAG Security Research Platform"
	@echo ""
	@echo "  make setup       Install the package and dev dependencies"
	@echo "  make test        Run the test suite (offline, ~2 min)"
	@echo "  make demo        Full offline pipeline end to end (no API key)"
	@echo "  make lint        Run ruff"
	@echo ""
	@echo "  make corpus      Build the real corpus       (DATASET=$(DATASET))"
	@echo "  make validate    Preflight: corpus, attacks, data integrity"
	@echo "  make pilot       Offline rehearsal of the real design"
	@echo ""
	@echo "  make main        Main grid       ~31,500 runs"
	@echo "  make budget      RQ3 sweep       ~6,000 runs"
	@echo "  make confirm     Multi-hop + dense confirmation runs"
	@echo "  make server      Everything, with preflight gates (~30-40 h)"
	@echo ""
	@echo "  make analyse     Ablation, abstention and stage decomposition"
	@echo "  make detect      Features, detectors, leave-one-attack-out"
	@echo "  make baselines   Defence comparison: quality against cost"
	@echo "  make figures     Regenerate the figures"
	@echo "  make dashboard   Launch the Streamlit dashboard"
	@echo ""
	@echo "  make clean       Remove generated data and caches"
	@echo ""
	@echo "Run 'make validate' before any grid. See docs/RUNBOOK.md."

setup:
	python3 -m pip install -e ".[dev,gbdt]"

demo:
	python3 scripts/demo.py

test:
	python3 -m pytest

lint:
	python3 -m ruff check src tests scripts

corpus:
	argus corpus build --dataset $(DATASET) --n-docs $(N_DOCS) --n-queries $(N_QUERIES)

# Preflight. The grid targets depend on it, so a broken corpus or an unretrievable
# attack stops the run before it starts rather than after 45 hours.
validate:
	argus validate all --dataset $(DATASET) --n-docs $(N_DOCS) --n-queries $(N_QUERIES)

pilot:
	python3 scripts/reproduce.py --phase pilot

main: validate
	argus run grid --preset main --yes

budget:
	argus run grid --preset budget --yes

confirm:
	argus run grid --preset multihop --yes
	argus run grid --preset retriever --yes

server:
	bash scripts/server_run.sh $(DATASET)

analyse:
	argus analyse ablation --save
	argus analyse stage --save

detect:
	argus features build
	argus detect train --model all
	argus detect evaluate --protocol loao --ablation --save

baselines:
	argus baselines compare --limit 300 --save

figures:
	python3 scripts/make_figures.py

dashboard:
	streamlit run dashboard/app.py

clean:
	rm -rf data/corpora data/traces data/results data/features data/models
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
