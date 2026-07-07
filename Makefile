# distil-vibevoice-asr — thin wrappers around the numbered scripts/ pipeline.
# All targets use the project venv explicitly; no activation required.

PY      := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := .venv/bin/pytest
RUFF    := .venv/bin/ruff
CONFIGS := configs

.PHONY: setup test lint label distill-4b distill-1p5b eval help

help:
	@echo "Targets:"
	@echo "  setup         install package (editable) + dev/train/audio extras"
	@echo "  test          run pytest (CPU-only, no network)"
	@echo "  lint          ruff check + format check"
	@echo "  label         pseudo-label real audio with the 8B teacher"
	@echo "  distill-4b    stage 1: prune 8B->4B then distill"
	@echo "  distill-1p5b  stage 2: prune 4B->1.5B then distill (+10% direct-8B)"
	@echo "  eval          run eval gates (MER/cpWER/DER/timestamp MAE)"

setup:
	$(PIP) install -e ".[dev,train,audio]"

test:
	$(PYTEST) tests

lint:
	$(RUFF) check src tests
	$(RUFF) format --check src tests

label:
	$(PY) scripts/20_pseudo_label.py --config $(CONFIGS)/data.yaml

distill-4b:
	$(PY) scripts/30_prune.py --config $(CONFIGS)/prune_4b.yaml
	$(PY) scripts/31_distill.py --config $(CONFIGS)/distill_stage1_4b.yaml

distill-1p5b:
	$(PY) scripts/30_prune.py --config $(CONFIGS)/prune_1p5b.yaml
	$(PY) scripts/31_distill.py --config $(CONFIGS)/distill_stage2_1p5b.yaml

eval:
	$(PY) scripts/50_eval_gates.py --config $(CONFIGS)/eval_gates.yaml
