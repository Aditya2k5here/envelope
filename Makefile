PY ?= python3

.PHONY: help train test lint sweep demo clean

help:
	@echo "make train   train the network the verifier reasons about"
	@echo "make test    235 tests, mostly property-based soundness checks"
	@echo "make sweep   the results table"
	@echo "make demo    verify one input, then break it at a larger radius"

train:
	$(PY) experiments/train.py

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m pyflakes ev tests experiments || true

sweep:
	$(PY) experiments/sweep.py

demo:
	$(PY) -m ev verify --eps 0.06 --input 7
	@echo
	$(PY) -m ev radius --input 3 --iters 10
	@echo
	$(PY) -m ev show --input 5 --eps 0.20

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info
