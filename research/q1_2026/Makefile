PYTHON ?= python
.PHONY: test paper-a paper-b core release amlgym-smoke

test:
	$(PYTHON) -m pytest -q
paper-a:
	$(PYTHON) benchmarks/run_paper_a.py
paper-b:
	$(PYTHON) benchmarks/run_paper_b.py
core:
	$(PYTHON) benchmarks/run_core.py
release: test paper-a paper-b core
amlgym-smoke:
	$(PYTHON) benchmarks/run_amlgym_matrix.py --domains blocksworld --algorithms SAM --trace-budgets 3
