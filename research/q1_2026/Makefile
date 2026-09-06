PYTHON ?= python

.PHONY: test core paper-a paper-b benchmark verify release bluebirds amlgym-smoke amlgym-matrix

test:
	$(PYTHON) -m pytest -q

core:
	$(PYTHON) benchmarks/run_core.py

paper-a:
	$(PYTHON) benchmarks/run_paper_a.py

paper-b:
	$(PYTHON) benchmarks/run_paper_b.py

benchmark: paper-a paper-b core

verify:
	$(PYTHON) scripts/verify_release.py

release: test benchmark verify

bluebirds:
	$(PYTHON) benchmarks/run_paper_b_bluebirds.py

amlgym-smoke:
	$(PYTHON) benchmarks/run_amlgym_matrix.py --domains blocksworld --algorithms SAM --trace-budgets 3 --timeout-per-case 600

amlgym-matrix:
	$(PYTHON) benchmarks/run_amlgym_matrix.py --timeout-per-case 900
