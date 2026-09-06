PYTHON ?= python
.PHONY: test paper-a paper-a-stress paper-a-verify paper-a-release paper-b paper-b-practical core release bluebirds amlgym-pilot amlgym-confirmatory-smoke

test:
	$(PYTHON) -m pytest -q

paper-a:
	$(PYTHON) benchmarks/run_paper_a.py

paper-a-stress:
	$(PYTHON) benchmarks/run_paper_a_stress.py

paper-a-verify:
	$(PYTHON) scripts/verify_paper_a.py

paper-a-release: test paper-a paper-a-stress paper-a-verify

paper-b:
	$(PYTHON) benchmarks/run_paper_b.py

paper-b-practical:
	$(PYTHON) benchmarks/run_paper_b_practical.py

core:
	$(PYTHON) benchmarks/run_core.py

release: paper-a-release paper-b paper-b-practical core

bluebirds:
	$(PYTHON) benchmarks/run_paper_b_bluebirds.py

amlgym-pilot:
	$(PYTHON) benchmarks/run_amlgym_matrix.py

amlgym-confirmatory-smoke:
	$(PYTHON) benchmarks/run_amlgym_confirmatory_case.py --domain blocksworld --algorithm SAM --trace-budget 3 --output artifacts/confirmatory-smoke.json
