.PHONY: install test verify reproduce-core clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

verify:
	python scripts/verify_reference_results.py

reproduce-core:
	bash scripts/reproduce_core.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache results/generated figures/generated
