.PHONY: install test lint benchmark serve demo

install:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check .

benchmark:
	python scripts/run_eval.py

serve:
	uvicorn evidence_rag.api:app --host 0.0.0.0 --port 8000 --reload

demo:
	python scripts/demo.py

