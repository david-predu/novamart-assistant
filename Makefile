.PHONY: install index chat ask search ui web eval eval-report test test-live lint
Q ?= Where is order NM-10432?

install:
	uv sync --extra ui

index:
	uv run novamart index

ask:
	uv run novamart ask "$(Q)"

chat:
	uv run novamart chat

search:
	uv run novamart search "$(Q)"

ui:
	uv run streamlit run ui/app.py

web:
	uv run adk web novamart_agent --port 8000

eval:
	uv run python -m evals.run_eval

eval-report:
	uv run python -m evals.run_eval --report-only

test:
	uv run pytest -q

test-live:
	uv run pytest -q -m live

lint:
	uv run ruff check .
