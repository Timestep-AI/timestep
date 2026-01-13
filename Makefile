# Makefile for Timestep

default: test

patch:
	cd python && uv version --bump patch && cd ../typescript && npm version patch --no-git-tag-version --no-commit-hooks

test:
	@echo "Running Python tests..."
	cd python && uv pip install -e ".[dev]" && uv run python -m pytest
	@echo "Running TypeScript tests..."
	cd typescript && pnpm install && pnpm test

test-python:
	cd python && uv pip install -e ".[dev]" && uv run python -m pytest

test-typescript:
	cd typescript && pnpm install && pnpm test

gaia-eval:
	uv run scripts/gaia_eval.py

test-app:
	@echo "Starting servers..."
	@OPENAI_API_KEY=$$OPENAI_API_KEY docker compose up -d
	@echo "Waiting for servers to be ready..."
	@timeout 30 bash -c 'until curl -s http://localhost:8000/agents/00000000-0000-0000-0000-000000000000/.well-known/agent-card.json > /dev/null 2>&1 && curl -s http://localhost:8080/mcp > /dev/null 2>&1; do sleep 0.5; done' || (echo "Servers failed to start"; docker compose logs; exit 1)
	@echo "Following logs and running test client..."
	@(docker compose logs -f & echo $$! > /tmp/compose-logs.pid) && \
	uv run app/test_client.py "$${TEST_MESSAGE:-What's the weather in Oakland?}"; \
	CLIENT_EXIT=$$?; \
	kill $$(cat /tmp/compose-logs.pid 2>/dev/null) 2>/dev/null || true; \
	docker compose down; \
	exit $$CLIENT_EXIT
