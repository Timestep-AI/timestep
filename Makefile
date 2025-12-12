default: check-all test-all

patch:
	cd python && uv version --bump patch && cd ../typescript && npm version patch --no-git-tag-version --no-commit-hooks

test: test-python test-typescript

test-all: test

test-python:
	cd python && \
	rm -rf .pytest_cache || true && \
	OPENAI_API_KEY="$(OPENAI_API_KEY)" uv run python -m pytest tests/ -v -x

clean-python:
	cd python && rm -rf .venv .pytest_cache __pycache__ */__pycache__ || true

clean-typescript:
	cd typescript && rm -rf node_modules .pnpm-store dist || true

clean: clean-python clean-typescript

fix-permissions:
	@echo "Fixing permissions for Docker-created files..."
	@if [ -d "python/.venv" ]; then \
		echo "Note: python/.venv should be in a Docker named volume. Removing local copy..."; \
		sudo rm -rf python/.venv 2>/dev/null || rm -rf python/.venv 2>/dev/null || true; \
	fi
	@echo "Done! If you still have permission issues, run: sudo chown -R $$(id -u):$$(id -g) python/ typescript/"

test-typescript:
	cd typescript && \
	OPENAI_API_KEY="$(OPENAI_API_KEY)" pnpm test

lint-python:
	cd python && uv run ruff check timestep/

type-check-python:
	cd python && uv run mypy timestep/

lint-typescript:
	cd typescript && pnpm lint

type-check-typescript:
	cd typescript && pnpm type-check

lint: lint-python lint-typescript

type-check: type-check-python type-check-typescript

check-all: lint type-check

run-app-typescript:
	docker compose -f docker-compose.yml -f docker-compose.typescript.yml up -d

run-app-python:
	docker compose -f docker-compose.yml -f docker-compose.python.yml up -d

stop-app:
	docker compose -f docker-compose.yml -f docker-compose.typescript.yml -f docker-compose.python.yml down

logs-app:
	docker compose -f docker-compose.yml -f docker-compose.typescript.yml -f docker-compose.python.yml logs -f
