# Export environment variables to subcommands
export

# PostgreSQL connection string for tests (shared across Python and TypeScript)
export PG_CONNECTION_URI=postgresql://postgres:postgres@localhost:5433/timestep_test?sslmode=disable

default: test

patch:
	cd python && uv version --bump patch && cd ../typescript && npm version patch --no-git-tag-version --no-commit-hooks

test-setup:
	@echo "Starting test PostgreSQL database..."
	docker compose -f docker-compose.test.yml up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@timeout=30; \
	while [ $$timeout -gt 0 ]; do \
		if docker compose -f docker-compose.test.yml exec -T postgres-test pg_isready -U postgres > /dev/null 2>&1; then \
			echo "PostgreSQL is ready!"; \
			break; \
		fi; \
		echo "Waiting for PostgreSQL... ($$timeout seconds remaining)"; \
		sleep 1; \
		timeout=$$((timeout - 1)); \
	done; \
	if [ $$timeout -eq 0 ]; then \
		echo "ERROR: PostgreSQL failed to start within 30 seconds"; \
		exit 1; \
	fi
	@echo "Test database is ready. PG_CONNECTION_URI=$(PG_CONNECTION_URI)"

test-teardown:
	@echo "Stopping test PostgreSQL database..."
	docker compose -f docker-compose.test.yml down

test: test-setup test-python-setup test-typescript test-python

test-all: test

test-python-setup:
	cd python && \
	uv pip install --force-reinstall ../3rdparty/openai-agents-python && \
	uv run python vendor_openai_agents.py

test-python: test-python-setup
	cd python && \
	uv run pytest

test-typescript: test-python-setup
	cd 3rdparty/openai-agents-js && \
	pnpm install && \
	pnpm build && cd - && \
	cd typescript && \
	pnpm install && \
	pnpm test
