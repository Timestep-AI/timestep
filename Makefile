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
