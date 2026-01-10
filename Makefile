# Makefile for Timestep

default: test

patch:
	cd python && uv version --bump patch && cd ../typescript && npm version patch --no-git-tag-version --no-commit-hooks

test:
	@echo "Running Python tests..."
	cd python && uv run pytest
	@echo "Running TypeScript tests..."
	cd typescript && pnpm install && pnpm test

test-python:
	cd python && uv run pytest

test-typescript:
	cd typescript && pnpm install && pnpm test
