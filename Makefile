# Export environment variables to subcommands
export

default: test

patch:
	cd python && uv version --bump patch && cd ../typescript && npm version patch --no-git-tag-version --no-commit-hooks

test: test-typescript test-python test-cross-language

test-all: test

test-python:
	cd python && \
	uv pip install --force-reinstall ../3rdparty/openai-agents-python && \
	uv run python vendor_openai_agents.py && \
	uv run pytest

test-typescript:
	cd 3rdparty/openai-agents-js && \
	pnpm install && \
	pnpm build && cd - && \
	cd typescript && \
	pnpm install && \
	pnpm test

test-cross-language: test-cross-language-ts-to-py test-cross-language-py-to-ts

test-cross-language-ts-to-py:
	cd typescript && pnpm exec vitest run --bail=1 tests/test_cross_language_ts_to_py.ts

test-cross-language-py-to-ts:
	cd python && uv run pytest tests/test_cross_language_py_to_ts.py
