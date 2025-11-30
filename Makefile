default: test

patch:
	cd python && uv version --bump patch && cd ../typescript && npm version patch --no-git-tag-version --no-commit-hooks

test: test-python test-typescript test-cross-language

test-all: test

test-python:
	cd python && \
	uv pip install --force-reinstall /home/mjschock/Projects/personal/openai-agents-python && \
	uv run pytest tests/test_run_agent.py -v -x && \
	uv run pytest tests/test_same_language_py_to_py.py -v -x

test-typescript:
	cd typescript && \
	npx tsx tests/test_run_agent.ts && \
	npx tsx tests/test_same_language_ts_to_ts.ts

test-cross-language: test-cross-language-ts-to-py test-cross-language-py-to-ts

test-cross-language-ts-to-py:
	cd typescript && npx tsx tests/test_cross_language_ts_to_py.ts

test-cross-language-py-to-ts:
	cd python && uv run pytest tests/test_cross_language_py_to_ts.py -v -x
