default: test

patch:
	cd python && uv version --bump patch && cd ../typescript && npm version patch --no-git-tag-version --no-commit-hooks

test: test-python test-typescript

test-all: test

test-python:
	cd python && uv pip install --force-reinstall /home/mjschock/Projects/personal/openai-agents-python && uv run pytest tests/test_run_agent.py -v -x

test-typescript:
	cd typescript && npx tsx tests/test_run_agent.ts
