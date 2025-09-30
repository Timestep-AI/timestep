default: install python typescript

clean:
	rm -rf python/timestep/.venv
	rm -rf python/timestep/data
	rm -rf typescript/timestep/data
	rm -rf typescript/timestep/node_modules

install:
	cd python/timestep && \
	uv sync && \
	cd ../..
	cd typescript/timestep && \
	pnpm install && \
	cd ../..

.PHONY: default clean install python typescript

python:
	cd python/timestep && \
	OPENAI_API_KEY=${OPENAI_API_KEY} OLLAMA_API_KEY=${OLLAMA_API_KEY} uv run index.py

typescript:
	cd typescript/timestep && \
	OPENAI_API_KEY=${OPENAI_API_KEY} OLLAMA_API_KEY=${OLLAMA_API_KEY} pnpm run start
