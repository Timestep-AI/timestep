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
	ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
	OLLAMA_API_KEY=${OLLAMA_API_KEY} \
	OPENAI_API_KEY=${OPENAI_API_KEY} \
	MODEL_ID=${MODEL_ID} \
	OPENAI_USE_RESPONSES=${OPENAI_USE_RESPONSES} \
	uv run main.py

typescript:
	cd typescript/timestep && \
	ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
	OLLAMA_API_KEY=${OLLAMA_API_KEY} \
	OPENAI_API_KEY=${OPENAI_API_KEY} \
	MODEL_ID=${MODEL_ID} \
	OPENAI_USE_RESPONSES=${OPENAI_USE_RESPONSES} \
	pnpm run start

.PHONY: test-all
test-all: install
	@set -e; \
	for t in gpt-5,false gpt-5,true anthropic/claude-sonnet-4-5,false ollama/smollm2:1.7b,false ollama/gpt-oss:120b-cloud,false; do \
		model_id="$${t%%,*}"; \
		use_responses="$${t#*,}"; \
		echo "\n=== Python $$model_id (OPENAI_USE_RESPONSES=$$use_responses) ==="; \
		$(MAKE) python MODEL_ID="$$model_id" OPENAI_USE_RESPONSES="$$use_responses"; \
		echo "\n=== TypeScript $$model_id (OPENAI_USE_RESPONSES=$$use_responses) ==="; \
		$(MAKE) typescript MODEL_ID="$$model_id" OPENAI_USE_RESPONSES="$$use_responses"; \
	done

test-typescript:
	MODEL_ID=ollama/gpt-oss:120b-cloud \
	OLLAMA_API_KEY=${OLLAMA_API_KEY} \
	OPENAI_USE_RESPONSES=false \
	$(MAKE) typescript

.PHONY: test-all test-typescript
