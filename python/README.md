# Timestep (Python)

Python bindings for the Timestep Agents SDK. See the root `README.md` for the full story; this file highlights Python-specific setup.

## Install
```bash
pip install timestep
```

## Prerequisites (Python)
- `OPENAI_API_KEY`
- Default storage uses PGLite via Node: install Node.js and `@electric-sql/pglite` (`npm install -g @electric-sql/pglite` or local install).
- For better performance use Postgres: set `TIMESTEP_DB_URL=postgresql://user:pass@host/db`. If you must stay on PGLite, keep a long-lived Node/Deno sidecar that holds a `PGlite` connection instead of spawning Node per query.

## Quick start
```python
from timestep import run_agent, RunStateStore, consume_result
from agents import Agent, Session

agent = Agent(model="gpt-4o")
session = Session()
state_store = RunStateStore(agent=agent, session_id=await session._get_session_id())

result = await run_agent(agent, input_items, session, stream=False)
result = await consume_result(result)

if result.interruptions:
    await state_store.save(result.to_state())
```

## Cross-language resume
Save in Python, load in TypeScript with the same `session_id`/`run_id` and `RunStateStore.load()`.

## Model routing
Use `MultiModelProvider` if you need OpenAI + Ollama routing:
```python
from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider

provider_map = MultiModelProviderMap()
provider_map.add_provider("ollama", OllamaModelProvider())
model_provider = MultiModelProvider(provider_map=provider_map)
```

## Documentation
Full docs: https://timestep-ai.github.io/timestep/
