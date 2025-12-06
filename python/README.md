# Timestep (Python)

Python bindings for the Timestep Agents SDK. See the root `README.md` for the full story; this file highlights Python-specific setup.

## Install
```bash
pip install timestep
```

## Prerequisites (Python)
- `OPENAI_API_KEY`
- **Database storage**:
  1. **PostgreSQL** (recommended): Set `PG_CONNECTION_URI=postgresql://user:pass@host/db`
  2. **PGLite** (default): Install Node.js and `@electric-sql/pglite` (`npm install -g @electric-sql/pglite`). Uses a high-performance sidecar process for optimal performance.

## Quick start
```python
from timestep import run_agent, RunStateStore
from agents import Agent, Session

agent = Agent(model="gpt-4.1")
session = Session()
state_store = RunStateStore(agent=agent, session_id=await session._get_session_id())

result = await run_agent(agent, input_items, session, stream=False)

if result.interruptions:
    await state_store.save(result.to_state())
```

## Cross-language resume
Save in Python, load in TypeScript with the same `session_id` and `RunStateStore.load()`.

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
