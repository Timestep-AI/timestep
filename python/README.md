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

## DBOS Workflows

Timestep supports durable agent execution via DBOS workflows. Run agents in workflows that automatically recover from crashes.

### Durable Execution

```python
from timestep import run_agent_workflow, configure_dbos, ensure_dbos_launched
from agents import Agent, OpenAIConversationsSession

configure_dbos()
ensure_dbos_launched()

agent = Agent(model="gpt-4.1")
session = OpenAIConversationsSession()

# Run in a durable workflow
result = await run_agent_workflow(
    agent=agent,
    input_items=input_items,
    session=session,
    stream=False,
    workflow_id="unique-id"
)
```

### Queued Execution

```python
from timestep import queue_agent_workflow

handle = await queue_agent_workflow(
    agent=agent,
    input_items=input_items,
    session=session,
    priority=1,
    deduplication_id="unique-id"
)

result = await handle.get_result()
```

### Scheduled Execution

```python
from timestep import create_scheduled_agent_workflow

await create_scheduled_agent_workflow(
    crontab="0 */6 * * *",  # Every 6 hours
    agent=agent,
    input_items=input_items,
    session=session
)
```

## Package Structure

The Python package is organized into clear modules:

- **`core/`**: Core agent execution functions (`run_agent`, `default_result_processor`)
- **`core/agent_workflow.py`**: DBOS workflows for durable agent execution
- **`config/`**: Configuration utilities (`dbos_config`, `app_dir`)
- **`stores/`**: Data access layer
  - **`agent_store/`**: Agent configuration persistence
  - **`session_store/`**: Session data persistence
  - **`run_state_store/`**: Run state persistence
  - **`shared/`**: Shared database utilities (`db_connection`, `schema`)
  - **`guardrail_registry.py`**: Guardrail registration
  - **`tool_registry.py`**: Tool registration
- **`tools/`**: Agent tools (e.g., `web_search`)
- **`model_providers/`**: Model provider implementations (`OllamaModelProvider`, `MultiModelProvider`)
- **`models/`**: Model implementations (`OllamaModel`)

## Documentation
Full docs: https://timestep-ai.github.io/timestep/
