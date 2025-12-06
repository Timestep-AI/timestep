# Timestep AI Agents SDK

Durable OpenAI Agents with one API across Python and TypeScript. Pause and resume runs (even across languages), keep state in one place, and route models with simple prefixes.

## What Timestep gives you
- Durable runs: save and resume `RunState` without changing your agent code.
- Cross-language parity: same surface in Python and TypeScript; state stays compatible.
- Single storage story: auto-detects local Postgres, falls back to PGLite, or use `TIMESTEP_DB_URL` for remote Postgres.
- Model routing without new APIs: prefix models (`ollama/gpt-oss:20b-cloud`) and let `MultiModelProvider` pick the backend.
- Minimal concepts: `run_agent` / `runAgent`, `RunStateStore`, `consume_result`.

## Prerequisites
- `OPENAI_API_KEY`
- **Python storage options** (in order of preference):
  1. **PostgreSQL** (recommended): Set `TIMESTEP_DB_URL=postgresql://user:pass@host/db` or use local Postgres (auto-detected on `localhost:5432`)
  2. **PGLite**: Install Node.js and `@electric-sql/pglite` (`npm install -g @electric-sql/pglite`). Uses a high-performance sidecar process.

## Quick start

### Python (async)

```python
from timestep import run_agent, RunStateStore, consume_result
from agents import Agent, Session

agent = Agent(model="gpt-4.1")
session = Session()
state_store = RunStateStore(agent=agent, session_id=await session._get_session_id())

result = await run_agent(agent, input_items, session, stream=False)
result = await consume_result(result)

if result.interruptions:
    state = result.to_state()
    await state_store.save(state)  # resume in Python or TypeScript
```

### TypeScript

```typescript
import { runAgent, RunStateStore, consumeResult } from '@timestep-ai/timestep';
import { Agent, Session } from '@openai/agents';

const agent = new Agent({ model: 'gpt-4.1' });
const session = new Session();
const stateStore = new RunStateStore({ agent, sessionId: await session.getSessionId() });

let result = await runAgent(agent, inputItems, session, false);
result = await consumeResult(result);

if (result.interruptions?.length) {
  await stateStore.save(result.state); // resume in TS or Python
}
```

## Cross-language resume

1) Start in Python, save state on interruption:
```python
state = result.to_state()
await state_store.save(state)
```
2) Load and continue in TypeScript:
```typescript
const saved = await stateStore.load();
for (const interruption of saved.getInterruptions()) saved.approve(interruption);
await runAgent(agent, saved, session, false);
```

## Routing models
- `gpt-4.1` or `openai/gpt-4.1` → OpenAI
- `ollama/gpt-oss:20b-cloud` → Ollama (local or cloud)
- Add your own prefixes via `MultiModelProviderMap`.

## Docs
- Full docs: https://timestep-ai.github.io/timestep/
- Python notes: python/README.md
- TypeScript notes: typescript/README.md

## License
MIT License - see `LICENSE`.
