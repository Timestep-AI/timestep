# Timestep AI Agents SDK

Durable OpenAI Agents with one API across Python and TypeScript. Pause and resume runs (even across languages), keep state in one place, and route models with simple prefixes.

## What Timestep gives you
- Durable runs: save and resume `RunState` without changing your agent code.
- Cross-language parity: same surface in Python and TypeScript; state stays compatible.
- Single storage story: default PGLite in an app directory; point `TIMESTEP_DB_URL` at Postgres for production.
- Model routing without new APIs: prefix models (`ollama/llama3`) and let `MultiModelProvider` pick the backend.
- Minimal concepts: `run_agent` / `runAgent`, `RunStateStore`, `consume_result`.

## Prerequisites
- `OPENAI_API_KEY`
- Python default storage: Node.js + `@electric-sql/pglite` on PATH (Python shells out to Node today).
- Better performance: set `TIMESTEP_DB_URL` to Postgres. If you must stay on PGLite from Python, run a small, long-lived Node/Deno sidecar that holds a `PGlite` connection and exposes a thin HTTP/IPC shim so Python isn’t spawning Node per query.

## Quick start

### Python (async)

```python
from timestep import run_agent, RunStateStore, consume_result
from agents import Agent, Session

agent = Agent(model="gpt-4o")
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

const agent = new Agent({ model: 'gpt-4o' });
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
- `gpt-4o` or `openai/gpt-4o` → OpenAI
- `ollama/llama3` → Ollama (local or cloud)
- Add your own prefixes via `MultiModelProviderMap`.

## Docs
- Full docs: https://timestep-ai.github.io/timestep/
- Python notes: python/README.md
- TypeScript notes: typescript/README.md

## License
MIT License - see `LICENSE`.
