# @timestep-ai/timestep (TypeScript)

TypeScript bindings for the Timestep Agents SDK. See the root `README.md` for the full story; this file highlights TypeScript-specific setup.

## Install
```bash
npm install @timestep-ai/timestep
# or pnpm add @timestep-ai/timestep
# or yarn add @timestep-ai/timestep
```

## Prerequisites (TypeScript)
- `OPENAI_API_KEY`
- Node 20+.
- `@electric-sql/pglite` is already a dependency; default storage lives under `~/.config/timestep/pglite/` (or the platform equivalent). Set `TIMESTEP_DB_URL` to Postgres for production.

## Quick start
```typescript
import { runAgent, RunStateStore, consumeResult } from '@timestep-ai/timestep';
import { Agent, Session } from '@openai/agents';

const agent = new Agent({ model: 'gpt-4.1' });
const session = new Session();
const stateStore = new RunStateStore({ agent, sessionId: await session.getSessionId() });

let result = await runAgent(agent, inputItems, session, false);
result = await consumeResult(result);

if (result.interruptions?.length) {
  await stateStore.save(result.state);
}
```

## Cross-language resume
Load state saved in Python with the same `sessionId` and continue the run; state format is compatible.

## Model routing
Prefix model names (`ollama/gpt-oss:20b-cloud`) or provide a custom `MultiModelProviderMap` to route to your providers.

## Documentation
Full docs: https://timestep-ai.github.io/timestep/
