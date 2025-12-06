# Timestep Documentation

Welcome! Timestep makes OpenAI Agents durable and resumable across Python and TypeScript with a tiny surface area: `run_agent` / `runAgent`, `RunStateStore`, and `consume_result`.

## Prerequisites
- `OPENAI_API_KEY`
- Python default storage: Node.js + `@electric-sql/pglite` on PATH.
- Production/fast path: set `TIMESTEP_DB_URL` to Postgres. If you stick with PGLite from Python, keep a long-lived Node/Deno sidecar that holds a `PGlite` connection instead of spawning Node per query.

## Quick navigation
- Getting started: [Installation and Quick Start](getting-started.md)
- Core concepts: [Architecture](architecture.md), [Use Cases](use-cases.md)
- API reference: [Utilities](api-reference/utilities.md), [MultiModelProvider](api-reference/multi-model-provider.md), [OllamaModelProvider](api-reference/ollama-model-provider.md), [MultiModelProviderMap](api-reference/multi-model-provider-map.md), [Tools](api-reference/tools.md)

## Core features
- Durable execution: save/load `RunState` from PGLite or Postgres.
- Cross-language state: same format in Python and TypeScript.
- Model routing: prefix (`ollama/llama3`) to select providers; defaults to OpenAI.
- Shared storage: default app directory (`~/.config/timestep/pglite/` on Linux; platform equivalents elsewhere).

## Packages
- Python: [`timestep`](https://pypi.org/project/timestep/)
- TypeScript: [`@timestep-ai/timestep`](https://www.npmjs.com/package/@timestep-ai/timestep)
