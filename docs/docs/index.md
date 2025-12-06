# Timestep Documentation

Welcome! Timestep makes OpenAI Agents durable and resumable across Python and TypeScript with a tiny surface area: `run_agent` / `runAgent`, `RunStateStore`, and `consume_result`.

## Prerequisites
- `OPENAI_API_KEY`
- **Python storage options** (in order of preference):
  1. **PostgreSQL** (recommended): Set `TIMESTEP_DB_URL=postgresql://user:pass@host/db` or use local Postgres (auto-detected on `localhost:5432`)
  2. **PGLite**: Install Node.js and `@electric-sql/pglite` (`npm install -g @electric-sql/pglite`). Uses a high-performance sidecar process.

## Quick navigation
- Getting started: [Installation and Quick Start](getting-started.md)
- Core concepts: [Architecture](architecture.md), [Use Cases](use-cases.md)
- API reference: [Utilities](api-reference/utilities.md), [MultiModelProvider](api-reference/multi-model-provider.md), [OllamaModelProvider](api-reference/ollama-model-provider.md), [MultiModelProviderMap](api-reference/multi-model-provider-map.md), [Tools](api-reference/tools.md)

## Core features
- Durable execution: save/load `RunState` from Postgres or PGLite.
- Cross-language state: same format in Python and TypeScript.
- Model routing: prefix (`ollama/gpt-oss:20b-cloud`) to select providers; defaults to OpenAI.
- Smart storage: auto-detects local Postgres, falls back to PGLite, or use `TIMESTEP_DB_URL` for remote Postgres.

## Packages
- Python: [`timestep`](https://pypi.org/project/timestep/)
- TypeScript: [`@timestep-ai/timestep`](https://www.npmjs.com/package/@timestep-ai/timestep)
