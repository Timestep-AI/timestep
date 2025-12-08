# Timestep Documentation

Welcome! Timestep makes OpenAI Agents durable and resumable across Python and TypeScript with a tiny surface area: `run_agent` / `runAgent` and `RunStateStore`.

## Prerequisites
- `OPENAI_API_KEY`
- **PostgreSQL**: Set `PG_CONNECTION_URI=postgresql://user:pass@host/db` or use local Postgres (auto-detected on `localhost:5432`)

## Quick navigation
- Getting started: [Installation and Quick Start](getting-started.md)
- Core concepts: [Architecture](architecture.md), [Use Cases](use-cases.md)
- **DBOS Workflows**: [Durable Execution, Queuing, and Scheduling](dbos-workflows.md) (New!)
- API reference: [Utilities](api-reference/utilities.md), [MultiModelProvider](api-reference/multi-model-provider.md), [OllamaModelProvider](api-reference/ollama-model-provider.md), [MultiModelProviderMap](api-reference/multi-model-provider-map.md), [Tools](api-reference/tools.md)

## Core features
- **Durable execution**: save/load `RunState` from PostgreSQL.
- **DBOS workflows**: Run agents in durable workflows that automatically recover from crashes, with queuing and scheduling support.
- Cross-language state: same format in Python and TypeScript.
- Model routing: prefix (`ollama/gpt-oss:20b-cloud`) to select providers; defaults to OpenAI.
- PostgreSQL storage: use `PG_CONNECTION_URI` for database connection.

## Packages
- Python: [`timestep`](https://pypi.org/project/timestep/)
- TypeScript: [`@timestep-ai/timestep`](https://www.npmjs.com/package/@timestep-ai/timestep)
