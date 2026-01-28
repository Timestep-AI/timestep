# TypeScript Library

This directory is reserved for the future TypeScript library implementation.

We intend to create a TypeScript version of the Timestep library that mirrors the Python implementation in `lib/python/core/`, providing:

- **Agent**: A2A Server that contains Loop (AgentExecutor) internally
- **Environment**: MCP Server that provides system prompt and tools
- **Loop**: AgentExecutor inside Agent that uses MCP client to get system prompt and tools from Environment, and provides `/v1/responses` endpoint with built-in handoff execution
- **Human-in-the-loop**: Via A2A input-required, MCP Elicitation, and MCP Sampling
- **Trace-based evals**: Verify handoff flows and tool calls using OpenTelemetry traces
- **All async/streaming-first**: Everything is async, streaming is the default

The TypeScript implementation will follow the same architecture and patterns as the Python version, ensuring cross-language parity.

## Status

The TypeScript library is planned but not yet implemented. The Python library in `lib/python/core/` is the reference implementation and should be used as the guide.

## Current Python Implementation

The Python library is fully implemented in `lib/python/core/` with working examples in `scripts/`:

- **Library Components**: `agent/`, `environment/`, `loop/`
- **Examples**: `scripts/personal_assistant_app.py`, `scripts/weather_assistant_app.py`
- **Features**: Built-in handoff tool, conditional tool registration, Loop component providing `/v1/responses` endpoint

## Future Implementation

When implementing the TypeScript version:

1. Use the same three core entities: Agent, Environment, Loop
2. Follow the same protocol communication patterns (A2A/MCP)
3. Maintain the same API surface where possible
4. Use TypeScript's type system for better type safety
5. Support the same observability and eval features
6. Include built-in handoff tool in Environment (conditionally registered)
7. Provide Loop component for `/v1/responses` endpoint
8. Examples should be in `scripts/` (or equivalent location) mirroring Python examples
