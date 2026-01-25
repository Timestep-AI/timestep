# TypeScript Library

This directory is reserved for the future TypeScript library implementation.

We intend to create a TypeScript version of the Timestep library that mirrors the Python implementation, providing:

- **Agent**: A2A Server that contains Loop (AgentExecutor) internally
- **Environment**: MCP Server (extends FastMCP) that provides system prompt and tools
- **Loop**: AgentExecutor inside Agent that uses MCP client to get system prompt and tools from Environment
- **Human-in-the-loop**: Via A2A input-required, MCP Elicitation, and MCP Sampling
- **Trace-based evals**: Verify handoff flows and tool calls using OpenTelemetry traces
- **All async/streaming-first**: Everything is async, streaming is the default

The TypeScript implementation will follow the same architecture and patterns as the Python version, ensuring cross-language parity.

## Status

The TypeScript library is planned but not yet implemented. The Python library should be used as the reference implementation.

## Future Implementation

When implementing the TypeScript version:

1. Use the same three core entities: Agent, Environment, Loop
2. Follow the same protocol communication patterns (A2A/MCP)
3. Maintain the same API surface where possible
4. Use TypeScript's type system for better type safety
5. Support the same observability and eval features
