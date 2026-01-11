# Timestep AI Agents SDK

Welcome! Timestep is a universal **Agents SDK** built around the core **agent-environment loop** using OpenAI-style chat message protocol. The SDK provides a clean foundation for building agents, with an **evaluation harness** as one powerful use case.

## Prerequisites

- **Python 3.11+** or **Node.js 20+**
- **OpenAI API key** (optional, for agents that use OpenAI or LLM-as-judge graders)

## Quick navigation

- Getting started: [Installation and Quick Start](getting-started.md)
- Core concepts: [Architecture](architecture.md)
- Eval-driven development: [Eval-Driven Development Guide](eval-driven-development.md)
- Examples: See `examples/` directory in the repository

## Core Features

### Core SDK
- **Universal protocol**: OpenAI chat message format - works with any agent framework
- **Simple agent harness interface**: Just a function `(messages, context) => assistant_message`
- **Tool execution**: Deterministic tool execution with automatic indexing
- **Token tracking**: Automatic tracking of input/output tokens and costs
- **Cross-language parity**: Same API in Python and TypeScript

### Evaluation Harness
- **Built-in graders**: Code-based (regex, contains, JSON), LLM-as-judge, outcome verification
- **JSONL task format**: Simple, human-readable task definitions
- **CLI interface**: Run eval suites and generate reports

## Packages

- Python: [`timestep`](https://pypi.org/project/timestep/)
- TypeScript: [`@timestep-ai/timestep`](https://www.npmjs.com/package/@timestep-ai/timestep)
