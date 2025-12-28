# Timestep Documentation

Welcome! Timestep is an MVP Agent System with Human-in-the-Loop, Guardrails, Handoffs, and Sessions.

## Prerequisites
- `OPENAI_API_KEY`
- Python 3.11+

## Quick navigation
- Getting started: [Installation and Quick Start](getting-started.md)
- Core concepts: [Architecture](architecture.md), [Use Cases](use-cases.md)
- API reference: [Core API](api-reference/utilities.md)

## Core features
- **Agents with Human-in-the-Loop**: Request human approval during tool execution
- **Guardrails**: Pre and post-execution validation and modification
- **Handoffs**: Agent-to-agent delegation via handoff tools
- **Sessions**: File-based conversation persistence
- **Custom Execution Loop**: Direct OpenAI API integration

## Packages
- Python: [`timestep`](https://pypi.org/project/timestep/)
