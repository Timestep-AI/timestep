# Timestep AI Agents SDK

Welcome! Timestep is a clean, low-level library for building **agentic systems** using the modern industry standards: **A2A (Agent-to-Agent)** and **MCP (Model Context Protocol)** protocols.

## Overview

Timestep provides a solid foundation for creating multi-agent systems with clear examples across multiple languages. We follow the **Task-generating Agents** philosophy from the [A2A Protocol](https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task), where agents always respond with Task objects that can transition through various states.

## Protocols

Timestep is built on two complementary industry standards:

- **[A2A Protocol](https://a2a-protocol.org/latest/specification/)**: Agent-to-Agent communication standard for peer-to-peer agent collaboration
- **[MCP Protocol](https://modelcontextprotocol.io/specification/latest)**: Model Context Protocol for tools, resources, and server-initiated LLM interactions (sampling)

### How They Work Together

- **A2A** handles agent discovery, task management, and agent-to-agent communication
- **MCP** provides tool execution and sampling capabilities
- **Handoffs** are implemented using MCP's sampling feature
- **Tool calls** are communicated via A2A's `input-required` task state with a `DataPart`

## First MVP: Handoffs

Our first MVP focuses on **handoffs** - enabling agents to seamlessly delegate tasks to other specialized agents. This demonstrates the power of combining A2A and MCP.

## Prerequisites

- **Python 3.11+** or **Node.js 20+**
- **OpenAI API key** (required for A2A agents using OpenAI)

## Quick navigation

- Getting started: [Installation and Quick Start](getting-started.md)
- Core concepts: [Architecture](architecture.md)
- Examples: See `examples/` directory in the repository

## Core Features

### A2A Integration
- **Task-generating Agents**: Always respond with Task objects
- **State management**: Tasks transition through states (created, input-required, completed)
- **Tool call communication**: Uses `input-required` state with `DataPart` for tool calls
- **Agent discovery**: Agent Card-based discovery mechanism

### MCP Integration
- **Tool execution**: MCP tools for agent capabilities
- **Sampling for handoffs**: Server-initiated LLM interactions enable agent-to-agent delegation
- **HTTP transport**: Streamable HTTP transport for client-server communication

### Cross-Language Support
- **Python**: Fully functional with working examples
- **TypeScript**: Pending MCP SDK v2 release (expected Q1 2026)
- **Web UI**: Browser-based chat interface

## Packages

- Python: [`timestep`](https://pypi.org/project/timestep/) (future)
- TypeScript: [`@timestep-ai/timestep`](https://www.npmjs.com/package/@timestep-ai/timestep) (future)

## Implementation Status

### Python
✅ **Fully functional** - Python implementation is complete and working with handoffs.

### TypeScript
⚠️ **Pending v2 SDK release** - TypeScript implementation is incomplete, pending `@modelcontextprotocol/sdk` v2 release.
