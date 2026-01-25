# Timestep Python Library

A clean, simple library for building multi-agent systems using A2A (Agent-to-Agent) and MCP (Model Context Protocol) protocols.

## Overview

Timestep provides three core entities:

- **Agent**: A2A Server that contains Loop (AgentExecutor) internally
- **Environment**: MCP Server (extends FastMCP) that provides system prompt and tools
- **Loop**: AgentExecutor inside Agent that uses MCP client to get system prompt and tools from Environment

## Architecture

```
Agent (A2A Server)
  └── Loop (AgentExecutor)
        ↓ (MCP client)
Environment (MCP Server / FastMCP)
```

- **Agent = A2A Server**: Contains Loop internally, exposes agent URI
- **Environment = MCP Server**: Provides system prompt (FastMCP prompt) and tools
- **Loop = AgentExecutor**: Inside Agent, uses MCP client to get system prompt/tools from Environment
- **Separate environment per agent**: Each agent has its own environment (different system prompts, different tools)

## Key Features

- **Protocol-based communication**: All communication via A2A and MCP protocols
- **Human-in-the-loop**: Via A2A input-required, MCP Elicitation, and MCP Sampling
- **Trace-based evals**: Verify handoff flows and tool calls using OpenTelemetry traces
- **All async/streaming-first**: Everything is async, streaming is the default
- **OpenTelemetry tracing**: Automatic tracing to file (traces.jsonl by default)

## Quick Start

### Create an Environment

```python
from timestep.core import Environment

# Create environment (MCP Server)
environment = Environment(
    environment_id="env-1",
    context_id="context-1",
    agent_id="agent-1"
)

# System prompt as FastMCP prompt
@environment.prompt
def system_prompt(agent_name: str) -> str:
    """System prompt for the agent."""
    return f"You are {agent_name}."

# Register tools
@environment.tool()
async def my_tool(param: str) -> dict:
    """Tool description."""
    return {"result": param}

# Start environment
env_uri = await environment.start(port=8080)
```

### Create an Agent

```python
from timestep.core import Agent

# Create agent (A2A Server containing Loop)
agent = Agent(
    agent_id="agent-1",
    name="My Agent",
    model="gpt-4o-mini",
    context_id_to_environment_uri={
        "context-1": "http://localhost:8080/mcp"
    },
    trace_to_file="traces/agent_1.jsonl"
)

# Start agent
agent_uri = await agent.start(port=8000)
```

### Example: Personal Assistant + Weather Assistant

See `examples/personal_assistant/` for a complete example showing:
- Personal assistant with handoff tool
- Weather assistant with get_weather tool
- Handoff flow between agents

## Human-in-the-Loop

Three existing protocol mechanisms:

1. **A2A input-required**: When agent calls tools, task state is `input-required`. Client can pause for human input.
2. **MCP Elicitation**: When MCP server needs human input, it can use elicitation.
3. **MCP Sampling**: When tool (like `handoff`) needs to call another agent, it uses MCP sampling. The sampling callback can pause for human input.

## Trace-Based Evals

Verify handoff flows and tool calls using OpenTelemetry traces:

```python
from timestep.evals import verify_handoff, verify_tool_call

# Verify handoff
handoff_ok = await verify_handoff(
    "traces/personal_assistant.jsonl",
    from_agent="personal-assistant",
    to_agent="weather-assistant"
)

# Verify tool call
tool_ok = await verify_tool_call(
    "traces/weather_assistant.jsonl",
    agent_id="weather-assistant",
    tool_name="get_weather",
    location="Oakland"
)
```

## CLI Usage

```bash
# Start agent
timestep agent start --id agent-1 --name "My Agent" --port 8000 \
  --context-id context-1 --environment-uri http://localhost:8080/mcp

# Start environment
timestep environment start --id env-1 --context-id context-1 --agent-id agent-1 --port 8080

# Run eval
timestep eval run --trace-file traces/agent_1.jsonl \
  --eval-type handoff --from-agent agent-1 --to-agent agent-2
```

## Observability

OpenTelemetry tracing is automatically enabled:

- Traces written to file by default (`traces.jsonl`)
- Spans for: get_system_prompt, call_openai, execute_tools, execute_tool
- Attributes include: agent_id, context_id, tool_name, tool_arguments, etc.

## Dependencies

- `openai>=1.0.0`
- `a2a-sdk[http-server]`
- `mcp`
- `opentelemetry-api>=1.20.0`
- `opentelemetry-sdk>=1.20.0`
- `opentelemetry-instrumentation-httpx>=0.42b0`
- `opentelemetry-exporter-json`

## Examples

See `examples/personal_assistant/` for complete examples including:
- Agent and environment setup
- Handoff between agents
- Trace-based evals
