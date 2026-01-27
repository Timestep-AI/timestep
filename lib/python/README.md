# Timestep Python Library

A clean, simple library for building multi-agent systems using A2A (Agent-to-Agent) and MCP (Model Context Protocol) protocols.

## Overview

Timestep provides four core entities:

- **Agent**: A2A Server that contains Loop (AgentExecutor) internally
- **Environment**: MCP Server (extends FastMCP) that provides system prompt and tools
- **Loop**: AgentExecutor inside Agent that uses MCP client to get system prompt and tools from Environment
- **ResponsesAPI**: Reusable component for `/v1/responses` endpoint with built-in handoff execution

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
- **All async/streaming-first**: Everything is async, streaming is the default

## Quick Start

### Create an Environment

```python
from timestep.core import Environment

# Create environment (MCP Server)
environment = Environment(
    environment_id="env-1",
    context_id="context-1",
    agent_id="agent-1",
    enable_handoff=True,  # Enable built-in handoff tool (default)
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

### Create ResponsesAPI

```python
from timestep.core import ResponsesAPI

# Create ResponsesAPI instance
responses_api = ResponsesAPI(
    agent=agent,
    agent_base_url=f"http://localhost:8000",
    context_id_to_environment_uri={
        "context-1": env_uri
    }
)

# Mount ResponsesAPI routes to your FastAPI app
for route in responses_api.fastapi_app.routes:
    fastapi_app.add_api_route(route.path, route.endpoint, methods=route.methods)
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
    }
)

# Start agent
agent_uri = await agent.start(port=8000)
```

### Example: Personal Assistant + Weather Assistant

See `scripts/` for complete working examples:
- `personal_assistant_app.py` - Personal assistant with handoff tool enabled
- `weather_assistant_app.py` - Weather assistant with handoff tool disabled and `get_weather` tool
- `personal_assistant_test_client.py` - Test client demonstrating handoff flow
- `weather_assistant_test_client.py` - Test client for weather assistant

These examples demonstrate:
- Using `Agent`, `Environment`, and `ResponsesAPI` classes
- Conditional handoff tool registration via `enable_handoff` parameter
- Complete handoff flow between agents
- Custom tool registration (e.g., `get_weather`)

## Built-in Handoff Tool

The `Environment` class includes a built-in `handoff` tool that enables agent-to-agent delegation:

- **Automatic Registration**: The handoff tool is registered by default when `enable_handoff=True` (default)
- **Conditional Registration**: Set `enable_handoff=False` to disable the tool (useful for specialized agents that don't need handoff capability)
- **MCP Sampling**: The handoff tool uses MCP sampling to call target agents via A2A
- **Built-in Execution**: The `ResponsesAPI` component includes built-in handoff execution, so no additional setup is required

Example:
```python
# Enable handoff (default)
environment = Environment(..., enable_handoff=True)

# Disable handoff
environment = Environment(..., enable_handoff=False)
```

## ResponsesAPI Component

The `ResponsesAPI` class provides a reusable `/v1/responses` endpoint:

- **Streaming and Non-streaming**: Handles both streaming and non-streaming response modes
- **Built-in Handoff Execution**: Automatically executes handoff tool calls via MCP sampling
- **Tool Execution**: Executes MCP tools and sends results back to the agent
- **A2A Integration**: Seamlessly integrates with the A2A protocol

The `ResponsesAPI` is designed to be mounted on a FastAPI application alongside the agent's A2A endpoints.

## Human-in-the-Loop

Three existing protocol mechanisms:

1. **A2A input-required**: When agent calls tools, task state is `input-required`. Client can pause for human input.
2. **MCP Elicitation**: When MCP server needs human input, it can use elicitation.
3. **MCP Sampling**: When tool (like `handoff`) needs to call another agent, it uses MCP sampling. The sampling callback can pause for human input.

## Dependencies

- `openai>=1.0.0`
- `a2a-sdk[http-server]`
- `mcp`

## Examples

See `scripts/` for complete working examples:
- `personal_assistant_app.py` - Complete personal assistant implementation
- `weather_assistant_app.py` - Complete weather assistant implementation
- Test clients demonstrating agent interactions and handoff flows

These examples show:
- Creating `Agent` and `Environment` instances
- Registering custom tools
- Using `ResponsesAPI` for `/v1/responses` endpoint
- Handoff flow between agents
