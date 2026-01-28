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

## OpenTelemetry Tracing

The Timestep library includes built-in support for OpenTelemetry tracing using zero-code instrumentation. Tracing exports to Jaeger or other OTLP-compatible backends for visualization.

### Quick Start

1. **Start Jaeger all-in-one** (for local development):
   ```bash
   docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
   ```

2. **Configure tracing** (optional - defaults work for local Jaeger):
   ```bash
   # Optional: Set service name (default: "timestep")
   export OTEL_SERVICE_NAME=timestep-personal-assistant

   # Optional: Set OTLP endpoint (default: "http://localhost:4317")
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

   # Optional: Disable tracing
   export OTEL_ENABLED=false
   ```

3. **Run your application** - traces will automatically be sent to Jaeger

4. **View traces** in Jaeger UI: http://localhost:16686

### What Gets Traced

- **FastAPI requests**: All HTTP requests to your FastAPI endpoints
- **HTTP clients**: Outgoing HTTP requests (httpx, requests)
- **OpenAI API calls**: Calls to OpenAI's API
- **A2A protocol**: A2A requests and responses (via `a2a-sdk[telemetry]`)
- **MCP protocol**: MCP tool calls and responses

### Programmatic Configuration

```python
from timestep.observability.tracing import enable_tracing

# Enable tracing and instrument FastAPI app in one call
from fastapi import FastAPI
app = FastAPI()
enable_tracing(app, service_name="my-service", otlp_endpoint="http://localhost:4317")
```

### Viewing Traces

Traces are sent to Jaeger (or another OTLP-compatible backend) and can be viewed in the Jaeger UI at http://localhost:16686.

**Using other backends:**
- **Tempo + Grafana**: Set `OTEL_EXPORTER_OTLP_ENDPOINT` to your Tempo endpoint
- **Custom OTLP endpoint**: Set `OTEL_EXPORTER_OTLP_ENDPOINT` to your endpoint URL

### Troubleshooting

If traces don't appear in Jaeger:
1. Ensure Jaeger is running: `docker ps | grep jaeger`
2. Check the OTLP endpoint: `echo $OTEL_EXPORTER_OTLP_ENDPOINT`
3. Verify the service name appears in Jaeger UI
4. Check application logs for tracing initialization messages

View traces using `jq` (if you need to export to file):

```bash
# View all traces
cat traces.jsonl | jq .

# View a specific trace
cat traces.jsonl | jq 'select(.trace_id == "abc123...")'

# Count spans per trace
cat traces.jsonl | jq '.spans | length'
```

See [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/) for more details.

## Dependencies

- `openai>=1.0.0`
- `a2a-sdk[http-server,telemetry]` - Includes telemetry extra for A2A protocol tracing
- `mcp`
- `opentelemetry-distro` - For automatic instrumentation (optional)
- `opentelemetry-sdk` - Core SDK (optional)
- `opentelemetry-instrumentation-*` - Library-specific instrumentations (optional)

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
