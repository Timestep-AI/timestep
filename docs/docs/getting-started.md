# Getting Started

Timestep provides a clean foundation for building agentic systems using A2A and MCP protocols. This guide will help you get started with the examples and understand how handoffs work.

## Installation

### Running the Examples

The easiest way to get started is to run the working examples:

**Python:**

The library includes working examples in the `scripts/` directory:

```bash
# Terminal 1: Start Weather Assistant (port 10000)
OPENAI_API_KEY=your-key-here uv run scripts/weather_assistant_app.py

# Terminal 2: Start Personal Assistant (port 9999)
OPENAI_API_KEY=your-key-here uv run scripts/personal_assistant_app.py

# Terminal 3: Run Test Client
uv run scripts/personal_assistant_test_client.py
```

The test client will send a weather query to the personal assistant, which will hand off to the weather assistant and return the result.

**TypeScript:**
```bash
# Currently shows pending v2 SDK message
make test-example-typescript
```

## Quick Start: Understanding the Architecture

Timestep provides a library in `lib/python/core/` with working examples in `scripts/`:

1. **Agent**: A2A Server that contains Loop (AgentExecutor) internally
2. **Environment**: MCP Server (extends FastMCP) that provides system prompt and tools
3. **Loop**: AgentExecutor inside Agent that uses MCP client to get system prompt and tools from Environment
4. **ResponsesAPI**: Reusable component for `/v1/responses` endpoint with built-in handoff execution

### Example Flow: Agent Handoff

Here's how a handoff works in the examples:

1. **User sends message** to Personal Assistant Agent via A2A
2. **Agent processes** and determines it needs weather information
3. **Agent calls MCP `handoff` tool** with weather agent's URI
4. **MCP server uses sampling** to trigger A2A request to weather agent
5. **Weather agent responds** with weather data
6. **Personal assistant receives** response and presents to user

## Agent Setup

The `Agent` class implements a **Task-generating Agent**:

```python
# scripts/personal_assistant_app.py
from timestep.core import Agent, Environment, ResponsesAPI

# Create environment (MCP Server)
environment = Environment(
    environment_id="personal-assistant-env",
    context_id="personal-context",
    agent_id="personal-assistant",
    enable_handoff=True,  # Enable built-in handoff tool
)

# Create agent (A2A Server)
agent = Agent(
    agent_id="personal-assistant",
    name="Personal Assistant",
    model="gpt-4o-mini",
    context_id_to_environment_uri={
        "personal-context": "http://localhost:9999/mcp"
    }
)

# Create ResponsesAPI for /v1/responses endpoint
responses_api = ResponsesAPI(
    agent=agent,
    agent_base_url="http://localhost:9999",
    context_id_to_environment_uri={
        "personal-context": "http://localhost:9999/mcp"
    }
)
```

Key points:
- `Agent` always responds with Task objects
- Uses `input-required` state when tool calls are needed
- Includes tool calls in `DataPart` within task status message
- `ResponsesAPI` provides `/v1/responses` endpoint with built-in handoff execution

## Environment Setup

The `Environment` class provides tools and sampling:

```python
# scripts/personal_assistant_app.py
from timestep.core import Environment

# Create environment (MCP Server)
environment = Environment(
    environment_id="personal-assistant-env",
    context_id="personal-context",
    agent_id="personal-assistant",
    enable_handoff=True,  # Enable built-in handoff tool (default)
)

# Register custom tools
@environment.tool()
async def my_tool(param: str) -> dict:
    """Tool description."""
    return {"result": param}
```

Key points:
- `Environment` extends FastMCP and provides MCP server functionality
- Built-in `handoff` tool is registered by default (controlled by `enable_handoff` parameter)
- Custom tools are registered with `@environment.tool()` decorator
- `handoff` tool uses MCP sampling to trigger A2A requests
- Handoff execution is built into `ResponsesAPI` component

## Client Setup

The test clients demonstrate how to interact with agents:

```python
# scripts/personal_assistant_test_client.py
from a2a.client import ClientFactory
from timestep.utils.event_helpers import extract_event_data, extract_task_from_tuple

# Connect to A2A server
a2a_client = await ClientFactory.connect("http://localhost:9999")

# Send message and process events
message_obj = create_text_message_object(role="user", content="What's the weather in Oakland?")
async for event in a2a_client.send_message(message_obj):
    task_obj = extract_task_from_tuple(event)
    event_data = extract_event_data(event)
    # Process task state transitions...
```

Key points:
- Test clients use `ClientFactory.connect()` to connect to agents
- Monitor A2A task state transitions
- Extract tool calls from `DataPart` when `input-required`
- The `ResponsesAPI` component handles tool execution and handoffs automatically
- For custom clients, MCP sampling callback is built into `ResponsesAPI`

## Understanding Tool Call Communication

When an agent needs to call tools, it uses A2A's `input-required` state with a `DataPart`:

1. **Agent sets task state** to `input-required`
2. **Agent includes DataPart** in task status message:
   ```json
   {
     "status": {
       "state": "input-required",
       "message": {
         "parts": [
           {
             "data": {
               "data": {
                 "tool_calls": [
                   {
                     "function": {
                       "name": "handoff",
                       "arguments": "{\"agent_uri\": \"...\", \"message\": \"...\"}"
                     }
                   }
                 ]
               }
             }
           }
         ]
       }
     }
   }
   ```
3. **Client extracts** tool calls from `DataPart.data.tool_calls`
4. **Client calls** MCP tools
5. **Client sends** results back to A2A server

## Understanding Handoffs via MCP Sampling

Handoffs use MCP's sampling feature:

1. **Agent calls** MCP `handoff` tool with target agent URI
2. **MCP server** calls client's sampling callback via `ctx.session.create_message()`
3. **Sampling callback** makes A2A request to target agent:
   ```python
   async def mcp_sampling_callback(context, params):
       agent_uri = params.metadata.get("agent_uri")
       message_text = params.messages[0].content.text
       
       # Make A2A request to target agent
       result_text = await handle_agent_handoff(agent_uri, message_text)
       
       return CreateMessageResult(
           role="assistant",
           content=TextContent(type="text", text=result_text)
       )
   ```
4. **Target agent** processes and responds
5. **Response** returned to MCP server
6. **MCP server** returns result to original agent

## Next Steps

- Learn about [Architecture](architecture.md) - A2A/MCP integration patterns
- Explore examples in `scripts/` - complete working implementations
- Review the library code in `lib/python/core/` - Agent, Environment, Loop, ResponsesAPI
- Review [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)
- Review [MCP Protocol Specification](https://modelcontextprotocol.io/specification/latest)
