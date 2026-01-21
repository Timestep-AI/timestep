# Getting Started

Timestep provides a clean foundation for building agentic systems using A2A and MCP protocols. This guide will help you get started with the examples and understand how handoffs work.

## Installation

### Running the Examples

The easiest way to get started is to run the working examples:

**Python:**
```bash
# Start A2A and MCP servers, then run test client
make test-example-python
```

**TypeScript:**
```bash
# Currently shows pending v2 SDK message
make test-example-typescript
```

## Quick Start: Understanding the Architecture

Timestep examples demonstrate a complete A2A/MCP setup:

1. **A2A Server**: Implements a Task-generating Agent
2. **MCP Server**: Provides tools and sampling capabilities
3. **Client**: Orchestrates A2A and MCP interactions

### Example Flow: Agent Handoff

Here's how a handoff works in the examples:

1. **User sends message** to Personal Assistant Agent via A2A
2. **Agent processes** and determines it needs weather information
3. **Agent calls MCP `handoff` tool** with weather agent's URI
4. **MCP server uses sampling** to trigger A2A request to weather agent
5. **Weather agent responds** with weather data
6. **Personal assistant receives** response and presents to user

## A2A Server Setup

The A2A server implements a **Task-generating Agent**:

```python
# examples/python/a2a_server.py
from a2a.server.apps.rest.fastapi_app import A2ARESTFastAPIApplication
from a2a.server.agent_execution.agent_executor import AgentExecutor

# Create agent executor
executor = MultiAgentExecutor(agent_id=agent_id)

# Create A2A app
a2a_app = A2ARESTFastAPIApplication(
    agent_card=agent_card,
    http_handler=handler,
)
```

Key points:
- Always responds with Task objects
- Uses `input-required` state when tool calls are needed
- Includes tool calls in `DataPart` within task status message

## MCP Server Setup

The MCP server provides tools and sampling:

```python
# examples/python/mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MCP Server")

@mcp.tool()
async def handoff(
    agent_uri: str,
    message: str,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Handoff tool that uses MCP sampling to call another agent."""
    # Use ctx.session.create_message() to trigger sampling
    result = await ctx.session.create_message(
        messages=[sampling_message],
        metadata={"agent_uri": agent_uri}
    )
    return {"response": result.content.text.strip()}
```

Key points:
- Tools are registered with `@mcp.tool()`
- `handoff` tool uses sampling to trigger A2A requests
- Sampling callback is implemented in the client

## Client Setup

The client orchestrates A2A and MCP:

```python
# examples/python/test_client.py
from a2a.client import ClientFactory
from mcp import ClientSession

# Connect to A2A server
a2a_client = await ClientFactory.connect(agent_url)

# Connect to MCP server with sampling callback
async with ClientSession(read, write, sampling_callback=mcp_sampling_callback) as session:
    # Process A2A message stream
    async for event in a2a_client.send_message(message_obj):
        task = extract_task_from_event(event)
        
        # Check for input-required state
        if task.status.state.value == "input-required":
            # Extract tool calls from DataPart
            tool_calls = extract_tool_calls(task)
            
            # Call MCP tools
            for tool_call in tool_calls:
                result = await call_mcp_tool(tool_name, tool_args)
                # Send result back to A2A server
```

Key points:
- Monitors A2A task state transitions
- Extracts tool calls from `DataPart` when `input-required`
- Implements MCP sampling callback for handoffs
- Calls MCP tools and sends results back to A2A

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
- Explore examples in `examples/python/` - complete working implementations
- Review [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)
- Review [MCP Protocol Specification](https://modelcontextprotocol.io/specification/latest)
