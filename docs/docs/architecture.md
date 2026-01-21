# Architecture

Timestep is built around the **A2A (Agent-to-Agent)** and **MCP (Model Context Protocol)** protocols, providing a clean foundation for building multi-agent systems. This document explains the architecture and how these protocols work together.

## Overview

Timestep follows the **Task-generating Agents** philosophy from the [A2A Protocol](https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task), where agents always respond with Task objects. We use **MCP sampling** to enable seamless agent-to-agent handoffs.

## A2A Protocol Integration

### Task-Generating Agents

Following the [A2A Task-generating Agents philosophy](https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task), our agents always respond with Task objects (never just Messages). This provides:

- **State management**: Tasks can be in various states (created, input-required, completed, canceled, etc.)
- **Progress tracking**: Task status updates provide visibility into agent progress
- **Multi-turn interactions**: Tasks support context IDs for grouping related interactions
- **Structured tool communication**: Tool calls are communicated via `input-required` state with `DataPart`

### A2A Server Architecture

The A2A server implements a Task-generating Agent:

```python
# Agent always responds with Task objects
class MultiAgentExecutor(AgentExecutor):
    async def execute(self, request: RequestContext) -> Task:
        # Process message and generate response
        # Always return a Task object
        return Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(
                state=TaskState.input_required,  # or completed
                message=Message(parts=[...])
            ),
            history=[...]
        )
```

Key characteristics:
- Uses A2A SDK's `AgentExecutor` interface
- Always returns Task objects
- Manages task lifecycle and state transitions
- Includes tool calls in `DataPart` when `input-required`

### A2A input-required with DataPart

When an agent needs to call tools, it uses A2A's `input-required` state with a `DataPart`:

1. **Agent sets task state** to `input-required`
2. **Agent includes DataPart** in task status message parts:
   ```python
   DataPart(
       data={
           "tool_calls": [
               {
                   "function": {
                       "name": "handoff",
                       "arguments": json.dumps({
                           "agent_uri": "http://.../agents/...",
                           "message": "What's the weather in Oakland?"
                       })
                   }
               }
           ]
       }
   )
   ```
3. **Client detects** `input-required` state
4. **Client extracts** tool calls from `DataPart.data.tool_calls`
5. **Client executes** tools via MCP
6. **Client sends** results back to A2A server

This pattern allows agents to communicate tool needs in a structured, protocol-compliant way.

## MCP Protocol Integration

### MCP Server Architecture

The MCP server provides tools and sampling capabilities:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MCP Server")

@mcp.tool()
async def handoff(
    agent_uri: str,
    message: str,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Handoff tool using MCP sampling."""
    # Use sampling to trigger A2A request
    result = await ctx.session.create_message(
        messages=[SamplingMessage(...)],
        metadata={"agent_uri": agent_uri}
    )
    return {"response": result.content.text.strip()}
```

Key characteristics:
- Tools registered with `@mcp.tool()` decorator
- `handoff` tool uses sampling to trigger A2A requests
- HTTP transport for client connections
- Supports both tool execution and sampling

### MCP Sampling for Handoffs

MCP's sampling feature enables server-initiated LLM interactions. We use it for handoffs:

**Flow:**
1. Agent calls MCP `handoff` tool with target agent URI
2. MCP server calls client's sampling callback via `ctx.session.create_message()`
3. Sampling callback (in client) makes A2A request to target agent
4. Target agent processes and responds
5. Response returned to MCP server
6. MCP server returns result to original agent

**Implementation:**
```python
# In client
async def mcp_sampling_callback(
    context: RequestContext,
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    # Extract agent_uri from metadata
    agent_uri = params.metadata.get("agent_uri")
    message_text = params.messages[0].content.text
    
    # Make A2A request to target agent
    result_text = await handle_agent_handoff(agent_uri, message_text)
    
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=result_text)
    )
```

This pattern allows agents to delegate work to specialized peers without requiring direct A2A client capabilities in the MCP server.

## Client Architecture

The client orchestrates A2A and MCP interactions:

```python
# Connect to A2A server
a2a_client = await ClientFactory.connect(agent_url)

# Connect to MCP server with sampling callback
async with ClientSession(read, write, sampling_callback=mcp_sampling_callback) as session:
    # Send message to A2A server
    async for event in a2a_client.send_message(message_obj):
        task = extract_task_from_event(event)
        
        # Monitor task state
        if task.status.state.value == "input-required":
            # Extract tool calls from DataPart
            tool_calls = extract_tool_calls(task)
            
            # Execute tools via MCP
            for tool_call in tool_calls:
                result = await call_mcp_tool(tool_name, tool_args)
                
                # Send result back to A2A server
                await a2a_client.send_message(tool_result_msg)
```

Key responsibilities:
- Connects to both A2A and MCP servers
- Monitors A2A task state transitions
- Extracts tool calls from `DataPart` when `input-required`
- Executes tools via MCP
- Implements MCP sampling callback for handoffs
- Sends tool results back to A2A server

## Complete Interaction Flow

Here's a complete example of a handoff interaction:

1. **User sends message** to Personal Assistant Agent via A2A client
2. **A2A server** creates Task and processes message
3. **Agent determines** it needs weather information
4. **Agent sets task state** to `input-required` with `DataPart` containing `handoff` tool call
5. **Client detects** `input-required` state
6. **Client extracts** `handoff` tool call from `DataPart`
7. **Client calls** MCP `handoff` tool with weather agent URI
8. **MCP server** calls client's sampling callback
9. **Sampling callback** makes A2A request to weather agent
10. **Weather agent** processes and responds with weather data
11. **Response** returned to MCP server
12. **MCP server** returns result to client
13. **Client sends** result back to A2A server as user message
14. **Personal assistant** receives weather data and presents to user
15. **Task state** transitions to `completed`

## Protocol Specifications

- **[A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)**: Complete A2A protocol reference
- **[A2A Task-generating Agents](https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task)**: Philosophy and patterns
- **[MCP Protocol Specification](https://modelcontextprotocol.io/specification/latest)**: Complete MCP protocol reference

## Key Design Decisions

### Why Task-Generating Agents?

- Provides clear state management
- Enables progress tracking
- Supports multi-turn interactions
- Structured tool call communication

### Why MCP Sampling for Handoffs?

- Allows agents to delegate without direct A2A client capabilities
- Leverages MCP's built-in sampling mechanism
- Keeps handoff logic in the client (where A2A client exists)
- Maintains separation of concerns

### Why DataPart for Tool Calls?

- Protocol-compliant way to communicate tool needs
- Structured format for tool call information
- Supports multiple tool calls in single message
- Clear separation between content and tool calls

## Cross-Language Parity

Python and TypeScript implementations:
- Use same A2A and MCP protocol patterns
- Follow same Task-generating Agent philosophy
- Use same `input-required` with `DataPart` pattern
- Implement same handoff flow via MCP sampling
- Compatible A2A Task and MCP message formats
