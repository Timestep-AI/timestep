# Timestep AI Agents SDK

A clean, low-level library for building **agentic systems** using the modern industry standards: **A2A (Agent-to-Agent)** and **MCP (Model Context Protocol)** protocols. Timestep provides a solid foundation for creating multi-agent systems with clear examples across multiple languages.

## Core Philosophy

Timestep follows the **Task-generating Agents** philosophy from the [A2A Protocol](https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task), where agents always respond with Task objects that can transition through various states (including `input-required` for tool calls). We use **MCP sampling** to enable seamless agent-to-agent handoffs, allowing agents to delegate work to specialized peers.

## Protocols

Timestep is built on two complementary industry standards:

- **[A2A Protocol](https://a2a-protocol.org/latest/specification/)**: Agent-to-Agent communication standard for peer-to-peer agent collaboration
- **[MCP Protocol](https://modelcontextprotocol.io/specification/latest)**: Model Context Protocol for tools, resources, and server-initiated LLM interactions (sampling)

### How They Work Together

- **A2A** handles agent discovery, task management, and agent-to-agent communication
- **MCP** provides tool execution and sampling capabilities
- **Handoffs** are implemented using MCP's sampling feature: when an agent needs to hand off to another agent, the MCP server uses sampling to trigger an A2A request to the target agent
- **Tool calls** are communicated via A2A's `input-required` task state with a `DataPart` containing the tool call information

## First MVP: Handoffs

Our first MVP focuses on **handoffs** - enabling agents to seamlessly delegate tasks to other specialized agents. This demonstrates the power of combining A2A and MCP:

1. Agent receives a task via A2A
2. Agent determines it should hand off to another agent
3. Agent calls the MCP `handoff` tool
4. MCP server uses sampling to trigger an A2A request to the target agent
5. Target agent processes the request and returns results
6. Original agent receives the response and continues

## Project Structure

```
timestep/
├── lib/                    # Library code
│   ├── python/             # Python library
│   │   └── core/           # Core library components
│   │       ├── agent/      # Agent class (A2A Server)
│   │       ├── environment/ # Environment class (MCP Server)
│   │       └── loop/        # Loop class (provides /v1/responses endpoint)
│   └── typescript/         # TypeScript library (planned)
├── scripts/                # Working example applications
│   ├── personal_assistant_app.py  # Personal assistant with handoff enabled
│   ├── weather_assistant_app.py   # Weather assistant with handoff disabled
│   ├── personal_assistant_test_client.py  # Test client for personal assistant
│   └── weather_assistant_test_client.py   # Test client for weather assistant
├── examples/               # Legacy examples directory (currently empty)
│   ├── python/             # (empty - examples moved to scripts/)
│   └── typescript/         # TypeScript examples (pending MCP SDK v2)
└── app/                    # Web UI for testing/chatting with agents
    ├── index.html
    ├── index.css
    └── index.js
```

## Implementation Status

### Python
✅ **Fully functional** - Python library is complete in `lib/python/core/` with working examples in `scripts/`.

The library includes:
- **Agent**: A2A Server that contains Loop (AgentExecutor) internally
- **Environment**: MCP Server (extends FastMCP) that provides system prompt and tools
- **Loop**: AgentExecutor inside Agent that uses MCP client to get system prompt and tools from Environment, and provides `/v1/responses` endpoint with built-in handoff execution

Working examples demonstrate:
- Personal assistant with handoff tool enabled
- Weather assistant with handoff tool disabled
- Complete handoff flow between agents

### TypeScript
⚠️ **Pending v2 SDK release** - TypeScript implementation is incomplete.

The TypeScript examples in `examples/typescript/` are pending the release of `@modelcontextprotocol/sdk` v2 (expected Q1 2026). The current v1.x SDK doesn't export the HTTP transport classes we need (`McpServer`, `StreamableHTTPServerTransport`, `createMcpExpressApp`).

We explored multiple approaches to use v2 from GitHub (git dependencies, `bun create`, etc.), but none work with our requirement for inline dependencies without additional files. When v2 is published to npm, we'll update the TypeScript implementation to use the new APIs.

See `examples/typescript/*.ts` files for detailed status comments and TODOs.

## Quick Start

### Running the Examples

The easiest way to get started is to run the working examples:

**Python:**

```bash
# Terminal 1: Start Weather Assistant (port 10000)
OPENAI_API_KEY=your-key-here uv run scripts/weather_assistant_app.py

# Terminal 2: Start Personal Assistant (port 9999)
OPENAI_API_KEY=your-key-here uv run scripts/personal_assistant_app.py

# Terminal 3: Run Test Client
uv run scripts/personal_assistant_test_client.py
```

**TypeScript:**
⚠️ TypeScript implementation is pending MCP SDK v2 release (expected Q1 2026). See `examples/typescript/` for details.

### Example: Agent Handoff

The examples demonstrate a personal assistant agent that can hand off weather queries to a specialized weather agent:

1. **Personal Assistant Agent** (A2A server) receives a user message
2. Agent determines it needs weather information
3. Agent calls the MCP `handoff` tool with the weather agent's URI
4. MCP server uses sampling to call the weather agent via A2A
5. Weather agent responds with weather data
6. Personal assistant receives the response and presents it to the user

See `scripts/personal_assistant_app.py` and `scripts/weather_assistant_app.py` for the complete implementation.

## Architecture

### A2A Server

The A2A server implements a **Task-generating Agent** that:
- Always responds with Task objects (never just Messages)
- Uses `input-required` task state when tool calls are needed
- Includes tool calls in a `DataPart` within the task status message
- Manages task lifecycle (created → input-required → completed)

### MCP Server

The MCP server (implemented by `Environment` class) provides:
- **Tools**: Custom tools registered via `@environment.tool()` decorator
- **Built-in `handoff` tool**: Automatically registered when `enable_handoff=True` (default)
- **Sampling**: Server-initiated LLM interactions that trigger A2A requests
- Tool execution for standard operations (e.g., `get_weather`)

The `handoff` tool is built into the `Environment` class and uses MCP sampling to enable seamless agent-to-agent delegation.

### Client

The client orchestrates the interaction:
- Connects to A2A server to send messages
- Monitors task state transitions
- When `input-required` state is detected, extracts tool calls from `DataPart`
- Calls MCP tools and sends results back to A2A server
- Handles handoffs via MCP sampling callback (built into `Loop`)

The `Loop` component provides a `/v1/responses` endpoint that handles both streaming and non-streaming modes, with built-in handoff execution via MCP sampling.

## Key Concepts

### Task-Generating Agents

Following the [A2A Protocol's Task-generating Agents philosophy](https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task), our agents always respond with Task objects. This provides:

- **State management**: Tasks can be in various states (created, input-required, completed, etc.)
- **Progress tracking**: Task status updates provide visibility into agent progress
- **Multi-turn interactions**: Tasks support context IDs for grouping related interactions

### A2A input-required with DataPart

When an agent needs to call tools, it:
1. Sets the task state to `input-required`
2. Includes a `DataPart` in the task status message
3. The `DataPart` contains the tool calls in a structured format
4. The client extracts tool calls from the `DataPart` and executes them via MCP

### MCP Sampling for Handoffs

MCP's sampling feature allows servers to initiate LLM interactions. We use this for handoffs:

1. Agent calls MCP `handoff` tool with target agent URI
2. MCP server calls the client's sampling callback
3. Sampling callback makes an A2A request to the target agent
4. Target agent processes and responds
5. Response is returned to the MCP server
6. MCP server returns the result to the original agent

## Examples

See `scripts/` for complete working examples:
- `personal_assistant_app.py` - Personal assistant with handoff tool enabled
- `weather_assistant_app.py` - Weather assistant with handoff tool disabled
- `personal_assistant_test_client.py` - Test client for personal assistant
- `weather_assistant_test_client.py` - Test client for weather assistant

These examples demonstrate:
- Using the `Agent`, `Environment`, and `Loop` classes from `lib/python/core/`
- Conditional handoff tool registration via `enable_handoff` parameter
- Complete handoff flow between agents

## Documentation

- **A2A Protocol Specification**: https://a2a-protocol.org/latest/specification/
- **A2A Task-generating Agents**: https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task
- **MCP Protocol Specification**: https://modelcontextprotocol.io/specification/latest
- **Full docs**: https://timestep-ai.github.io/timestep/

## License

MIT License - see `LICENSE`.
