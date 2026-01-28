# AGENTS.md - Development Guide for AI Coding Agents

This document provides guidance for AI coding agents working on the Timestep project, including development environment setup, testing instructions, PR guidelines, and project-specific conventions.

## Development Environment

### Prerequisites

- **Python 3.11+** for Python development
- **Node.js 20+** for TypeScript development
- **Bun** for TypeScript examples (uses inline dependencies)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd timestep
   ```

2. **Python setup** (for examples):
   ```bash
   cd examples/python
   # Dependencies are inline in Python files using PEP 723
   ```

3. **TypeScript setup** (for examples):
   ```bash
   cd examples/typescript
   # Dependencies are auto-installed by Bun from import statements
   ```

4. **Environment variables**:
   ```bash
   export OPENAI_API_KEY="your-key-here"  # Required for A2A agents using OpenAI
   ```

## Project Structure

The codebase is organized around A2A and MCP protocols:

```
timestep/
├── lib/                    # Library code
│   ├── python/             # Python library
│   │   └── core/           # Core library components
│   │       ├── agent.py    # Agent class (A2A Server)
│   │       ├── environment.py  # Environment class (MCP Server)
│   │       ├── loop.py     # Loop class (AgentExecutor)
│   │       └── responses_api.py  # ResponsesAPI class (/v1/responses endpoint)
│   └── typescript/         # TypeScript library (planned)
├── scripts/                # Working example applications
│   ├── personal_assistant_app.py  # Personal assistant with handoff enabled
│   ├── weather_assistant_app.py   # Weather assistant with handoff disabled
│   ├── personal_assistant_test_client.py  # Test client for personal assistant
│   └── weather_assistant_test_client.py   # Test client for weather assistant
├── examples/               # Legacy examples directory (currently empty)
│   ├── python/             # (empty - examples moved to scripts/)
│   └── typescript/         # TypeScript examples (pending v2 SDK)
└── app/                    # Web UI
    ├── index.html
    ├── index.css
    └── index.js
```

**Important**: The library code is implemented in `lib/python/core/` with working examples in `scripts/`. The `examples/` directory is legacy and may be removed.

## Core Concepts

### A2A Protocol

The [Agent-to-Agent (A2A) Protocol](https://a2a-protocol.org/latest/specification/) standardizes how independent AI agents communicate and collaborate as peers. Key concepts:

- **Agent Card**: Discovery mechanism for agent capabilities and endpoints
- **Task**: Stateful interaction object that tracks progress
- **Task States**: `created`, `input-required`, `completed`, `canceled`, etc.
- **Context ID**: Groups related interactions across multiple tasks

### Task-Generating Agents

Following the [A2A Task-generating Agents philosophy](https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task), our agents always respond with Task objects (never just Messages). This provides:

- Clear state management
- Progress visibility
- Support for multi-turn interactions
- Structured tool call communication

### A2A input-required with DataPart

When an agent needs to call tools:

1. Agent sets task state to `input-required`
2. Agent includes a `DataPart` in the task status message
3. The `DataPart.data` contains tool calls in structured format:
   ```json
   {
     "tool_calls": [
       {
         "function": {
           "name": "handoff",
           "arguments": "{\"agent_uri\": \"...\", \"message\": \"...\"}"
         }
       }
     ]
   }
   ```
4. Client extracts tool calls from `DataPart` and executes via MCP

### A2A Message Structure and Flow

#### A2A Message Structure

A2A Messages use a `parts` array (not `content`) to represent message content. Each part can be:

- **TextPart**: Contains text content
  ```json
  {
    "text": "What's the weather in Oakland?"
  }
  ```

- **DataPart**: Contains structured data (tool calls, tool results)
  ```json
  {
    "data": {
      "tool_calls": [
        {
          "call_id": "call_abc123",
          "name": "get_weather",
          "arguments": {"location": "Oakland"}
        }
      ]
    }
  }
  ```

- **FilePart**: Contains file attachments (not used in current implementation)

A complete A2A Message structure:
```json
{
  "role": "user",  // or "agent"
  "parts": [
    {"text": "What's the weather in Oakland?"}
  ]
}
```

#### Complete Message Flow Example

This example traces a complete interaction from initial request through tool execution to final response:

**Step 1: Initial Request to `/v1/responses`**
```json
{
  "input": "What's the weather in Oakland?"
}
```

**Step 2: Converted to A2A Message (sent to agent)**
```json
{
  "role": "user",
  "parts": [
    {"text": "What's the weather in Oakland?"}
  ]
}
```

**Step 3: Agent Processes, Calls OpenAI, Gets Tool Calls**

OpenAI returns:
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"Oakland\"}"
      }
    }
  ]
}
```

**Step 4: Agent Emits A2A TaskStatusUpdateEvent with Tool Calls**

The agent emits a task with `input-required` state:
```json
{
  "id": "task_xyz789",
  "context_id": "context_123",
  "status": {
    "state": {"value": "input-required"},
    "message": {
      "role": "agent",
      "parts": [
        {
          "data": {
            "tool_calls": [
              {
                "call_id": "call_abc123",
                "name": "get_weather",
                "arguments": {"location": "Oakland"}
              }
            ]
          }
        }
      ]
    }
  },
  "history": [
    {
      "role": "user",
      "parts": [{"text": "What's the weather in Oakland?"}]
    },
    {
      "role": "agent",
      "parts": [
        {
          "data": {
            "tool_calls": [
              {
                "call_id": "call_abc123",
                "name": "get_weather",
                "arguments": {"location": "Oakland"}
              }
            ]
          }
        }
      ]
    }
  ]
}
```

**Step 5: Tool Results Come Back**

Tool execution result:
```json
{
  "result": "The weather in Oakland is sunny, 72°F"
}
```

**Step 6: Second A2A Message (with tool results)**

Client sends tool results back to agent:
```json
{
  "role": "user",
  "task_id": "task_xyz789",
  "context_id": "context_123",
  "parts": [
    {
      "data": {
        "tool_results": [
          {
            "call_id": "call_abc123",
            "name": "get_weather",
            "output": "The weather in Oakland is sunny, 72°F"
          }
        ]
      }
    }
  ]
}
```

**Step 7: Agent Processes Tool Results**

When the agent receives tool results:
- `context.current_task.history` should contain the assistant message with tool_calls from Step 4
- The agent builds OpenAI messages array from history + new tool results
- OpenAI messages array should look like:
  ```json
  [
    {"role": "system", "content": "You are a weather assistant..."},
    {"role": "user", "content": "What's the weather in Oakland?"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"Oakland\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "The weather in Oakland is sunny, 72°F"
    }
  ]
  ```

**Step 8: Agent Emits Final Response**

Agent emits completed task:
```json
{
  "id": "task_xyz789",
  "context_id": "context_123",
  "status": {
    "state": {"value": "completed"},
    "message": {
      "role": "agent",
      "parts": [
        {"text": "The weather in Oakland is sunny, 72°F"}
      ]
    }
  },
  "history": [
    // ... previous messages ...
    {
      "role": "user",
      "parts": [
        {
          "data": {
            "tool_results": [
              {
                "call_id": "call_abc123",
                "name": "get_weather",
                "output": "The weather in Oakland is sunny, 72°F"
              }
            ]
          }
        }
      ]
    },
    {
      "role": "agent",
      "parts": [
        {"text": "The weather in Oakland is sunny, 72°F"}
      ]
    }
  ]
}
```

#### Personal Assistant Handoff Flow Example

This example shows the same input message ("What's the weather in Oakland?") but sent to the personal assistant, which then hands off to the weather assistant:

**Step 1: Initial Request to Personal Assistant `/v1/responses`**
```json
{
  "input": "What's the weather in Oakland?"
}
```

**Step 2: Converted to A2A Message (sent to personal assistant)**
```json
{
  "role": "user",
  "parts": [
    {"text": "What's the weather in Oakland?"}
  ]
}
```

**Step 3: Personal Assistant Processes, Calls OpenAI, Gets Handoff Tool Call**

OpenAI returns:
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_handoff123",
      "type": "function",
      "function": {
        "name": "handoff",
        "arguments": "{\"agent_uri\": \"http://localhost:10000\", \"message\": \"What's the weather in Oakland?\"}"
      }
    }
  ]
}
```

**Step 4: Personal Assistant Emits A2A TaskStatusUpdateEvent with Handoff Tool Call**

The personal assistant emits a task with `input-required` state:
```json
{
  "id": "task_personal_789",
  "context_id": "personal-context",
  "status": {
    "state": {"value": "input-required"},
    "message": {
      "role": "agent",
      "parts": [
        {
          "data": {
            "tool_calls": [
              {
                "call_id": "call_handoff123",
                "name": "handoff",
                "arguments": {
                  "agent_uri": "http://localhost:10000",
                  "message": "What's the weather in Oakland?"
                }
              }
            ]
          }
        }
      ]
    }
  },
  "history": [
    {
      "role": "user",
      "parts": [{"text": "What's the weather in Oakland?"}]
    },
    {
      "role": "agent",
      "parts": [
        {
          "data": {
            "tool_calls": [
              {
                "call_id": "call_handoff123",
                "name": "handoff",
                "arguments": {
                  "agent_uri": "http://localhost:10000",
                  "message": "What's the weather in Oakland?"
                }
              }
            ]
          }
        }
      ]
    }
  ]
}
```

**Step 5: Handoff Tool Execution**

The personal assistant's MCP server calls the sampling callback, which:
1. Connects to weather assistant at `http://localhost:10000`
2. Sends A2A message to weather assistant:
   ```json
   {
     "role": "user",
     "parts": [
       {"text": "What's the weather in Oakland?"}
     ]
   }
   ```

**Step 6: Weather Assistant Processes Request**

The weather assistant follows the same flow as the direct weather assistant example (Steps 3-8), eventually returning:
```json
{
  "id": "task_weather_456",
  "context_id": "weather-context",
  "status": {
    "state": {"value": "completed"},
    "message": {
      "role": "agent",
      "parts": [
        {"text": "The current weather in Oakland is 72°F and sunny, with a humidity level of 65%."}
      ]
    }
  }
}
```

**Step 7: Handoff Returns Result to Personal Assistant**

The sampling callback extracts the final message from the weather assistant's completed task and returns it to the personal assistant's MCP server as a tool result:
```json
{
  "result": "The current weather in Oakland is 72°F and sunny, with a humidity level of 65%."
}
```

**Step 8: Personal Assistant Receives Tool Result**

The personal assistant receives the tool result as a user message:
```json
{
  "role": "user",
  "task_id": "task_personal_789",
  "context_id": "personal-context",
  "parts": [
    {
      "data": {
        "tool_results": [
          {
            "call_id": "call_handoff123",
            "name": "handoff",
            "output": "The current weather in Oakland is 72°F and sunny, with a humidity level of 65%."
          }
        ]
      }
    }
  ]
}
```

**Step 9: Personal Assistant Processes Tool Result**

When the personal assistant receives the tool result:
- `context.current_task.history` contains the assistant message with handoff tool_call from Step 4
- The agent builds OpenAI messages array from history + new tool results
- OpenAI messages array should look like:
  ```json
  [
    {"role": "system", "content": "You are a personal assistant..."},
    {"role": "user", "content": "What's the weather in Oakland?"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_handoff123",
          "type": "function",
          "function": {
            "name": "handoff",
            "arguments": "{\"agent_uri\": \"http://localhost:10000\", \"message\": \"What's the weather in Oakland?\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_handoff123",
      "content": "The current weather in Oakland is 72°F and sunny, with a humidity level of 65%."
    }
  ]
  ```

**Step 10: Personal Assistant Emits Final Response**

The personal assistant emits a completed task with the weather information:
```json
{
  "id": "task_personal_789",
  "context_id": "personal-context",
  "status": {
    "state": {"value": "completed"},
    "message": {
      "role": "agent",
      "parts": [
        {"text": "The current weather in Oakland is 72°F and sunny, with a humidity level of 65%."}
      ]
    }
  },
  "history": [
    {
      "role": "user",
      "parts": [{"text": "What's the weather in Oakland?"}]
    },
    {
      "role": "agent",
      "parts": [
        {
          "data": {
            "tool_calls": [
              {
                "call_id": "call_handoff123",
                "name": "handoff",
                "arguments": {
                  "agent_uri": "http://localhost:10000",
                  "message": "What's the weather in Oakland?"
                }
              }
            ]
          }
        }
      ]
    },
    {
      "role": "user",
      "parts": [
        {
          "data": {
            "tool_results": [
              {
                "call_id": "call_handoff123",
                "name": "handoff",
                "output": "The current weather in Oakland is 72°F and sunny, with a humidity level of 65%."
              }
            ]
          }
        }
      ]
    },
    {
      "role": "agent",
      "parts": [
        {"text": "The current weather in Oakland is 72°F and sunny, with a humidity level of 65%."}
      ]
    }
  ]
}
```

**Key Differences from Direct Weather Assistant Flow:**

1. **Tool Type**: Personal assistant uses `handoff` tool instead of `get_weather`
2. **Tool Execution**: Handoff tool uses MCP sampling to call another A2A agent
3. **Nested Agent Interaction**: Weather assistant processes the request independently
4. **Result Propagation**: Weather assistant's response is returned as tool result to personal assistant
5. **Final Response**: Personal assistant can modify or forward the weather assistant's response

#### Task History Semantics

**When Messages Are Added to History:**

- When `TaskStatusUpdateEvent` is emitted with a `message` field, the A2A SDK (`DefaultRequestHandler` + `InMemoryTaskStore`) automatically adds that message to the task's history
- The incoming user message (from `RequestContext.message`) is also added to history
- History is maintained per task, grouped by `context_id` and `task_id`

**What History Contains at Each Stage:**

1. **After Initial Request**: History contains the user message
2. **After Tool Calls Emitted**: History contains user message + agent message with tool_calls
3. **When Tool Results Arrive**: History should already contain the assistant message with tool_calls (from step 2)
4. **After Final Response**: History contains the complete conversation: user → agent (tool_calls) → user (tool_results) → agent (final response)

**System Behavior Guidelines:**

- **Task history should always contain the assistant message with tool_calls before tool results arrive**: When `Loop.execute()` is called with tool results, `context.current_task.history` must already contain the assistant message that emitted those tool calls
- **`context.current_task.history` should be populated when entering `Loop.execute()`**: The A2A SDK maintains task history automatically, so `context.current_task` should have the full history when `execute()` is called
- **History processing should not need to search or manually insert messages**: The history should be complete and correct, requiring only conversion from A2A format to OpenAI format

### MCP Protocol

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/specification/latest) provides:

- **Tools**: Functions for agents to execute
- **Sampling**: Server-initiated LLM interactions (used for handoffs)
- **Resources**: Context and data for agents
- **Prompts**: Templated messages and workflows

### MCP Sampling for Handoffs

MCP's sampling feature enables server-initiated agentic behaviors. We use it for handoffs:

1. Agent calls MCP `handoff` tool with target agent URI
2. MCP server calls client's sampling callback via `ctx.session.create_message()`
3. Sampling callback makes A2A request to target agent
4. Target agent processes and responds
5. Response returned to MCP server
6. MCP server returns result to original agent

This allows agents to delegate work to specialized peers without the original agent needing direct A2A client capabilities.

## Architecture

### A2A Server

The A2A server implements a Task-generating Agent:

- Uses A2A SDK to handle protocol operations
- Always responds with Task objects
- Manages task lifecycle and state transitions
- Includes tool calls in `DataPart` when `input-required`
- Supports streaming for real-time updates

**Task History Management:**

The A2A server uses `DefaultRequestHandler` and `InMemoryTaskStore` from the A2A SDK to manage task history:

- **`DefaultRequestHandler`**: Handles incoming A2A requests and manages the request lifecycle
- **`InMemoryTaskStore`**: Stores tasks and maintains their history in memory
- **Automatic History Updates**: When `TaskStatusUpdateEvent` is emitted via `EventQueue.enqueue_event()`, the A2A SDK automatically:
  - Adds the event's `status.message` to the task's history
  - Updates the task's status
  - Makes the updated task available via `RequestContext.current_task`

**RequestContext and Task History:**

- When `Loop.execute()` is called, `RequestContext.current_task` contains the current task state
- `context.current_task.history` contains all messages that have been added to the task
- The incoming message (`context.message`) will be added to history by the A2A SDK after processing
- History is keyed by `task_id` and `context_id`, allowing multi-turn conversations within the same context

### MCP Server

The MCP server (implemented by `Environment` class) provides:

- Tool registration (e.g., `handoff`, `get_weather`)
- Built-in `handoff` tool (conditionally registered via `enable_handoff` parameter)
- Sampling callback registration (for handoffs)
- HTTP transport for client connections
- Tool execution and result formatting

**Built-in Handoff Tool:**

The `handoff` tool is built into the `Environment` class and is registered by default. It can be disabled by setting `enable_handoff=False` when creating an Environment instance. The tool uses MCP sampling to call other agents via A2A, enabling seamless agent-to-agent delegation.

### Client

The client orchestrates A2A and MCP:

- Connects to A2A server to send messages
- Monitors task state transitions
- Extracts tool calls from `DataPart` when `input-required`
- Calls MCP tools and sends results back
- Implements MCP sampling callback for handoffs

## Testing

### Running Examples

**Python:**

The library includes working examples in the `scripts/` directory. To run the personal assistant and weather assistant example:

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
make test-example-typescript  # Currently shows pending v2 message
```

### Test Organization

- Examples are in `scripts/` directory
- Each example includes an A2A server application and a test client
- The library components (`Agent`, `Environment`, `Loop`, `ResponsesAPI`) are in `lib/python/core/`

## Code Style and Conventions

### Python

- Follow PEP 8 style guide
- Use type hints for all function signatures
- Docstrings should follow Google style
- Use inline dependencies (PEP 723) in script files

### TypeScript

- Use TypeScript strict mode
- Prefer `async`/`await` over promises
- Use JSDoc comments for documentation
- Follow the existing code style (2-space indentation)
- Use Bun's auto-install for dependencies (no package.json)

### Cross-Language Parity

**Critical**: When adding features, ensure both Python and TypeScript implementations:
- Have the same API surface
- Use the same function/class names
- Follow the same parameter naming conventions
- Produce compatible A2A Task and MCP message formats

## Pull Request Guidelines

### Before Submitting

1. **Run examples**: Ensure Python examples work end-to-end
2. **Check protocol compliance**: Verify A2A and MCP protocol adherence
3. **Update documentation**: Update relevant docs if adding features
4. **Test handoffs**: Verify handoff flow works correctly

### PR Checklist

- [ ] Examples run successfully (Python)
- [ ] Code follows project conventions
- [ ] A2A and MCP protocol compliance verified
- [ ] Documentation updated (if needed)
- [ ] Handoff functionality tested
- [ ] No breaking changes (or clearly documented)

### Commit Messages

Use clear, descriptive commit messages:
- `feat: Add new MCP tool for X`
- `fix: Resolve issue with A2A handoff`
- `refactor: Reorganize MCP server structure`
- `docs: Update README with A2A/MCP details`

## Common Tasks

### Adding a New MCP Tool

1. Add tool function to your Environment instance using `@environment.tool()` decorator
2. The tool will automatically be available to the agent via MCP
3. Test tool execution via client or test scripts
4. Update documentation

Example:
```python
@environment.tool()
async def my_tool(param: str) -> dict:
    """Tool description."""
    return {"result": param}
```

### Adding a New Agent

1. Create an `Environment` instance with agent-specific configuration
2. Add tools to the environment using `@environment.tool()` decorator
3. Create an `Agent` instance with the environment URI
4. Optionally add `ResponsesAPI` for `/v1/responses` endpoint
5. Start the agent server
6. Test agent via client or test scripts
7. Update documentation

See `scripts/personal_assistant_app.py` and `scripts/weather_assistant_app.py` for complete examples.

### Implementing Handoffs

1. Create an `Environment` with `enable_handoff=True` (default)
2. The `handoff` tool is automatically registered in the Environment
3. Create an `Agent` with `ResponsesAPI` (which includes built-in handoff execution)
4. The handoff tool uses MCP sampling to call target agents via A2A
5. Test handoff flow end-to-end using test clients
6. Verify task state transitions

The handoff functionality is built into the library - no additional setup required. See `scripts/personal_assistant_app.py` for a complete example.

## OpenTelemetry Tracing

The Timestep library includes built-in support for OpenTelemetry tracing using zero-code instrumentation. This provides automatic tracing of FastAPI requests, HTTP clients, OpenAI API calls, A2A protocol operations, and MCP tool calls.

### Quick Start with Jaeger

1. **Start Jaeger all-in-one**:
   ```bash
   docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
   ```

2. **Run your application** - traces are automatically sent to Jaeger

3. **View traces** at http://localhost:16686

### Configuration

Tracing is automatically enabled when OpenTelemetry packages are installed. Configure via environment variables:

- `OTEL_ENABLED` (default: `true`) - Enable/disable tracing
- `OTEL_SERVICE_NAME` (default: `timestep`) - Service name for traces
- `OTEL_EXPORTER_OTLP_ENDPOINT` (default: `http://localhost:4317`) - OTLP endpoint URL

### Usage

Traces are exported via OTLP to Jaeger or other compatible backends. The library automatically instruments:
- FastAPI applications
- HTTP clients (httpx, requests)
- OpenAI API calls
- A2A protocol (via `a2a-sdk[telemetry]`)
- MCP protocol operations

### Programmatic Setup

```python
from timestep.observability.tracing import enable_tracing

# Enable tracing and instrument FastAPI app in one call
enable_tracing(app, service_name="my-service", otlp_endpoint="http://localhost:4317")
```

See `lib/python/README.md` for more details on tracing configuration and usage.

## Resources

- **A2A Protocol Specification**: https://a2a-protocol.org/latest/specification/
- **A2A Task-generating Agents**: https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task
- **MCP Protocol Specification**: https://modelcontextprotocol.io/specification/latest
- **Documentation**: https://timestep-ai.github.io/timestep/
- **OpenTelemetry Python**: https://opentelemetry.io/docs/languages/python/

## Notes for AI Agents

- **Always maintain cross-language parity**: Changes in one language should be reflected in the other
- **Follow A2A and MCP specifications**: Ensure protocol compliance
- **Test handoffs**: Verify handoff functionality when making changes
- **Keep it simple**: The library is intentionally minimal - avoid over-engineering
- **Do NOT create documentation files unless explicitly requested**: Only create markdown documentation files (like README, architecture docs, etc.) when the user explicitly asks for them. Do not create evaluation documents, mapping documents, or other analysis documents unless specifically requested.
