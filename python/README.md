# Timestep

Streaming agent implementation using OpenAI's streaming API with tool support.

## Installation

```bash
pip install timestep
```

## Quick Start

### Basic Usage

```python
from timestep import run_agent
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Basic usage
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What's 2+2?"},
]

response = await run_agent(messages)
print(response)  # "2 + 2 = 4"
```

### Using Tools

Tools are Pydantic models that define the tool schema:

```python
from timestep import run_agent, GetWeather, WebSearch
from pydantic import BaseModel, Field

# Use built-in tools
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant that can answer questions about weather.",
    },
    {"role": "user", "content": "What's the weather in Oakland?"},
]

tools = [GetWeather]
response = await run_agent(messages, tools=tools)
print(response)

# Create a custom tool
class MyCustomTool(BaseModel):
    """Description of what this tool does."""
    arg1: str = Field(..., description="Description of arg1")
    arg2: int = Field(..., description="Description of arg2")

MyCustomTool.__name__ = "my_custom_tool"

# Register your tool's execute function
from timestep.core import call_function

async def my_custom_tool_execute(args: dict) -> str:
    return f"Result: {args['arg1']} and {args['arg2']}"

# Add to call_function mapping (you'll need to modify tools.py)
tools = [MyCustomTool]
response = await run_agent(messages, tools=tools)
```

### Conversation Context

The `run_agent` function maintains conversation context by modifying the messages array:

```python
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What's 2+2?"},
]

# First message (run_agent appends assistant response to messages)
response1 = await run_agent(messages)
print(response1)  # "2 + 2 = 4"

# Follow-up message (messages now includes the previous assistant response)
messages.append({"role": "user", "content": "What's three times that number?"})
response2 = await run_agent(messages)
print(response2)  # "Three times 4 is 12"
```

### Streaming and Tool Approval

```python
from timestep import run_agent

messages = [
    {"role": "user", "content": "What's the weather in San Francisco?"},
]

# Streaming callback
def on_delta(delta: dict):
    if "content" in delta:
        print(delta["content"], end="", flush=True)

# Tool approval callback
async def on_tool_approval_required(tool_call: dict) -> bool:
    print(f"Tool call requested: {tool_call['function']['name']}")
    # Return True to approve, False to reject
    return True

response = await run_agent(
    messages,
    tools=[GetWeather],
    on_delta=on_delta,
    on_tool_approval_required=on_tool_approval_required,
)
```

## API Reference

### `run_agent`

Run agent with streaming OpenAI API and tool support.

**Parameters:**
- `messages` (list[ChatCompletionMessageParam], required): List of message dictionaries with 'role' and 'content'
- `tools` (list[type[BaseModel]], optional): List of Pydantic models defining tool schemas
- `model` (str, optional): OpenAI model name (default: "gpt-4.1")
- `api_key` (str, optional): OpenAI API key (defaults to OPENAI_API_KEY env var)
- `on_delta` (Callable[[dict], None], optional): Callback for streaming deltas: `on_delta(delta_dict)`
- `on_tool_approval_required` (Callable[[dict], Awaitable[bool]], optional): Async callback for tool approval: `on_tool_approval_required(tool_call) -> bool`

**Returns:**
- `str`: Final assistant response as string

**Note:** The function modifies the `messages` array in place, appending assistant responses and tool results.

### Tools

Tools are Pydantic models that define the tool schema. The model class name should be set to the tool name:

```python
from pydantic import BaseModel, Field

class MyTool(BaseModel):
    """Tool description for the LLM."""
    param1: str = Field(..., description="Parameter description")

MyTool.__name__ = "my_tool"
```

The tool's execute function must be registered in `timestep.tools.call_function`.

### Built-in Tools

- **GetWeather**: Simple weather tool (example implementation)
- **WebSearch**: Web search using Firecrawl (requires `FIRECRAWL_API_KEY` environment variable)

## A2A Protocol Support

Timestep includes full support for the [Agent2Agent (A2A) Protocol](https://a2a-protocol.org/), enabling your agent to communicate with other A2A-compatible agents and systems.

### Running the A2A Server

**Command line (recommended):**

```bash
# Run the A2A server
timestep-a2a --host 0.0.0.0 --port 8080 --model gpt-4.1

# Or use defaults (0.0.0.0:8080, gpt-4.1)
timestep-a2a

# Show help
timestep-a2a --help
```

**Programmatically:**

```python
from timestep.a2a import run_server

# Start the A2A server
run_server(host="0.0.0.0", port=8080, model="gpt-4.1")
```

**What happens when you run it?**

1. Server starts on the specified host/port
2. Agent Card is available at `http://localhost:8080/.well-known/agent-card.json`
3. You can send tasks via HTTP POST to `/tasks`
4. Task updates stream via Server-Sent Events (SSE)
5. Other A2A agents can discover and interact with your agent

### A2A Features

- **Task-generating Agents**: All agent responses are encapsulated in `Task` objects, following the A2A "Task-generating Agents" philosophy
- **Human-in-the-loop**: Tool calls can require approval using the `input-required` task status
- **Streaming Updates**: Real-time task status updates via Server-Sent Events (SSE)
- **Agent Skills**: Tools are automatically exposed as `AgentSkill` objects in the Agent Card
- **Agent Card**: Discoverable agent metadata at `/.well-known/agent-card.json`

### Agent Card

Once the server is running, access the Agent Card at:
```
http://localhost:8080/.well-known/agent-card.json
```

The Agent Card includes:
- Agent name, description, and version
- Available skills (mapped from tools)
- Supported input/output modes
- Streaming capabilities
- Example queries

### Customizing the A2A Server

```python
from timestep.a2a import create_server, TimestepAgentExecutor
from timestep.core import GetWeather, WebSearch
import uvicorn

# Create server with custom tools
app = create_server(
    host="0.0.0.0",
    port=8080,
    tools=[GetWeather, WebSearch],
    model="gpt-4.1"
)

# Run with uvicorn
uvicorn.run(app, host="0.0.0.0", port=8080)
```

For more information about the A2A Protocol, visit [https://a2a-protocol.org/](https://a2a-protocol.org/).

## Testing

Timestep includes comprehensive test suites for both the core agent functionality and A2A server integration.

### Running Tests

```bash
# Run all Python tests
make test-python

# Or run directly with pytest
cd python
uv run python -m pytest tests/ -v
```

### Test Coverage

The test suite includes:
- **Core Agent Tests** (`test_run_agent.py`): Tests for the `run_agent` function with tools and streaming
- **A2A Server Tests** (`test_a2a_server.py`): Tests for A2A server setup and agent card generation
- **A2A Agent Executor Tests** (`test_a2a_agent_executor.py`): Tests for the A2A agent executor implementation
- **A2A Integration Tests** (`test_a2a_integration.py`): End-to-end integration tests for A2A protocol

## Requirements

- Python >=3.11
- openai >=1.0.0
- pydantic >=2.0.0
- firecrawl-py >=1.0.0 (for WebSearch tool)
- a2a-sdk[http-server] >=0.1.0 (for A2A Protocol support)
- uvicorn >=0.27.0 (for A2A server)
- starlette >=0.50.0 (for A2A server)

## License

MIT
