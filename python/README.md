# Timestep (Python)

A clean, RISC-style abstraction layer over the OpenAI Agents SDK, providing durable execution and cross-language state persistence for AI agent workflows.

## Architecture Philosophy

Timestep provides a **RISC (Reduced Instruction Set Computer)** approach to the OpenAI Agents SDK's **CISC (Complex Instruction Set Computer)** architecture. This means:

- **Simpler API**: Clean abstractions focused on essential operations
- **Durable Execution**: Built-in state persistence and resumable workflows
- **Cross-Language Parity**: Identical behavior with the TypeScript implementation
- **Composable**: Fundamental building blocks that can be combined as needed

## Installation

```bash
pip install timestep
```

Or using uv:

```bash
uv add timestep
```

## Quick Start

### Durable Execution with State Persistence

The core feature of Timestep is durable execution with cross-language state persistence:

```python
from timestep import run_agent, RunStateStore, consume_result
from agents import Agent, Session

# Create agent
agent = Agent(model="gpt-4")

# Create session and state store
session = Session()
state_store = RunStateStore("agent_state.json", agent)

# Run agent
result = await run_agent(agent, input_items, session, stream=False)
result = await consume_result(result)

# Handle interruptions
if result.interruptions:
    # Save state for later resume (even in TypeScript!)
    state = result.to_state()
    await state_store.save(state)
    
    # Load state and approve interruptions
    loaded_state = await state_store.load()
    for interruption in loaded_state.get_interruptions():
        loaded_state.approve(interruption)
    
    # Resume execution
    result = await run_agent(agent, loaded_state, session, stream=False)
    result = await consume_result(result)
```

### Cross-Language State Transfer

Start execution in Python, interrupt for tool approval, and resume in TypeScript:

```python
# Python: Start agent and save state at interruption
from timestep import run_agent, RunStateStore
from agents import Agent, Session

agent = Agent(model="gpt-4")
session = Session()
state_store = RunStateStore("cross_lang_state.json", agent)

# Run until interruption
result = await run_agent(agent, input_items, session, stream=False)
result = await consume_result(result)

if result.interruptions:
    # Save state - can be loaded in TypeScript!
    state = result.to_state()
    await state_store.save(state)
    print(f"State saved. Resume in TypeScript with session_id: {session._get_session_id()}")
```

Then in TypeScript:

```typescript
// TypeScript: Load Python state and resume
import { runAgent, RunStateStore } from '@timestep-ai/timestep';
import { Agent, Session } from '@openai/agents';

const agent = new Agent({ model: 'gpt-4' });
const session = new Session();
const stateStore = new RunStateStore('cross_lang_state.json', agent);

// Load state saved from Python
const savedState = await stateStore.load();

// Approve interruptions
for (const interruption of savedState.getInterruptions()) {
  savedState.approve(interruption);
}

// Resume execution
const result = await runAgent(agent, savedState, session, false);
```

### Multi-Model Provider Support

Timestep also provides multi-model provider support for OpenAI and Ollama:

```python
from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
from agents import Agent, Runner, RunConfig
import os

# Create a provider map and add Ollama support
model_provider_map = MultiModelProviderMap()

if os.environ.get("OLLAMA_API_KEY"):
    model_provider_map.add_provider(
        "ollama",
        OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
    )

# Create MultiModelProvider with OpenAI fallback
model_provider = MultiModelProvider(
    provider_map=model_provider_map,
    openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
)

# Create agent with model name
agent = Agent(model="gpt-4")  # Uses OpenAI by default
# Or: agent = Agent(model="ollama/llama3")  # Uses Ollama

# Run agent with RunConfig
run_config = RunConfig(model_provider=model_provider)
result = Runner.run_streamed(agent, agent_input, run_config=run_config)
```

## Core Components

### `run_agent()`

Simplified agent execution with built-in state management:

```python
from timestep import run_agent
from agents import Agent, Session

agent = Agent(model="gpt-4")
session = Session()

# Run with streaming
result = await run_agent(agent, input_items, session, stream=True)

# Run without streaming
result = await run_agent(agent, input_items, session, stream=False)
```

### `RunStateStore`

Persistent storage for agent state. Supports file-based storage (with database-backed storage coming via DBOS/PGLite):

```python
from timestep import RunStateStore
from agents import Agent

agent = Agent(model="gpt-4")
state_store = RunStateStore("state.json", agent)

# Save state
await state_store.save(state)

# Load state
loaded_state = await state_store.load()

# Clear state
await state_store.clear()
```

### `consume_result()`

Utility for handling both streaming and non-streaming results:

```python
from timestep import run_agent, consume_result

result = await run_agent(agent, input_items, session, stream=True)
result = await consume_result(result)  # Ensures all stream events are consumed

# Now safe to access result properties
print(result.final_output)
```

### `InterruptionException`

Exception raised when agent execution is interrupted for approval:

```python
from timestep import InterruptionException

try:
    result = await run_agent(agent, input_items, session, stream=False)
except InterruptionException as e:
    # Handle interruption
    pass
```

## Model Providers

### MultiModelProvider

Automatically routes model requests to the appropriate provider based on model name prefixes:

```python
from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider

# Create a custom mapping
provider_map = MultiModelProviderMap()

# Add Ollama provider
if os.environ.get("OLLAMA_API_KEY"):
    provider_map.add_provider(
        "ollama",
        OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
    )

# Use in MultiModelProvider
model_provider = MultiModelProvider(
    provider_map=provider_map,
    openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
)
```

**Features:**
- Automatic provider selection based on model name prefix
- Default fallback to OpenAI for unprefixed models
- Support for custom provider mappings

### OllamaModelProvider

Provides access to Ollama models (local or cloud):

```python
from timestep import OllamaModelProvider

# Create an Ollama provider for local Ollama instance
ollama_provider = OllamaModelProvider()  # Defaults to localhost:11434

# For Ollama Cloud, use the API key
cloud_provider = OllamaModelProvider(api_key="your-ollama-cloud-key")
```

**Options:**
- `api_key` (str, optional): API key for Ollama Cloud
- `base_url` (str, optional): Base URL for Ollama instance (defaults to `http://localhost:11434` for local, `https://ollama.com` for cloud)
- `ollama_client` (Any, optional): Custom Ollama client instance

**Features:**
- Lazy client initialization (only loads when needed)
- Automatic cloud detection for models ending with `-cloud`
- Support for both local Ollama instances and Ollama Cloud
- Seamless switching between local and cloud models

### OllamaModel

Direct model implementation that converts Ollama responses to OpenAI-compatible format:

**Features:**
- Converts Ollama API responses to OpenAI format
- Supports streaming responses
- Handles tool calls and function calling
- Compatible with OpenAI Agents SDK

### MultiModelProviderMap

Manages custom mappings of model name prefixes to providers:

**Methods:**
- `add_provider(prefix, provider)`: Add a prefix-to-provider mapping
- `remove_provider(prefix)`: Remove a mapping
- `get_provider(prefix)`: Get provider for a prefix
- `has_prefix(prefix)`: Check if prefix exists
- `get_mapping()`: Get all mappings
- `set_mapping(mapping)`: Replace all mappings

## Durable Execution with DBOS/PGLite

Timestep uses **DBOS** (Database Operating System) and **PGLite** (PostgreSQL in WebAssembly) to provide durable execution:

- **State Persistence**: Agent run states are stored in PostgreSQL
- **Cross-Language Compatibility**: State format is identical between Python and TypeScript
- **Resumable Workflows**: Load any saved state and continue execution
- **Database Schema**: See [database/README.md](../database/README.md) for the complete schema

The database schema supports:
- Agent definitions and configurations
- Run state snapshots for resumability
- Session history and conversation items
- Tool calls and interruptions
- Usage metrics and cost tracking

## Model Naming Conventions

Timestep uses model name prefixes to determine which provider to use:

- **No prefix** (e.g., `gpt-4`): Defaults to OpenAI
- **`openai/` prefix** (e.g., `openai/gpt-4`): Explicitly uses OpenAI
- **`ollama/` prefix** (e.g., `ollama/llama3`): Uses Ollama

## Features

- ✅ **Durable Execution**: Interrupt and resume agent workflows
- ✅ **Cross-Language State Persistence**: Seamless state transfer with TypeScript
- ✅ **Multi-Model Support**: Seamlessly switch between OpenAI and Ollama models
- ✅ **Automatic Routing**: Model names with prefixes automatically route to the correct provider
- ✅ **Customizable**: Add your own providers using `MultiModelProviderMap`
- ✅ **OpenAI Compatible**: Works with the OpenAI Agents SDK
- ✅ **Ollama Integration**: Full support for both local Ollama instances and Ollama Cloud
- ✅ **RISC-Style API**: Clean, simple abstractions over the OpenAI Agents SDK

## Requirements

- Python >=3.11
- ollama >=0.6.0
- openai-agents >=0.4.2

## Examples

See the [tests](./tests/) directory for complete examples including:
- Basic agent execution
- Durable execution with state persistence
- Cross-language state transfer (Python → TypeScript)
- Same-language state persistence (Python → Python)

## Future Plans

We're actively developing additional features:

- **Enhanced DBOS Integration**: Full integration with DBOS for distributed durable execution
- **Additional Abstractions**: More RISC-style components extracted from complex SDK patterns
- **CLI Tool**: Command-line interface with tracing support for debugging and monitoring

## License

MIT
