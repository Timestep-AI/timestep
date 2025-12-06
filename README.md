# Timestep AI Agents SDK

A clean, RISC-style abstraction layer over the OpenAI Agents SDK, providing durable execution and cross-language state persistence for AI agent workflows.

## Architecture Philosophy: RISC vs CISC

Timestep is built on the principle that the OpenAI Agents SDK is like a **CISC (Complex Instruction Set Computer)** architecture—powerful and feature-rich, but with many abstractions and layers that can be complex to work with. Timestep provides a **RISC (Reduced Instruction Set Computer)** approach—a simpler, cleaner interface that exposes the fundamental components needed for building robust agent systems.

### What This Means

- **Simpler API**: Clean abstractions that focus on essential operations
- **Durable Execution**: Built-in support for state persistence and resumable workflows
- **Cross-Language Parity**: Identical behavior and APIs in Python and TypeScript
- **Composable**: Fundamental building blocks that can be combined as needed

## Core Features

### 🔄 Durable Execution

Timestep enables durable, resumable agent workflows with built-in state persistence. You can:

- **Interrupt and Resume**: Pause agent execution at any point and resume later
- **Cross-Language State Transfer**: Start in Python, interrupt for tool approval, and resume in TypeScript (or vice versa)
- **Persistent Sessions**: Agent state is stored in a database (DBOS/PGLite) for durability
- **Human-in-the-Loop**: Seamlessly handle interruptions for approvals, tool calls, or user input

### 🌐 Cross-Language State Persistence

Timestep provides identical implementations in Python and TypeScript with full state compatibility:

```python
# Python: Start agent execution
from timestep import run_agent, RunStateStore
from agents import Agent

agent = Agent(model="gpt-4")
state_store = RunStateStore("state.json", agent)

# Run agent and handle interruption
result = await run_agent(agent, input_items, session, stream=False)
if result.interruptions:
    # Save state for cross-language resume
    await state_store.save(result.state)
    # State can now be loaded in TypeScript!
```

```typescript
// TypeScript: Resume from Python state
import { runAgent, RunStateStore } from '@timestep-ai/timestep';
import { Agent } from '@openai/agents';

const agent = new Agent({ model: 'gpt-4' });
const stateStore = new RunStateStore('state.json', agent);

// Load state saved from Python
const savedState = await stateStore.load();

// Approve interruptions and continue
for (const interruption of savedState.getInterruptions()) {
  savedState.approve(interruption);
}

// Resume execution
const result = await runAgent(agent, savedState, session, false);
```

### 🔌 Multi-Model Provider Support

Timestep supports multiple AI model providers through a unified interface:

- **OpenAI**: Full support for all OpenAI models
- **Ollama**: Support for both local Ollama instances and Ollama Cloud
- **Automatic Routing**: Model names with prefixes (e.g., `ollama/llama3`) automatically route to the correct provider
- **Extensible**: Add custom providers using `MultiModelProviderMap`

## Packages

- **[Python Package (`timestep`)](./python/)** - [PyPI](https://pypi.org/project/timestep/)
- **[TypeScript Package (`@timestep-ai/timestep`)](./typescript/)** - [npm](https://www.npmjs.com/package/@timestep-ai/timestep)

## Quick Start

### Python

```bash
pip install timestep
```

```python
from timestep import run_agent, RunStateStore, MultiModelProvider, OllamaModelProvider
from agents import Agent, Session
import os

# Setup model provider
model_provider = MultiModelProvider(
    openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
)

# Create agent
agent = Agent(model="gpt-4")

# Run with durable state
session = Session()
state_store = RunStateStore("agent_state.json", agent)

result = await run_agent(agent, input_items, session, stream=False)

# Handle interruptions
if result.interruptions:
    # Save state for later resume (even in TypeScript!)
    await state_store.save(result.state)
```

### TypeScript

```bash
npm install @timestep-ai/timestep
```

```typescript
import { runAgent, RunStateStore, MultiModelProvider } from '@timestep-ai/timestep';
import { Agent, Session } from '@openai/agents';

// Setup model provider
const modelProvider = new MultiModelProvider({
  openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
});

// Create agent
const agent = new Agent({ model: 'gpt-4' });

// Run with durable state
const session = new Session();
const stateStore = new RunStateStore('agent_state.json', agent);

const result = await runAgent(agent, inputItems, session, false);

// Handle interruptions
if (result.interruptions) {
  // Save state for later resume (even in Python!)
  await stateStore.save(result.state);
}
```

## Durable Execution with DBOS/PGLite

Timestep uses **DBOS** (Database Operating System) and **PGLite** (PostgreSQL in WebAssembly) to provide durable execution:

- **State Persistence**: Agent run states are stored in PostgreSQL
- **Cross-Language Compatibility**: State format is identical between Python and TypeScript
- **Resumable Workflows**: Load any saved state and continue execution
- **Database Schema**: See [database/README.md](./database/README.md) for the complete schema

The database schema supports:
- Agent definitions and configurations
- Run state snapshots for resumability
- Session history and conversation items
- Tool calls and interruptions
- Usage metrics and cost tracking

## Components

### Core Abstractions

- **`run_agent()`**: Simplified agent execution with built-in state management
- **`RunStateStore`**: Persistent storage for agent state (file-based or database-backed)
- **`consume_result()`**: Utility for handling streaming and non-streaming results

### Model Providers

- **`MultiModelProvider`**: Automatic routing to OpenAI or Ollama based on model name prefixes
- **`OllamaModelProvider`**: Direct access to Ollama models (local or cloud)
- **`MultiModelProviderMap`**: Custom provider mappings

## Features

- ✅ **Durable Execution**: Interrupt and resume agent workflows across languages
- ✅ **Cross-Language State Persistence**: Seamless state transfer between Python and TypeScript
- ✅ **Multi-Model Support**: OpenAI and Ollama (local or cloud) through unified interface
- ✅ **RISC-Style API**: Clean, simple abstractions over the OpenAI Agents SDK
- ✅ **Identical Behavior**: Python and TypeScript implementations work identically
- ✅ **Database-Backed**: DBOS/PGLite integration for production-ready durability

## Development

### Running Tests

Run tests for both Python and TypeScript implementations:

```bash
# Run all tests (including cross-language tests)
make test-all

# Run individually
make test-python
make test-typescript
make test-cross-language
```

Cross-language tests verify that state saved in one language can be loaded and resumed in the other.

### Bumping Versions

To bump the patch version for both Python and TypeScript packages:

```bash
make patch
```

## Documentation

- **[Full Documentation](https://timestep-ai.github.io/timestep/)** - Comprehensive documentation with API reference, guides, and examples
- [Python Package Documentation](./python/README.md)
- [TypeScript Package Documentation](./typescript/README.md)
- [Database Schema](./database/README.md) - Schema for durable execution

## Future Plans

We're actively developing additional features:

- **Enhanced DBOS Integration**: Full integration with DBOS for distributed durable execution
- **Additional Abstractions**: More RISC-style components extracted from complex SDK patterns
- **CLI Tool**: Command-line interface with tracing support for debugging and monitoring
- **Advanced State Management**: More sophisticated state persistence and recovery patterns

## License

MIT License - see [LICENSE](LICENSE) file for details.
