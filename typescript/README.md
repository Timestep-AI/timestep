# @timestep-ai/timestep (TypeScript)

A clean, RISC-style abstraction layer over the OpenAI Agents SDK, providing durable execution and cross-language state persistence for AI agent workflows.

## Architecture Philosophy

Timestep provides a **RISC (Reduced Instruction Set Computer)** approach to the OpenAI Agents SDK's **CISC (Complex Instruction Set Computer)** architecture. This means:

- **Simpler API**: Clean abstractions focused on essential operations
- **Durable Execution**: Built-in state persistence and resumable workflows
- **Cross-Language Parity**: Identical behavior with the Python implementation
- **Composable**: Fundamental building blocks that can be combined as needed

## Installation

```bash
npm install @timestep-ai/timestep
# or
pnpm add @timestep-ai/timestep
# or
yarn add @timestep-ai/timestep
```

## Quick Start

### Durable Execution with State Persistence

The core feature of Timestep is durable execution with cross-language state persistence:

```typescript
import { runAgent, RunStateStore, consumeResult } from '@timestep-ai/timestep';
import { Agent, Session } from '@openai/agents';

// Create agent
const agent = new Agent({ model: 'gpt-4' });

// Create session and state store
// RunStateStore uses PGLite by default (stored in ~/.config/timestep/pglite/)
const session = new Session();
const stateStore = new RunStateStore({ 
  agent, 
  sessionId: await session.getSessionId() 
});

// Run agent
let result = await runAgent(agent, inputItems, session, false);
result = await consumeResult(result);

// Handle interruptions
if (result.interruptions?.length) {
  // Save state for later resume (even in Python!)
  await stateStore.save(result.state);
  
  // Load state and approve interruptions
  const loadedState = await stateStore.load();
  for (const interruption of loadedState.getInterruptions()) {
    loadedState.approve(interruption);
  }
  
  // Resume execution
  result = await runAgent(agent, loadedState, session, false);
  result = await consumeResult(result);
}
```

### Cross-Language State Transfer

Start execution in TypeScript, interrupt for tool approval, and resume in Python:

```typescript
// TypeScript: Start agent and save state at interruption
import { runAgent, RunStateStore } from '@timestep-ai/timestep';
import { Agent, Session } from '@openai/agents';

const agent = new Agent({ model: 'gpt-4' });
const session = new Session();
// Uses PGLite in shared app directory (~/.config/timestep)
const stateStore = new RunStateStore({ 
  agent, 
  sessionId: await session.getSessionId() 
});

// Run until interruption
let result = await runAgent(agent, inputItems, session, false);
result = await consumeResult(result);

if (result.interruptions?.length) {
  // Save state - can be loaded in Python!
  await stateStore.save(result.state);
  const sessionId = await session.getSessionId();
  console.log(`State saved. Resume in Python with session_id: ${sessionId}`);
}
```

Then in Python:

```python
# Python: Load TypeScript state and resume
from timestep import run_agent, RunStateStore
from agents import Agent, Session

agent = Agent(model="gpt-4")
session = Session()
state_store = RunStateStore("cross_lang_state.json", agent)

# Load state saved from TypeScript
saved_state = await state_store.load()

# Approve interruptions
for interruption in saved_state.get_interruptions():
    saved_state.approve(interruption)

# Resume execution
result = await run_agent(agent, saved_state, session, False)
```

### Multi-Model Provider Support

Timestep also provides multi-model provider support for OpenAI and Ollama:

```typescript
import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';
import { Agent, Runner } from '@openai/agents';

// Create a provider map and add Ollama support
const modelProviderMap = new MultiModelProviderMap();

if (Deno.env.get('OLLAMA_API_KEY')) {
  modelProviderMap.addProvider(
    'ollama',
    new OllamaModelProvider({
      apiKey: Deno.env.get('OLLAMA_API_KEY'),
    })
  );
}

// Create MultiModelProvider with OpenAI fallback
const modelProvider = new MultiModelProvider({
  provider_map: modelProviderMap,
  openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
});

// Create agent with model name
const agent = new Agent({ model: 'gpt-4' }); // Uses OpenAI by default
// Or: new Agent({ model: 'ollama/llama3' }) // Uses Ollama

// Run agent with Runner
const runner = new Runner({ modelProvider });
const result = await runner.run(agent, agentInput, { stream: true });
```

## Core Components

### `runAgent()`

Simplified agent execution with built-in state management:

```typescript
import { runAgent } from '@timestep-ai/timestep';
import { Agent, Session } from '@openai/agents';

const agent = new Agent({ model: 'gpt-4' });
const session = new Session();

// Run with streaming
const result = await runAgent(agent, inputItems, session, true);

// Run without streaming
const result = await runAgent(agent, inputItems, session, false);
```

### `RunStateStore`

Persistent storage for agent state using PGLite by default (stored in platform-appropriate app directory):

```typescript
import { RunStateStore } from '@timestep-ai/timestep';
import { Agent, Session } from '@openai/agents';

const agent = new Agent({ model: 'gpt-4' });
const session = new Session();

// Default: Uses PGLite (stored in ~/.config/timestep/pglite/ on Linux)
const stateStore = new RunStateStore({ 
  agent, 
  sessionId: await session.getSessionId() 
});

// Or use PostgreSQL explicitly
const stateStore = new RunStateStore({
  agent,
  sessionId: await session.getSessionId(),
  connectionString: 'postgresql://user:pass@host/db'
});

// Save state
await stateStore.save(state);

// Load state
const loadedState = await stateStore.load();

// Clear state
await stateStore.clear();
```

**Storage Locations:**
- **Linux**: `~/.config/timestep/pglite/`
- **macOS**: `~/Library/Application Support/timestep/pglite/`
- **Windows**: `%APPDATA%/timestep/pglite/`

### `consumeResult()`

Utility for handling both streaming and non-streaming results:

```typescript
import { runAgent, consumeResult } from '@timestep-ai/timestep';

const result = await runAgent(agent, inputItems, session, true);
const consumed = await consumeResult(result); // Ensures all stream events are consumed

// Now safe to access result properties
console.log(consumed.finalOutput);
```

### `InterruptionException`

Exception raised when agent execution is interrupted for approval:

```typescript
import { runAgent, InterruptionException } from '@timestep-ai/timestep';

try {
  const result = await runAgent(agent, inputItems, session, false);
} catch (e) {
  if (e instanceof InterruptionException) {
    // Handle interruption
  }
}
```

## Model Providers

### MultiModelProvider

Automatically routes model requests to the appropriate provider based on model name prefixes:

```typescript
import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';

// Create a custom mapping
const providerMap = new MultiModelProviderMap();

// Add Ollama provider
if (Deno.env.get('OLLAMA_API_KEY')) {
  providerMap.addProvider(
    'ollama',
    new OllamaModelProvider({
      apiKey: Deno.env.get('OLLAMA_API_KEY'),
    })
  );
}

// Use in MultiModelProvider
const modelProvider = new MultiModelProvider({
  provider_map: providerMap,
  openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
});
```

**Features:**
- Automatic provider selection based on model name prefix
- Default fallback to OpenAI for unprefixed models
- Support for custom provider mappings

**Options:**
- `provider_map` (MultiModelProviderMap, optional): Custom provider mapping
- `openai_api_key` (string, optional): OpenAI API key
- `openai_base_url` (string, optional): OpenAI base URL
- `openai_client` (Any, optional): Custom OpenAI client
- `openai_organization` (string, optional): OpenAI organization
- `openai_project` (string, optional): OpenAI project
- `openai_use_responses` (boolean, optional): Use OpenAI responses API

### OllamaModelProvider

Provides access to Ollama models (local or cloud):

```typescript
import { OllamaModelProvider } from '@timestep-ai/timestep';

// Create an Ollama provider for local Ollama instance
const ollamaProvider = new OllamaModelProvider(); // Defaults to localhost:11434

// For Ollama Cloud, use the API key
const cloudProvider = new OllamaModelProvider({
  apiKey: 'your-ollama-cloud-key',
});
```

**Options:**
- `apiKey` (string, optional): API key for Ollama Cloud
- `baseURL` (string, optional): Base URL for Ollama instance (defaults to `http://localhost:11434` for local, `https://ollama.com` for cloud)
- `ollamaClient` (Any, optional): Custom Ollama client instance

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
- `addProvider(prefix, provider)`: Add a prefix-to-provider mapping
- `removeProvider(prefix)`: Remove a mapping
- `getProvider(prefix)`: Get provider for a prefix
- `hasPrefix(prefix)`: Check if prefix exists
- `getMapping()`: Get all mappings
- `setMapping(mapping)`: Replace all mappings

## Durable Execution with PGLite

Timestep uses **PGLite** (PostgreSQL in WebAssembly) as the default storage backend for durable execution:

- **PGLite by Default**: No setup required - works out of the box
- **Cross-Platform App Directory**: State stored in platform-appropriate directories (shared with Python):
  - Linux: `~/.config/timestep/pglite/`
  - macOS: `~/Library/Application Support/timestep/pglite/`
  - Windows: `%APPDATA%/timestep/pglite/`
- **PostgreSQL Option**: Use full PostgreSQL by setting `TIMESTEP_DB_URL` environment variable
- **State Persistence**: Agent run states are stored in the database
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
- ✅ **Cross-Language State Persistence**: Seamless state transfer with Python
- ✅ **Multi-Model Support**: Seamlessly switch between OpenAI and Ollama models
- ✅ **Automatic Routing**: Model names with prefixes automatically route to the correct provider
- ✅ **Customizable**: Add your own providers using `MultiModelProviderMap`
- ✅ **OpenAI Compatible**: Works with the OpenAI Agents SDK
- ✅ **Ollama Integration**: Full support for both local Ollama instances and Ollama Cloud
- ✅ **TypeScript**: Fully typed with TypeScript support
- ✅ **RISC-Style API**: Clean, simple abstractions over the OpenAI Agents SDK

## Requirements

- Node.js >=20
- @openai/agents-core ^0.2.1
- @openai/agents-openai ^0.2.1
- ollama ^0.6.2

## Examples

See the [tests](./tests/) directory for complete examples including:
- Basic agent execution
- Durable execution with state persistence
- Cross-language state transfer (TypeScript → Python)
- Same-language state persistence (TypeScript → TypeScript)

## Future Plans

We're actively developing additional features:

- **Enhanced DBOS Integration**: Full integration with DBOS for distributed durable execution
- **Additional Abstractions**: More RISC-style components extracted from complex SDK patterns
- **CLI Tool**: Command-line interface with tracing support for debugging and monitoring

## License

MIT
