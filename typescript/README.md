# @timestep-ai/timestep

Multi-model provider implementations for OpenAI Agents, supporting both OpenAI and Ollama models. Works with both local Ollama instances and Ollama Cloud.

## Installation

```bash
npm install @timestep-ai/timestep
# or
pnpm add @timestep-ai/timestep
# or
yarn add @timestep-ai/timestep
```

## Quick Start

### Using MultiModelProvider (Recommended)

The `MultiModelProvider` automatically routes requests to the appropriate provider based on model name prefixes:

```typescript
import { MultiModelProvider } from '@timestep-ai/timestep';
import { Agent } from '@openai/agents';

// Create a provider that supports both OpenAI and Ollama
const provider = new MultiModelProvider({
  openai_api_key: 'your-openai-key' // Optional, uses default if not provided
});

// Create an agent that can use any model
const agent = new Agent({
  model: 'gpt-4', // Uses OpenAI
  provider
});

// Or use Ollama models
const ollamaAgent = new Agent({
  model: 'ollama/llama3', // Uses Ollama
  provider
});
```

### Using OllamaModelProvider Directly

```typescript
import { OllamaModelProvider } from '@timestep-ai/timestep';
import { Agent } from '@openai/agents';

// Create an Ollama provider for local Ollama instance
const ollamaProvider = new OllamaModelProvider({
  baseURL: 'http://localhost:11434' // Optional, defaults to localhost
});

// Create an agent using Ollama
const agent = new Agent({
  model: 'llama3',
  provider: ollamaProvider
});

// For Ollama Cloud, use the API key
const cloudProvider = new OllamaModelProvider({
  apiKey: 'your-ollama-cloud-key',
  baseURL: 'https://ollama.com' // Optional, auto-detected for models ending with "-cloud"
});

// Or provide a custom Ollama client
import { Ollama } from 'ollama';
const customClient = new Ollama({ host: 'http://custom-host:11434' });
const customProvider = new OllamaModelProvider({ ollamaClient: customClient });
```

### Using OllamaModel Directly

```typescript
import { OllamaModel } from '@timestep-ai/timestep';
import { Ollama } from 'ollama';
import { Agent } from '@openai/agents';

// Create an Ollama client
const client = new Ollama({ host: 'http://localhost:11434' });

// Create a model instance directly
const model = new OllamaModel('llama3', client);

// Use with agents
const agent = new Agent({ model });
```

### Custom Provider Mapping

```typescript
import { 
  MultiModelProvider, 
  MultiModelProviderMap, 
  OllamaModelProvider 
} from '@timestep-ai/timestep';
import { Agent } from '@openai/agents';

// Create a custom mapping
const providerMap = new MultiModelProviderMap();
providerMap.addProvider('custom', yourCustomProvider);

// Use the custom mapping
const provider = new MultiModelProvider({
  provider_map: providerMap,
  openai_api_key: 'your-key'
});

const agent = new Agent({
  model: 'custom/my-model',
  provider
});
```

## Components

### MultiModelProvider

Automatically routes model requests to the appropriate provider based on model name prefixes. Supports both OpenAI and Ollama models out of the box.

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

Provides access to Ollama models (local or cloud).

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

Direct model implementation that converts Ollama responses to OpenAI-compatible format.

**Features:**
- Converts Ollama API responses to OpenAI format
- Supports streaming responses
- Handles tool calls and function calling
- Compatible with OpenAI Agents SDK

### MultiModelProviderMap

Manages custom mappings of model name prefixes to providers.

**Methods:**
- `addProvider(prefix, provider)`: Add a prefix-to-provider mapping
- `removeProvider(prefix)`: Remove a mapping
- `getProvider(prefix)`: Get provider for a prefix
- `hasPrefix(prefix)`: Check if prefix exists
- `getMapping()`: Get all mappings
- `setMapping(mapping)`: Replace all mappings

## Features

- **Multi-Model Support**: Seamlessly switch between OpenAI and Ollama models
- **Automatic Routing**: Model names with prefixes (e.g., `ollama/llama3`) automatically route to the correct provider
- **Customizable**: Add your own providers using `MultiModelProviderMap`
- **OpenAI Compatible**: Works with the OpenAI Agents SDK
- **Ollama Integration**: Full support for both local Ollama instances and Ollama Cloud
- **TypeScript**: Fully typed with TypeScript support

## Model Naming

- Models without a prefix (e.g., `gpt-4`) default to OpenAI
- Models with `openai/` prefix (e.g., `openai/gpt-4`) use OpenAI
- Models with `ollama/` prefix (e.g., `ollama/llama3`) use Ollama

## Requirements

- Node.js >=20
- @openai/agents-core ^0.2.1
- @openai/agents-openai ^0.2.1
- ollama ^0.6.2

## Future Plans

We're actively developing additional features for the `@timestep-ai/timestep` library:

- **Additional Abstractions**: Gradually abstracting out other logic from [Timestep AI](https://github.com/Timestep-AI/timestep-ai) into reusable library components
- **CLI Tool**: A proper command-line interface with tracing support for debugging and monitoring agent interactions

## License

MIT
