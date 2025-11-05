# Timestep

Multi-model provider implementations for OpenAI Agents, supporting both OpenAI and Ollama models. Works with both local Ollama instances and Ollama Cloud.

This repository contains equivalent implementations in both Python and TypeScript.

## Packages

- **[Python Package (`timestep`)](./python/)** - [PyPI](https://pypi.org/project/timestep/)
- **[TypeScript Package (`@timestep-ai/timestep`)](./typescript/)** - [npm](https://www.npmjs.com/package/@timestep-ai/timestep)

## Quick Start

### Python

```bash
pip install timestep
```

```python
from timestep import MultiModelProvider
from agents import Agent

provider = MultiModelProvider(openai_api_key="your-key")
agent = Agent(model="gpt-4", provider=provider)
# Or use Ollama: Agent(model="ollama/llama3", provider=provider)
```

### TypeScript

```bash
npm install @timestep-ai/timestep
# or
pnpm add @timestep-ai/timestep
```

```typescript
import { MultiModelProvider } from '@timestep-ai/timestep';
import { Agent } from '@openai/agents';

const provider = new MultiModelProvider({ openai_api_key: 'your-key' });
const agent = new Agent({ model: 'gpt-4', provider });
// Or use Ollama: new Agent({ model: 'ollama/llama3', provider })
```

## Components

Both implementations provide:

- **MultiModelProvider**: Automatically routes requests to OpenAI or Ollama based on model name prefixes
- **OllamaModelProvider**: Provides access to Ollama models (local or cloud)
- **OllamaModel**: Direct model implementation that converts Ollama responses to OpenAI-compatible format
- **MultiModelProviderMap**: Manages custom mappings of model name prefixes to providers

## Features

- **Multi-Model Support**: Seamlessly switch between OpenAI and Ollama models
- **Automatic Routing**: Model names with prefixes (e.g., `ollama/llama3`) automatically route to the correct provider
- **Customizable**: Add your own providers using `MultiModelProviderMap`
- **OpenAI Compatible**: Works with the OpenAI Agents SDK
- **Ollama Integration**: Full support for both local Ollama instances and Ollama Cloud
- **Cross-Language**: Equivalent implementations in Python and TypeScript

## Documentation

- [Python Documentation](./python/README.md)
- [TypeScript Documentation](./typescript/README.md)

## Future Plans

We're actively developing additional features for the `timestep` libraries:

- **Additional Abstractions**: Gradually abstracting out other logic from [Timestep AI](https://github.com/Timestep-AI/timestep-ai) into reusable library components
- **CLI Tool**: A proper command-line interface with tracing support for debugging and monitoring agent interactions

## License

MIT License - see [LICENSE](LICENSE) file for details.
