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
from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
from agents import Agent, Runner, RunConfig
import os

# Setup provider
model_provider_map = MultiModelProviderMap()
if os.environ.get("OLLAMA_API_KEY"):
    model_provider_map.add_provider(
        "ollama",
        OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
    )

model_provider = MultiModelProvider(
    provider_map=model_provider_map,
    openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
)

# Create agent and run
agent = Agent(model="ollama/llama3")
run_config = RunConfig(model_provider=model_provider)
result = Runner.run_streamed(agent, agent_input, run_config=run_config)
```

### TypeScript

```bash
npm install @timestep-ai/timestep
# or
pnpm add @timestep-ai/timestep
```

```typescript
import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';
import { Agent, Runner } from '@openai/agents';

// Setup provider
const modelProviderMap = new MultiModelProviderMap();
if (Deno.env.get('OLLAMA_API_KEY')) {
  modelProviderMap.addProvider(
    'ollama',
    new OllamaModelProvider({ apiKey: Deno.env.get('OLLAMA_API_KEY') })
  );
}

const modelProvider = new MultiModelProvider({
  provider_map: modelProviderMap,
  openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
});

// Create agent and run
const agent = new Agent({ model: 'ollama/llama3' });
const runner = new Runner({ modelProvider });
const result = await runner.run(agent, agentInput, { stream: true });
```

## Development

### Running Tests

Run the behavior test harness to verify both Python and TypeScript implementations:

```bash
cd rust && OPENAI_API_KEY="your-api-key-here" cargo run -- test --format json
```

This command runs all behavior tests against both Python and TypeScript implementations and outputs the results in JSON format. Replace `your-api-key-here` with your actual OpenAI API key.

### Bumping Versions

To bump the patch version for both Python and TypeScript packages:

```bash
cd python && uv version --bump patch && cd ../typescript && npm version patch --no-git-tag-version --no-commit-hooks
```

This increments the patch version (e.g., `2026.0.3` → `2026.0.4`) for both packages in a single command.

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
