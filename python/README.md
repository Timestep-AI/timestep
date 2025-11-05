# Timestep

Multi-model provider implementations for OpenAI Agents, supporting both OpenAI and Ollama models. Works with both local Ollama instances and Ollama Cloud.

## Installation

```bash
pip install timestep
```

## Quick Start

### Using MultiModelProvider (Recommended)

The `MultiModelProvider` automatically routes requests to the appropriate provider based on model name prefixes:

```python
from timestep import MultiModelProvider, MultiModelProviderMap
from agents import Agent, Runner, RunConfig
import os

# Create a provider map and add Ollama support
model_provider_map = MultiModelProviderMap()

if os.environ.get("OLLAMA_API_KEY"):
    from timestep import OllamaModelProvider
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

### Using OllamaModelProvider Directly

```python
from timestep import OllamaModelProvider
from agents import Agent, Runner, RunConfig

# Create an Ollama provider for local Ollama instance
ollama_provider = OllamaModelProvider()  # Defaults to localhost:11434

# For Ollama Cloud, use the API key
cloud_provider = OllamaModelProvider(api_key="your-ollama-cloud-key")

# Create agent and run
agent = Agent(model="llama3")
run_config = RunConfig(model_provider=ollama_provider)
result = Runner.run_streamed(agent, agent_input, run_config=run_config)
```

### Custom Provider Mapping

```python
from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
from agents import Agent, Runner, RunConfig
import os

# Create a custom mapping
model_provider_map = MultiModelProviderMap()

# Add Ollama provider
if os.environ.get("OLLAMA_API_KEY"):
    model_provider_map.add_provider(
        "ollama",
        OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
    )

# Use the custom mapping
model_provider = MultiModelProvider(
    provider_map=model_provider_map,
    openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
)

agent = Agent(model="ollama/llama3")
run_config = RunConfig(model_provider=model_provider)
result = Runner.run_streamed(agent, agent_input, run_config=run_config)
```

## Components

### MultiModelProvider

Automatically routes model requests to the appropriate provider based on model name prefixes. Supports both OpenAI and Ollama models out of the box.

**Features:**
- Automatic provider selection based on model name prefix
- Default fallback to OpenAI for unprefixed models
- Support for custom provider mappings

### OllamaModelProvider

Provides access to Ollama models (local or cloud).

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

Direct model implementation that converts Ollama responses to OpenAI-compatible format.

**Features:**
- Converts Ollama API responses to OpenAI format
- Supports streaming responses
- Handles tool calls and function calling
- Compatible with OpenAI Agents SDK

### MultiModelProviderMap

Manages custom mappings of model name prefixes to providers.

**Methods:**
- `add_provider(prefix, provider)`: Add a prefix-to-provider mapping
- `remove_provider(prefix)`: Remove a mapping
- `get_provider(prefix)`: Get provider for a prefix
- `has_prefix(prefix)`: Check if prefix exists
- `get_mapping()`: Get all mappings
- `set_mapping(mapping)`: Replace all mappings

## Features

- **Multi-Model Support**: Seamlessly switch between OpenAI and Ollama models
- **Automatic Routing**: Model names with prefixes (e.g., `ollama/llama3`) automatically route to the correct provider
- **Customizable**: Add your own providers using `MultiModelProviderMap`
- **OpenAI Compatible**: Works with the OpenAI Agents SDK
- **Ollama Integration**: Full support for both local Ollama instances and Ollama Cloud

## Model Naming

- Models without a prefix (e.g., `gpt-4`) default to OpenAI
- Models with `openai/` prefix (e.g., `openai/gpt-4`) use OpenAI
- Models with `ollama/` prefix (e.g., `ollama/llama3`) use Ollama

## Requirements

- Python >=3.11
- ollama >=0.6.0
- openai-agents >=0.4.2

## Future Plans

We're actively developing additional features for the `timestep` library:

- **Additional Abstractions**: Gradually abstracting out other logic from [Timestep AI](https://github.com/Timestep-AI/timestep-ai) into reusable library components
- **CLI Tool**: A proper command-line interface with tracing support for debugging and monitoring agent interactions

## License

MIT
