# Getting Started

This guide will help you get up and running with Timestep in both Python and TypeScript.

## Installation

=== "Python"

    Install Timestep using pip:

    ```bash
    pip install timestep
    ```

    Or using uv:

    ```bash
    uv add timestep
    ```

=== "TypeScript"

    Install Timestep using npm:

    ```bash
    npm install @timestep-ai/timestep
    ```

    Or using pnpm:

    ```bash
    pnpm add @timestep-ai/timestep
    ```

    Or using yarn:

    ```bash
    yarn add @timestep-ai/timestep
    ```

## Prerequisites

### OpenAI API Key

You'll need an OpenAI API key to use OpenAI models. Set it as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Ollama Setup

Timestep supports both **local Ollama instances** and **Ollama Cloud**. You can use either or both!

#### Option 1: Ollama Cloud (Recommended for Production)

Ollama Cloud provides managed Ollama instances with no local setup required:

1. Get an API key from [Ollama Cloud](https://ollama.com/cloud)
2. Set it as an environment variable:

```bash
export OLLAMA_API_KEY="your-ollama-cloud-key"
```

That's it! No installation or local setup needed. Timestep will automatically use Ollama Cloud when the API key is set.

#### Option 2: Local Ollama

For local development or when you want to run models on your own infrastructure:

1. Install [Ollama](https://ollama.com/)
2. Start the Ollama service (usually runs on `http://localhost:11434`)
3. Pull the models you want to use: `ollama pull llama3`

Timestep will automatically detect and use your local Ollama instance if no API key is provided.

## Quick Start

### Using MultiModelProvider (Recommended)

The `MultiModelProvider` automatically routes requests to the appropriate provider based on model name prefixes. This is the recommended approach for most use cases.

=== "Python"

    ```python
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
    from agents import Agent, Runner, RunConfig
    import os

    # Create a provider map and add Ollama support (works with both local and cloud)
    model_provider_map = MultiModelProviderMap()

    # Add Ollama Cloud support (if API key is set)
    if os.environ.get("OLLAMA_API_KEY"):
        model_provider_map.add_provider(
            "ollama",
            OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
        )
    # Or use local Ollama by omitting the API key:
    # model_provider_map.add_provider("ollama", OllamaModelProvider())

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

=== "TypeScript"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    // Create a provider map and add Ollama support (works with both local and cloud)
    const modelProviderMap = new MultiModelProviderMap();

    // Add Ollama Cloud support (if API key is set)
    if (Deno.env.get('OLLAMA_API_KEY')) {
      modelProviderMap.addProvider(
        'ollama',
        new OllamaModelProvider({
          apiKey: Deno.env.get('OLLAMA_API_KEY'),
        })
      );
    }
    // Or use local Ollama by omitting the API key:
    // modelProviderMap.addProvider('ollama', new OllamaModelProvider());

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

### Using OllamaModelProvider Directly

If you only need Ollama models, you can use `OllamaModelProvider` directly. It supports both **Ollama Cloud** and **local Ollama instances**:

=== "Python"

    ```python
    from timestep import OllamaModelProvider
    from agents import Agent, Runner, RunConfig

    # Option 1: Use Ollama Cloud (recommended for production)
    cloud_provider = OllamaModelProvider(api_key="your-ollama-cloud-key")

    # Option 2: Use local Ollama instance (for development)
    local_provider = OllamaModelProvider()  # Defaults to localhost:11434

    # Create agent and run with Ollama Cloud
    agent = Agent(model="llama3")
    run_config = RunConfig(model_provider=cloud_provider)
    result = Runner.run_streamed(agent, agent_input, run_config=run_config)
    ```

=== "TypeScript"

    ```typescript
    import { OllamaModelProvider } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    // Option 1: Use Ollama Cloud (recommended for production)
    const cloudProvider = new OllamaModelProvider({
      apiKey: 'your-ollama-cloud-key',
    });

    // Option 2: Use local Ollama instance (for development)
    const localProvider = new OllamaModelProvider(); // Defaults to localhost:11434

    // Create agent and run with Ollama Cloud
    const agent = new Agent({ model: 'llama3' });
    const runner = new Runner({ modelProvider: cloudProvider });
    const result = await runner.run(agent, agentInput, { stream: true });
    ```

## Model Naming Conventions

Timestep uses model name prefixes to determine which provider to use:

- **No prefix** (e.g., `gpt-4`): Defaults to OpenAI
- **`openai/` prefix** (e.g., `openai/gpt-4`): Explicitly uses OpenAI
- **`ollama/` prefix** (e.g., `ollama/llama3`): Uses Ollama

### Examples

=== "Python"

    ```python
    # OpenAI models
    agent1 = Agent(model="gpt-4")           # Uses OpenAI
    agent2 = Agent(model="openai/gpt-4")    # Also uses OpenAI

    # Ollama models
    agent3 = Agent(model="ollama/llama3")    # Uses Ollama (local or cloud)
    agent4 = Agent(model="ollama/llama3-cloud")  # Uses Ollama Cloud
    ```

=== "TypeScript"

    ```typescript
    // OpenAI models
    const agent1 = new Agent({ model: 'gpt-4' });        // Uses OpenAI
    const agent2 = new Agent({ model: 'openai/gpt-4' }); // Also uses OpenAI

    // Ollama models
    const agent3 = new Agent({ model: 'ollama/llama3' });      // Uses Ollama (local or cloud)
    const agent4 = new Agent({ model: 'ollama/llama3-cloud' }); // Uses Ollama Cloud
    ```

## Environment Variables

Here's a summary of the environment variables you might need:

| Variable | Description | Required For |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | Your OpenAI API key | OpenAI models |
| `OLLAMA_API_KEY` | Your Ollama Cloud API key | **Ollama Cloud** (highly recommended - no local setup needed!) |
| `FIRECRAWL_API_KEY` | Firecrawl API key for web search tool | Web search functionality |

!!! tip "Ollama Cloud vs Local"
    - **Ollama Cloud**: Set `OLLAMA_API_KEY` - no installation or local setup required, perfect for production
    - **Local Ollama**: Omit `OLLAMA_API_KEY` - requires local Ollama installation, great for development and offline use

## Next Steps

- Learn about the [Architecture](architecture.md) to understand how Timestep works
- Explore [Use Cases](use-cases.md) for common patterns and examples
- Check out the [API Reference](api-reference/multi-model-provider.md) for detailed documentation

