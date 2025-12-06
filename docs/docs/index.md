# Timestep Documentation

Welcome to the Timestep documentation! Timestep provides multi-model provider implementations for OpenAI Agents, supporting both OpenAI and Ollama models. This library works seamlessly with **local Ollama instances** and **Ollama Cloud**, giving you the flexibility to run models locally or in the cloud.

## Packages

- **[Python Package (`timestep`)](https://pypi.org/project/timestep/)** - Available on PyPI
- **[TypeScript Package (`@timestep-ai/timestep`)](https://www.npmjs.com/package/@timestep-ai/timestep)** - Available on npm

## Quick Navigation

### Getting Started
- [Installation and Quick Start](getting-started.md) - Get up and running with Timestep in minutes

### Core Concepts
- [Architecture](architecture.md) - Understand how Timestep works under the hood
- [Use Cases](use-cases.md) - Common patterns and real-world examples

### API Reference
- [MultiModelProvider](api-reference/multi-model-provider.md) - Automatic model routing
- [OllamaModelProvider](api-reference/ollama-model-provider.md) - Ollama integration
- [MultiModelProviderMap](api-reference/multi-model-provider-map.md) - Custom provider mappings
- [Tools](api-reference/tools.md) - Built-in tools like web search
- [Utilities](api-reference/utilities.md) - Helper functions and classes

## Features

- **Multi-Model Support**: Seamlessly switch between OpenAI and Ollama models
- **Automatic Routing**: Model names with prefixes (e.g., `ollama/llama3`) automatically route to the correct provider
- **Customizable**: Add your own providers using `MultiModelProviderMap`
- **OpenAI Compatible**: Works with the OpenAI Agents SDK
- **Ollama Integration**: Full support for both **local Ollama instances** and **Ollama Cloud** - switch seamlessly between local and cloud models
- **Cross-Language**: Equivalent implementations in Python and TypeScript

## Quick Example

=== "Python"

    ```python
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
    from agents import Agent, Runner, RunConfig
    import os

    # Setup provider with Ollama Cloud support
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

=== "TypeScript"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    // Setup provider with Ollama Cloud support
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

## Requirements

### Python
- Python >=3.11
- ollama >=0.6.0
- openai-agents >=0.4.2

### TypeScript
- Node.js >=20
- @openai/agents-core ^0.2.1
- @openai/agents-openai ^0.2.1
- ollama ^0.6.2

## License

MIT License - see [LICENSE](https://github.com/Timestep-AI/timestep/blob/main/LICENSE) file for details.

