# OllamaModelProvider

The `OllamaModelProvider` class provides access to Ollama models, supporting both local Ollama instances and [Ollama Cloud](https://ollama.com/cloud). It implements the `ModelProvider` interface from the OpenAI Agents SDK.

## Overview

`OllamaModelProvider` handles:

- **Local Ollama**: Connects to local Ollama instances (default: `http://localhost:11434`)
- **[Ollama Cloud](https://ollama.com/cloud)**: Connects to Ollama Cloud using API key authentication
- **Automatic Detection**: Automatically detects cloud vs local based on model name suffix (`-cloud`) or API key presence
- **Lazy Initialization**: Only creates the Ollama client when actually needed

## Constructor

=== "Python"

    ```python
    OllamaModelProvider(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        ollama_client: Optional[Any] = None,
    )
    ```

=== "TypeScript"

    ```typescript
    new OllamaModelProvider(options?: {
      apiKey?: string;
      baseURL?: string;
      ollamaClient?: any;
    })
    ```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` / `apiKey` | `string \| undefined` | [Ollama Cloud](https://ollama.com/cloud) API key. If provided, uses Ollama Cloud. |
| `base_url` / `baseURL` | `string \| undefined` | Base URL for Ollama instance. Defaults to `http://localhost:11434` for local or `https://ollama.com` for cloud. |
| `ollama_client` / `ollamaClient` | `Any \| undefined` | Custom Ollama client instance. If provided, `api_key` and `base_url` are ignored. |

### Notes

- If `api_key` is provided, the provider will use [Ollama Cloud](https://ollama.com/cloud)
- If `api_key` is not provided, the provider defaults to local Ollama
- Models ending with `-cloud` suffix automatically use [Ollama Cloud](https://ollama.com/cloud) URL
- The Ollama client is created lazily (only when `get_model()` is called)

## Methods

### `get_model()`

Returns an `OllamaModel` instance for the specified model name. The client is initialized on first call.

=== "Python"

    ```python
    def get_model(self, model_name: str) -> Model
    ```

=== "TypeScript"

    ```typescript
    async getModel(modelName: string): Promise<Model>
    ```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` / `modelName` | `string` | The name of the Ollama model (e.g., `llama3`, `mistral`). |

#### Returns

| Type | Description |
|------|-------------|
| `Model` | An `OllamaModel` instance that converts Ollama responses to OpenAI format. |

#### Example

=== "Python"

    ```python
    from timestep import OllamaModelProvider

    # Local Ollama
    local_provider = OllamaModelProvider()
    model = local_provider.get_model("llama3")

    # Ollama Cloud
    cloud_provider = OllamaModelProvider(api_key="your-api-key")
    cloud_model = cloud_provider.get_model("llama3")
    ```

=== "TypeScript"

    ```typescript
    import { OllamaModelProvider } from '@timestep-ai/timestep';

    // Local Ollama
    const localProvider = new OllamaModelProvider();
    const model = await localProvider.getModel('llama3');

    // Ollama Cloud
    const cloudProvider = new OllamaModelProvider({ apiKey: 'your-api-key' });
    const cloudModel = await cloudProvider.getModel('llama3');
    ```

## Examples

### Local Ollama Instance

=== "Python"

    ```python
    from timestep import OllamaModelProvider
    from agents import Agent, Runner, RunConfig

    # Defaults to localhost:11434
    ollama_provider = OllamaModelProvider()

    agent = Agent(model="llama3")
    run_config = RunConfig(model_provider=ollama_provider)
    result = Runner.run_streamed(agent, agent_input, run_config=run_config)
    ```

=== "TypeScript"

    ```typescript
    import { OllamaModelProvider } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    // Defaults to localhost:11434
    const ollamaProvider = new OllamaModelProvider();

    const agent = new Agent({ model: 'llama3' });
    const runner = new Runner({ modelProvider: ollamaProvider });
    const result = await runner.run(agent, agentInput, { stream: true });
    ```

### [Ollama Cloud](https://ollama.com/cloud)

=== "Python"

    ```python
    from timestep import OllamaModelProvider
    from agents import Agent, Runner, RunConfig
    import os

    # Use Ollama Cloud with API key
    cloud_provider = OllamaModelProvider(
        api_key=os.environ.get("OLLAMA_API_KEY")
    )

    agent = Agent(model="llama3")
    run_config = RunConfig(model_provider=cloud_provider)
    result = Runner.run_streamed(agent, agent_input, run_config=run_config)
    ```

=== "TypeScript"

    ```typescript
    import { OllamaModelProvider } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    // Use Ollama Cloud with API key
    const cloudProvider = new OllamaModelProvider({
      apiKey: Deno.env.get('OLLAMA_API_KEY'),
    });

    const agent = new Agent({ model: 'llama3' });
    const runner = new Runner({ modelProvider: cloudProvider });
    const result = await runner.run(agent, agentInput, { stream: true });
    ```

### Custom Base URL

=== "Python"

    ```python
    from timestep import OllamaModelProvider

    # Connect to remote Ollama instance
    remote_provider = OllamaModelProvider(
        base_url="http://ollama-server:11434"
    )

    model = remote_provider.get_model("llama3")
    ```

=== "TypeScript"

    ```typescript
    import { OllamaModelProvider } from '@timestep-ai/timestep';

    // Connect to remote Ollama instance
    const remoteProvider = new OllamaModelProvider({
      baseURL: 'http://ollama-server:11434',
    });

    const model = await remoteProvider.getModel('llama3');
    ```

### Automatic Cloud Detection

Models ending with `-cloud` automatically use [Ollama Cloud](https://ollama.com/cloud):

=== "Python"

    ```python
    from timestep import OllamaModelProvider

    # Provider will automatically use cloud URL for models ending with -cloud
    provider = OllamaModelProvider(api_key="your-api-key")
    
    # This will use Ollama Cloud
    model = provider.get_model("llama3-cloud")
    ```

=== "TypeScript"

    ```typescript
    import { OllamaModelProvider } from '@timestep-ai/timestep';

    // Provider will automatically use cloud URL for models ending with -cloud
    const provider = new OllamaModelProvider({ apiKey: 'your-api-key' });
    
    // This will use Ollama Cloud
    const model = await provider.getModel('llama3-cloud');
    ```

### Custom Client

=== "Python"

    ```python
    from timestep import OllamaModelProvider
    from ollama import AsyncClient

    # Use custom Ollama client
    custom_client = AsyncClient(host="http://custom-host:11434")
    provider = OllamaModelProvider(ollama_client=custom_client)

    model = provider.get_model("llama3")
    ```

=== "TypeScript"

    ```typescript
    import { OllamaModelProvider } from '@timestep-ai/timestep';
    import { Ollama } from 'ollama';

    // Use custom Ollama client
    const customClient = new Ollama({ host: 'http://custom-host:11434' });
    const provider = new OllamaModelProvider({ ollamaClient: customClient });

    const model = await provider.getModel('llama3');
    ```

## Error Handling

### Missing Ollama Package

If the `ollama` package is not installed, an error will be raised when `get_model()` is called:

=== "Python"

    ```python
    # This will raise ImportError if ollama package is not installed
    provider = OllamaModelProvider()
    model = provider.get_model("llama3")  # Raises ImportError
    ```

=== "TypeScript"

    ```typescript
    // This will throw an error if ollama package is not installed
    const provider = new OllamaModelProvider();
    const model = await provider.getModel('llama3'); // Throws error
    ```

### Invalid Configuration

If both `api_key` and `ollama_client` are provided, an error is raised:

=== "Python"

    ```python
    # This will raise ValueError
    provider = OllamaModelProvider(
        api_key="key",
        ollama_client=client
    )
    ```

=== "TypeScript"

    ```typescript
    // This will throw Error
    const provider = new OllamaModelProvider({
      apiKey: 'key',
      ollamaClient: client,
    });
    ```

## Lazy Initialization

The Ollama client is only created when `get_model()` is first called. This means:

- You can create the provider even if Ollama isn't running
- Errors only occur when actually trying to use a model
- Useful for optional Ollama support in applications

=== "Python"

    ```python
    # This won't fail even if Ollama isn't running
    provider = OllamaModelProvider()

    # Error only occurs here if Ollama isn't available
    try:
        model = provider.get_model("llama3")
    except Exception as e:
        print(f"Ollama not available: {e}")
    ```

=== "TypeScript"

    ```typescript
    // This won't fail even if Ollama isn't running
    const provider = new OllamaModelProvider();

    // Error only occurs here if Ollama isn't available
    try {
      const model = await provider.getModel('llama3');
    } catch (e) {
      console.error('Ollama not available:', e);
    }
    ```

## See Also

- [OllamaModel](ollama-model.md) - The model class that handles response conversion
- [MultiModelProvider](multi-model-provider.md) - For using Ollama alongside OpenAI
- [Architecture](../architecture.md) - For details on Ollama integration

