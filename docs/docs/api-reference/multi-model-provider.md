# MultiModelProvider

The `MultiModelProvider` class automatically routes model requests to the appropriate provider based on model name prefixes. It implements the `ModelProvider` interface from the OpenAI Agents SDK.

## Overview

`MultiModelProvider` maps model names to providers using a prefix-based routing system:

- Models without a prefix (e.g., `gpt-4`) default to OpenAI
- Models with `openai/` prefix (e.g., `openai/gpt-4`) use OpenAI
- Models with `ollama/` prefix (e.g., `ollama/gpt-oss:20b-cloud`) use Ollama
- Custom prefixes can be mapped using `MultiModelProviderMap`

## Constructor

=== "Python"

    ```python
    MultiModelProvider(
        provider_map: Optional[MultiModelProviderMap] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_client: Optional[Any] = None,
        openai_organization: Optional[str] = None,
        openai_project: Optional[str] = None,
        openai_use_responses: Optional[bool] = None,
    )
    ```

=== "TypeScript"

    ```typescript
    new MultiModelProvider(options?: {
      provider_map?: MultiModelProviderMap;
      openai_api_key?: string;
      openai_base_url?: string;
      openai_client?: any;
      openai_organization?: string;
      openai_project?: string;
      openai_use_responses?: boolean;
    })
    ```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider_map` | `MultiModelProviderMap \| undefined` | Custom provider mapping. If not provided, uses default mapping. |
| `openai_api_key` | `string \| undefined` | OpenAI API key. If not provided, uses default from environment. |
| `openai_base_url` | `string \| undefined` | Base URL for OpenAI API. If not provided, uses default. |
| `openai_client` | `Any \| undefined` | Custom OpenAI client instance. If provided, other OpenAI options are ignored. |
| `openai_organization` | `string \| undefined` | OpenAI organization ID. |
| `openai_project` | `string \| undefined` | OpenAI project ID. |
| `openai_use_responses` | `boolean \| undefined` | Whether to use OpenAI responses API. |

## Methods

### `get_model()`

Returns a `Model` instance based on the model name. The model name can have a prefix that determines which provider to use.

=== "Python"

    ```python
    def get_model(self, model_name: Optional[str]) -> Model
    ```

=== "TypeScript"

    ```typescript
    async getModel(model_name: string | undefined): Promise<Model>
    ```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `string \| None` | The name of the model to get. Can include a prefix (e.g., `ollama/gpt-oss:20b-cloud`). |

#### Returns

| Type | Description |
|------|-------------|
| `Model` | A Model instance from the appropriate provider. |

#### Example

=== "Python"

    ```python
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
    import os

    # Create provider
    provider_map = MultiModelProviderMap()
    provider_map.add_provider(
        "ollama",
        OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
    )

    model_provider = MultiModelProvider(
        provider_map=provider_map,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    # Get OpenAI model
    openai_model = model_provider.get_model("gpt-4.1")

    # Get Ollama model
    ollama_model = model_provider.get_model("ollama/gpt-oss:20b-cloud")
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';

    // Create provider
    const providerMap = new MultiModelProviderMap();
    providerMap.addProvider(
      'ollama',
      new OllamaModelProvider({ apiKey: Deno.env.get('OLLAMA_API_KEY') })
    );

    const modelProvider = new MultiModelProvider({
      provider_map: providerMap,
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });

    // Get OpenAI model
    const openaiModel = await modelProvider.getModel('gpt-4.1');

    // Get Ollama model
    const ollamaModel = await modelProvider.getModel('ollama/gpt-oss:20b-cloud');
    ```

## Routing Logic

The routing logic works as follows:

1. **Parse Model Name**: Extract prefix and actual model name from the full model name
2. **Check Custom Mapping**: If `provider_map` is provided, check for custom prefix mapping
3. **Fallback to Default**: If no custom mapping found, use default providers:
   - `openai/` or no prefix → OpenAI
   - `ollama/` → Ollama (creates default OllamaModelProvider if needed)

## Examples

### Basic Usage

=== "Python"

    ```python
    from timestep import MultiModelProvider
    from agents import Agent, Runner, RunConfig
    import os

    # Simple setup with defaults
    model_provider = MultiModelProvider(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    # Use OpenAI model
    agent = Agent(model="gpt-4.1")
    run_config = RunConfig(model_provider=model_provider)
    result = Runner.run_streamed(agent, agent_input, run_config=run_config)
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProvider } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    // Simple setup with defaults
    const modelProvider = new MultiModelProvider({
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });

    // Use OpenAI model
    const agent = new Agent({ model: 'gpt-4.1' });
    const runner = new Runner({ modelProvider });
    const result = await runner.run(agent, agentInput, { stream: true });
    ```

### With Custom Provider Mapping

=== "Python"

    ```python
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
    import os

    # Create custom mapping
    provider_map = MultiModelProviderMap()
    provider_map.add_provider(
        "ollama",
        OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
    )

    model_provider = MultiModelProvider(
        provider_map=provider_map,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    # Now you can use both providers
    openai_agent = Agent(model="gpt-4.1")
    ollama_agent = Agent(model="ollama/gpt-oss:20b-cloud")
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';

    // Create custom mapping
    const providerMap = new MultiModelProviderMap();
    providerMap.addProvider(
      'ollama',
      new OllamaModelProvider({ apiKey: Deno.env.get('OLLAMA_API_KEY') })
    );

    const modelProvider = new MultiModelProvider({
      provider_map: providerMap,
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });

    // Now you can use both providers
    const openaiAgent = new Agent({ model: 'gpt-4.1' });
    const ollamaAgent = new Agent({ model: 'ollama/gpt-oss:20b-cloud' });
    ```

### With Custom OpenAI Configuration

=== "Python"

    ```python
    from timestep import MultiModelProvider

    model_provider = MultiModelProvider(
        openai_api_key="sk-...",
        openai_base_url="https://api.openai.com/v1",
        openai_organization="org-...",
        openai_project="proj-...",
    )
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProvider } from '@timestep-ai/timestep';

    const modelProvider = new MultiModelProvider({
      openai_api_key: 'sk-...',
      openai_base_url: 'https://api.openai.com/v1',
      openai_organization: 'org-...',
      openai_project: 'proj-...',
    });
    ```

## Error Handling

If an unknown prefix is used and no custom mapping exists, `get_model()` will raise an error:

=== "Python"

    ```python
    # This will raise ValueError if "custom" prefix is not mapped
    model = model_provider.get_model("custom/model")
    ```

=== "TypeScript"

    ```typescript
    // This will throw Error if "custom" prefix is not mapped
    const model = await modelProvider.getModel('custom/model');
    ```

## See Also

- [MultiModelProviderMap](multi-model-provider-map.md) - For custom provider mappings
- [OllamaModelProvider](ollama-model-provider.md) - For Ollama-specific configuration
- [Architecture](../architecture.md) - For detailed routing explanation

