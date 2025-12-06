# MultiModelProviderMap

The `MultiModelProviderMap` class manages custom mappings of model name prefixes to `ModelProvider` instances. It allows you to customize how `MultiModelProvider` routes model requests.

## Overview

`MultiModelProviderMap` provides a flexible way to:

- Map custom prefixes to specific provider instances
- Override default provider behavior
- Support multiple instances of the same provider type
- Dynamically add or remove provider mappings

## Constructor

=== "Python"

    ```python
    MultiModelProviderMap()
    ```

=== "TypeScript"

    ```typescript
    new MultiModelProviderMap()
    ```

Creates an empty provider map. Use the methods below to add provider mappings.

## Methods

### `add_provider()` / `addProvider()`

Adds a new prefix-to-provider mapping.

=== "Python"

    ```python
    def add_provider(self, prefix: str, provider: ModelProvider) -> None
    ```

=== "TypeScript"

    ```typescript
    addProvider(prefix: string, provider: ModelProvider): void
    ```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prefix` | `string` | The prefix to map (e.g., `"ollama"`, `"custom"`). |
| `provider` | `ModelProvider` | The provider instance to use for this prefix. |

#### Example

=== "Python"

    ```python
    from timestep import MultiModelProviderMap, OllamaModelProvider
    import os

    provider_map = MultiModelProviderMap()
    provider_map.add_provider(
        "ollama",
        OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
    )
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';

    const providerMap = new MultiModelProviderMap();
    providerMap.addProvider(
      'ollama',
      new OllamaModelProvider({ apiKey: Deno.env.get('OLLAMA_API_KEY') })
    );
    ```

### `remove_provider()` / `removeProvider()`

Removes a prefix-to-provider mapping.

=== "Python"

    ```python
    def remove_provider(self, prefix: str) -> None
    ```

=== "TypeScript"

    ```typescript
    removeProvider(prefix: string): void
    ```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prefix` | `string` | The prefix to remove from the mapping. |

#### Example

=== "Python"

    ```python
    provider_map = MultiModelProviderMap()
    provider_map.add_provider("ollama", ollama_provider)
    
    # Later, remove it
    provider_map.remove_provider("ollama")
    ```

=== "TypeScript"

    ```typescript
    const providerMap = new MultiModelProviderMap();
    providerMap.addProvider('ollama', ollamaProvider);
    
    // Later, remove it
    providerMap.removeProvider('ollama');
    ```

### `get_provider()` / `getProvider()`

Gets the provider for a given prefix.

=== "Python"

    ```python
    def get_provider(self, prefix: str) -> Optional[ModelProvider]
    ```

=== "TypeScript"

    ```typescript
    getProvider(prefix: string): ModelProvider | undefined
    ```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prefix` | `string` | The prefix to look up. |

#### Returns

| Type | Description |
|------|-------------|
| `ModelProvider \| None` / `ModelProvider \| undefined` | The provider for the prefix, or `None`/`undefined` if not found. |

#### Example

=== "Python"

    ```python
    provider_map = MultiModelProviderMap()
    provider_map.add_provider("ollama", ollama_provider)
    
    provider = provider_map.get_provider("ollama")
    if provider:
        print("Found provider")
    ```

=== "TypeScript"

    ```typescript
    const providerMap = new MultiModelProviderMap();
    providerMap.addProvider('ollama', ollamaProvider);
    
    const provider = providerMap.getProvider('ollama');
    if (provider) {
      console.log('Found provider');
    }
    ```

### `has_prefix()` / `hasPrefix()`

Checks if a prefix exists in the mapping.

=== "Python"

    ```python
    def has_prefix(self, prefix: str) -> bool
    ```

=== "TypeScript"

    ```typescript
    hasPrefix(prefix: string): boolean
    ```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prefix` | `string` | The prefix to check. |

#### Returns

| Type | Description |
|------|-------------|
| `boolean` | `True` if the prefix exists, `False` otherwise. |

#### Example

=== "Python"

    ```python
    provider_map = MultiModelProviderMap()
    provider_map.add_provider("ollama", ollama_provider)
    
    if provider_map.has_prefix("ollama"):
        print("Ollama provider is configured")
    ```

=== "TypeScript"

    ```typescript
    const providerMap = new MultiModelProviderMap();
    providerMap.addProvider('ollama', ollamaProvider);
    
    if (providerMap.hasPrefix('ollama')) {
      console.log('Ollama provider is configured');
    }
    ```

### `get_mapping()` / `getMapping()`

Gets a copy of all prefix-to-provider mappings.

=== "Python"

    ```python
    def get_mapping(self) -> Dict[str, ModelProvider]
    ```

=== "TypeScript"

    ```typescript
    getMapping(): Map<string, ModelProvider>
    ```

#### Returns

| Type | Description |
|------|-------------|
| `Dict[str, ModelProvider]` / `Map<string, ModelProvider>` | A copy of all mappings. Modifying this copy does not affect the original map. |

#### Example

=== "Python"

    ```python
    provider_map = MultiModelProviderMap()
    provider_map.add_provider("ollama", ollama_provider)
    provider_map.add_provider("custom", custom_provider)
    
    all_mappings = provider_map.get_mapping()
    for prefix, provider in all_mappings.items():
        print(f"{prefix}: {provider}")
    ```

=== "TypeScript"

    ```typescript
    const providerMap = new MultiModelProviderMap();
    providerMap.addProvider('ollama', ollamaProvider);
    providerMap.addProvider('custom', customProvider);
    
    const allMappings = providerMap.getMapping();
    for (const [prefix, provider] of allMappings) {
      console.log(`${prefix}: ${provider}`);
    }
    ```

### `set_mapping()` / `setMapping()`

Replaces all mappings with a new set of mappings.

=== "Python"

    ```python
    def set_mapping(self, mapping: Dict[str, ModelProvider]) -> None
    ```

=== "TypeScript"

    ```typescript
    setMapping(mapping: Map<string, ModelProvider>): void
    ```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `mapping` | `Dict[str, ModelProvider]` / `Map<string, ModelProvider>` | The new mapping to use. Replaces all existing mappings. |

#### Example

=== "Python"

    ```python
    from timestep import MultiModelProviderMap, OllamaModelProvider

    # Create new mappings
    new_mappings = {
        "ollama": OllamaModelProvider(),
        "custom": custom_provider,
    }
    
    provider_map = MultiModelProviderMap()
    provider_map.set_mapping(new_mappings)
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';

    // Create new mappings
    const newMappings = new Map([
      ['ollama', new OllamaModelProvider()],
      ['custom', customProvider],
    ]);
    
    const providerMap = new MultiModelProviderMap();
    providerMap.setMapping(newMappings);
    ```

## Examples

### Basic Usage

=== "Python"

    ```python
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
    import os

    # Create and configure provider map
    provider_map = MultiModelProviderMap()
    provider_map.add_provider(
        "ollama",
        OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
    )

    # Use in MultiModelProvider
    model_provider = MultiModelProvider(
        provider_map=provider_map,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';

    // Create and configure provider map
    const providerMap = new MultiModelProviderMap();
    providerMap.addProvider(
      'ollama',
      new OllamaModelProvider({ apiKey: Deno.env.get('OLLAMA_API_KEY') })
    );

    // Use in MultiModelProvider
    const modelProvider = new MultiModelProvider({
      provider_map: providerMap,
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });
    ```

### Multiple Provider Instances

=== "Python"

    ```python
    from timestep import MultiModelProviderMap, OllamaModelProvider

    provider_map = MultiModelProviderMap()
    
    # Local Ollama
    provider_map.add_provider(
        "local",
        OllamaModelProvider(base_url="http://localhost:11434")
    )
    
    # Remote Ollama
    provider_map.add_provider(
        "remote",
        OllamaModelProvider(base_url="http://ollama-server:11434")
    )
    
    # Ollama Cloud
    provider_map.add_provider(
        "cloud",
        OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
    )
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';

    const providerMap = new MultiModelProviderMap();
    
    // Local Ollama
    providerMap.addProvider(
      'local',
      new OllamaModelProvider({ baseURL: 'http://localhost:11434' })
    );
    
    // Remote Ollama
    providerMap.addProvider(
      'remote',
      new OllamaModelProvider({ baseURL: 'http://ollama-server:11434' })
    );
    
    // Ollama Cloud
    providerMap.addProvider(
      'cloud',
      new OllamaModelProvider({ apiKey: Deno.env.get('OLLAMA_API_KEY') })
    );
    ```

### Dynamic Provider Management

=== "Python"

    ```python
    from timestep import MultiModelProviderMap, OllamaModelProvider

    provider_map = MultiModelProviderMap()

    # Add provider conditionally
    if os.environ.get("OLLAMA_API_KEY"):
        provider_map.add_provider(
            "ollama",
            OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
        )

    # Check if provider exists before using
    if provider_map.has_prefix("ollama"):
        # Use ollama models
        pass
    else:
        # Fall back to OpenAI only
        pass
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';

    const providerMap = new MultiModelProviderMap();

    // Add provider conditionally
    if (Deno.env.get('OLLAMA_API_KEY')) {
      providerMap.addProvider(
        'ollama',
        new OllamaModelProvider({ apiKey: Deno.env.get('OLLAMA_API_KEY') })
      );
    }

    // Check if provider exists before using
    if (providerMap.hasPrefix('ollama')) {
      // Use ollama models
    } else {
      // Fall back to OpenAI only
    }
    ```

## See Also

- [MultiModelProvider](multi-model-provider.md) - Uses MultiModelProviderMap for routing
- [OllamaModelProvider](ollama-model-provider.md) - Common provider to add to the map
- [Architecture](../architecture.md) - For details on provider mapping
