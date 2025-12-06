# Architecture

This document explains how Timestep works under the hood, including model routing, provider mapping, and integration with OpenAI and Ollama.

## Overview

Timestep provides a unified interface for using multiple AI model providers (OpenAI and Ollama) with the OpenAI Agents SDK. The core design principle is to automatically route model requests to the appropriate provider based on model name prefixes, while maintaining full compatibility with the OpenAI Agents SDK.

!!! important "Ollama Cloud Support"
    Timestep provides **first-class support for Ollama Cloud**, allowing you to use managed Ollama instances without any local setup. Simply set the `OLLAMA_API_KEY` environment variable and Timestep will automatically use Ollama Cloud. This makes it easy to deploy applications without managing local infrastructure.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenAI Agents SDK                        │
│                  (Agent, Runner, etc.)                     │
└──────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              MultiModelProvider (Timestep)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Model Name Parser                             │   │
│  │  - Parses model name (e.g., "ollama/llama3")        │   │
│  │  - Extracts prefix and actual model name             │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                      │                                       │
│  ┌───────────────────▼───────────────────────────────────┐ │
│  │         Provider Router                               │ │
│  │  - Checks MultiModelProviderMap for custom mappings  │ │
│  │  - Falls back to default providers                    │ │
│  └───────┬───────────────────────┬───────────────────────┘ │
│          │                       │                          │
└──────────┼───────────────────────┼──────────────────────────┘
           │                       │
           ▼                       ▼
┌──────────────────┐    ┌──────────────────────┐
│  OpenAIProvider   │    │  OllamaModelProvider  │
│  (from SDK)       │    │  (Timestep)          │
└────────┬─────────┘    └──────────┬───────────┘
         │                         │
         ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│   OpenAI API      │    │    Ollama API         │
│                   │    │  (local or cloud)    │
└───────────────────┘    └──────────────────────┘
```

## Model Routing

### Prefix-Based Routing

Timestep uses a simple but powerful prefix-based routing system:

1. **Model Name Parsing**: When a model is requested, the name is parsed to extract:
   - **Prefix**: The part before the first `/` (e.g., `ollama` from `ollama/llama3`)
   - **Model Name**: The part after the `/` (e.g., `llama3` from `ollama/llama3`)

2. **Provider Lookup**: The system looks up the provider in this order:
   - First checks `MultiModelProviderMap` for custom mappings
   - Falls back to default providers:
     - `openai/` or no prefix → `OpenAIProvider`
     - `ollama/` → `OllamaModelProvider`

3. **Model Retrieval**: The appropriate provider's `get_model()` method is called with the actual model name.

### Default Routing Rules

| Model Name Format | Provider | Example |
|-------------------|----------|---------|
| `gpt-4` | OpenAI | `gpt-4`, `gpt-3.5-turbo` |
| `openai/gpt-4` | OpenAI | `openai/gpt-4` |
| `ollama/llama3` | Ollama | `ollama/llama3`, `ollama/mistral` |

## Provider Mapping

### MultiModelProviderMap

The `MultiModelProviderMap` class allows you to customize the routing behavior by mapping prefixes to specific provider instances. This enables:

- **Custom Providers**: Add your own model providers
- **Multiple Ollama Instances**: Route different prefixes to different Ollama endpoints
- **Provider Overrides**: Override default behavior for specific prefixes

### Example Custom Mapping

=== "Python"

    ```python
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider

    # Create custom mapping
    provider_map = MultiModelProviderMap()
    
    # Add custom Ollama instance
    provider_map.add_provider(
        "custom-ollama",
        OllamaModelProvider(base_url="http://custom-ollama:11434")
    )
    
    # Use in MultiModelProvider
    model_provider = MultiModelProvider(provider_map=provider_map)
    
    # Now "custom-ollama/llama3" will route to the custom instance
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';

    // Create custom mapping
    const providerMap = new MultiModelProviderMap();
    
    // Add custom Ollama instance
    providerMap.addProvider(
      'custom-ollama',
      new OllamaModelProvider({ baseURL: 'http://custom-ollama:11434' })
    );
    
    // Use in MultiModelProvider
    const modelProvider = new MultiModelProvider({ provider_map: providerMap });
    
    // Now "custom-ollama/llama3" will route to the custom instance
    ```

## Ollama Integration

### Local vs Cloud Detection

Timestep automatically detects whether to use local or cloud Ollama based on:

1. **Model Name Suffix**: Models ending with `-cloud` automatically use Ollama Cloud
2. **API Key Presence**: If `OLLAMA_API_KEY` is set, uses Ollama Cloud
3. **Base URL**: Explicit `base_url` parameter overrides automatic detection

### Lazy Client Initialization

The `OllamaModelProvider` uses lazy initialization to avoid errors when Ollama isn't available:

- The Ollama client is only created when `get_model()` is first called
- This allows the provider to be instantiated even if Ollama isn't running
- Errors only occur when actually trying to use an Ollama model

=== "Python"

    ```python
    # This won't fail even if Ollama isn't running
    provider = OllamaModelProvider()
    
    # Error only occurs here if Ollama isn't available
    model = provider.get_model("llama3")
    ```

=== "TypeScript"

    ```typescript
    // This won't fail even if Ollama isn't running
    const provider = new OllamaModelProvider();
    
    // Error only occurs here if Ollama isn't available
    const model = await provider.getModel('llama3');
    ```

## Response Format Conversion

### Ollama to OpenAI Format

The `OllamaModel` class converts Ollama API responses to OpenAI-compatible format. This includes:

1. **Message Format**: Converting Ollama's message format to OpenAI's format
2. **Tool Calls**: Converting Ollama function calls to OpenAI tool calls
3. **Streaming**: Handling streaming responses and converting chunks
4. **Usage Statistics**: Converting token usage information
5. **IDs**: Generating OpenAI-compatible IDs for completions and tool calls

### Key Conversions

| Ollama Format | OpenAI Format |
|--------------|---------------|
| `message.content` | `choices[0].message.content` |
| `message.role` | `choices[0].message.role` |
| `done` (streaming) | `choices[0].finish_reason` |
| Function calls | Tool calls with `function` format |

## Design Decisions

### Why Prefix-Based Routing?

- **Simple and Intuitive**: Model names like `ollama/llama3` are self-documenting
- **Flexible**: Easy to add new providers without changing existing code
- **Backward Compatible**: Unprefixed models default to OpenAI, maintaining compatibility

### Why Lazy Initialization?

- **Graceful Degradation**: Code can be written that works with or without Ollama
- **Performance**: Avoids unnecessary client creation
- **Error Handling**: Errors occur at the right time (when using the model, not when creating the provider)

### Why Response Conversion?

- **SDK Compatibility**: Ensures seamless integration with OpenAI Agents SDK
- **Unified Interface**: Applications don't need to handle different response formats
- **Future-Proof**: Easy to add support for other providers that need conversion

## Cross-Language Parity

Timestep maintains feature parity between Python and TypeScript implementations:

- **Same API**: Method names and signatures are equivalent
- **Same Behavior**: Routing logic and response conversion work identically
- **Same Features**: All features available in both languages

This allows teams to use the same patterns and code structure regardless of their language choice.

## Performance Considerations

- **Provider Caching**: Fallback providers are cached to avoid repeated creation
- **Lazy Loading**: Ollama client is only created when needed
- **Minimal Overhead**: Routing logic adds negligible overhead to model requests

## Security Considerations

- **API Keys**: API keys should be stored in environment variables, not in code
- **Local Ollama**: Defaults to localhost, but can be configured for remote instances
- **Cloud Authentication**: Ollama Cloud uses Bearer token authentication

