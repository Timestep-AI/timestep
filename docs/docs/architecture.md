# Architecture

How Timestep fits around the OpenAI Agents SDK and keeps runs durable across Python and TypeScript.

## Durable Execution Architecture

Timestep's durable execution system enables resumable agent workflows with cross-language state persistence.

### State Persistence Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Execution                           │
│                  (Python or TypeScript)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Interruption Detection                         │
│  - Tool calls requiring approval                           │
│  - Human-in-the-loop requests                              │
│  - Custom interruption points                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              RunStateStore                                  │
│  - Serialize RunState to JSON                              │
│  - Save to file or database                                │
│  - Cross-language compatible format                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              State Storage                                  │
│  - File-based (development)                                 │
│  - PostgreSQL (production)                                │
│  - PostgreSQL schema for durability                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Resume in Any Language                         │
│  - Load state from storage                                 │
│  - Approve interruptions                                   │
│  - Continue execution                                      │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Language State Transfer

Timestep's state format is identical between Python and TypeScript, enabling seamless state transfer:

1. **Python** saves state using `RunStateStore.save()`
2. State is serialized to JSON format
3. **TypeScript** loads the same JSON using `RunStateStore.load()`
4. Execution continues with full context preserved

### Storage

- **PostgreSQL**: Use `PG_CONNECTION_URI` environment variable to connect to your PostgreSQL database.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenAI Agents SDK                        │
│                  (Agent, Runner, etc.)                     │
│              (Complex API Surface)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Timestep Simplified API Layer                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  run_agent() - Simplified execution                 │  │
│  │  RunStateStore - State persistence                   │  │
│  │  result_processor parameter - Result processing      │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                      │                                       │
│  ┌───────────────────▼───────────────────────────────────┐ │
│  │              MultiModelProvider                        │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │         Model Name Parser                       │   │ │
│  │  │  - Parses model name (e.g., "ollama/gpt-oss:20b-cloud")   │   │ │
│  │  │  - Extracts prefix and actual model name        │   │ │
│  │  └──────────────────┬──────────────────────────────┘   │ │
│  │                      │                                   │ │
│  │  ┌───────────────────▼───────────────────────────────┐ │ │
│  │  │         Provider Router                          │ │ │
│  │  │  - Checks MultiModelProviderMap for custom maps  │ │ │
│  │  │  - Falls back to default providers              │ │ │
│  │  └───────┬───────────────────────┬─────────────────┘ │ │
│  └──────────┼───────────────────────┼────────────────────┘ │
└─────────────┼───────────────────────┼───────────────────────┘
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
   - **Prefix**: The part before the first `/` (e.g., `ollama` from `ollama/gpt-oss:20b-cloud`)
   - **Model Name**: The part after the `/` (e.g., `llama3` from `ollama/gpt-oss:20b-cloud`)

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
| `ollama/gpt-oss:20b-cloud` | Ollama | `ollama/gpt-oss:20b-cloud`, `ollama/gpt-oss:120b-cloud` |

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
    
    # Now "custom-ollama/gpt-oss:20b-cloud" will route to the custom instance
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
    
    // Now "custom-ollama/gpt-oss:20b-cloud" will route to the custom instance
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

## Package Structure

Timestep is organized into clear modules for maintainability and clarity:

### Core Modules
- **`core/`**: Core agent execution functions (`run_agent`/`runAgent`, `default_result_processor`/`defaultResultProcessor`)
- **`core/agent_workflow.py`/`core/agent_workflow.ts`**: DBOS workflows for durable agent execution

### Configuration
- **`config/`**: Configuration utilities (`dbos_config`, `app_dir`)

### Data Access Layer
- **`stores/`**: Data access layer with organized subfolders
  - **`agent_store/`**: Agent configuration persistence
  - **`session_store/`**: Session data persistence
  - **`run_state_store/`**: Run state persistence
  - **`shared/`**: Shared database utilities (`db_connection`, `schema`)
  - **`guardrail_registry.py`/`guardrail_registry.ts`**: Guardrail registration (in-memory, future: persistent)
  - **`tool_registry.py`/`tool_registry.ts`**: Tool registration (in-memory, future: persistent)

### Tools and Models
- **`tools/`**: Agent tools (e.g., `web_search`/`webSearch`)
- **`model_providers/`**: Model provider implementations (`OllamaModelProvider`, `MultiModelProvider`)
- **`models/`**: Model implementations (`OllamaModel`)

This structure provides clear separation of concerns and makes the codebase easier to navigate and maintain.

## Design Notes

- Durable execution: pause/resume with a single `RunStateStore` call.
- Cross-language: same state format; same APIs.
- Prefix-based routing: model names stay self-documenting; defaults stay OpenAI.
- Lazy initialization: providers are created only when needed; Ollama client comes up on first use.
- Organized architecture: clear module separation for maintainability.

## Cross-Language Parity

- Same API names/signatures.
- Same routing behavior and state format.
- State records in the database are interchangeable between languages.

## Performance Considerations

- PostgreSQL is the recommended backend for all environments.
- Minimal overhead in routing; provider instances are cached.

## Security Considerations

- Keep API keys in env vars.
- Lock down state storage (permissions, encryption) if you persist real data.
