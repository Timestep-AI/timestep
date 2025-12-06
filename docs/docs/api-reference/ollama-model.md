# OllamaModel

The `OllamaModel` class is an internal implementation that converts Ollama API responses to OpenAI-compatible format. It implements the `Model` interface from the OpenAI Agents SDK.

!!! note "Internal Class"
    This class is typically not used directly. It's created automatically by `OllamaModelProvider` when you call `get_model()`. However, understanding how it works can be helpful for debugging and advanced use cases.

## Overview

`OllamaModel` handles:

- **Response Conversion**: Converts Ollama API responses to OpenAI format
- **Streaming Support**: Handles both streaming and non-streaming responses
- **Tool Calls**: Converts Ollama function calls to OpenAI tool calls
- **Usage Statistics**: Converts token usage information
- **ID Generation**: Generates OpenAI-compatible IDs for completions and tool calls

## Constructor

=== "Python"

    ```python
    OllamaModel(model: str, ollama_client: Any)
    ```

=== "TypeScript"

    ```typescript
    new OllamaModel(model: string, ollama_client: any)
    ```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `string` | The name of the Ollama model (e.g., `"llama3"`). |
| `ollama_client` | `Any` | The Ollama client instance (from `ollama` package). |

## Response Format Conversion

The main purpose of `OllamaModel` is to convert Ollama's response format to OpenAI's format. Here's what gets converted:

### Message Format

| Ollama | OpenAI |
|--------|--------|
| `message.role` | `choices[0].message.role` |
| `message.content` | `choices[0].message.content` |
| `message.tool_calls` | `choices[0].message.tool_calls` |

### Tool Calls

Ollama function calls are converted to OpenAI tool calls:

- Function names are preserved
- Function arguments are JSON-stringified
- Tool call IDs are generated if not present or invalid

### Usage Statistics

Token usage is converted:

| Ollama | OpenAI |
|--------|--------|
| `eval_count` | `usage.completion_tokens` |
| `prompt_eval_count` | `usage.prompt_tokens` |
| `eval_count + prompt_eval_count` | `usage.total_tokens` |

### Response Structure

The converted response follows OpenAI's chat completion format:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama3",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "...",
      "tool_calls": [...]
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

## Usage

Typically, you don't create `OllamaModel` directly. It's created by `OllamaModelProvider`:

=== "Python"

    ```python
    from timestep import OllamaModelProvider

    provider = OllamaModelProvider()
    # This internally creates an OllamaModel
    model = provider.get_model("llama3")
    ```

=== "TypeScript"

    ```typescript
    import { OllamaModelProvider } from '@timestep-ai/timestep';

    const provider = new OllamaModelProvider();
    // This internally creates an OllamaModel
    const model = await provider.getModel('llama3');
    ```

### Direct Usage (Advanced)

If you need to create an `OllamaModel` directly (e.g., for testing or custom scenarios):

=== "Python"

    ```python
    from timestep import OllamaModel
    from ollama import AsyncClient

    client = AsyncClient()
    model = OllamaModel("llama3", client)
    ```

=== "TypeScript"

    ```typescript
    import { OllamaModel } from '@timestep-ai/timestep';
    import { Ollama } from 'ollama';

    const client = new Ollama();
    const model = new OllamaModel('llama3', client);
    ```

## Implementation Details

### ID Generation

`OllamaModel` generates OpenAI-compatible IDs:

- **Completion IDs**: Format `chatcmpl-{29 random chars}`
- **Tool Call IDs**: Format `call_{24 random chars}`

### Streaming

The model handles streaming responses by:

1. Converting each stream chunk from Ollama format
2. Emitting events in OpenAI format
3. Aggregating the final response

### Error Handling

If Ollama returns an error or unexpected format, the model will:

- Attempt to convert what it can
- Raise appropriate errors for critical failures
- Preserve error information when possible

## See Also

- [OllamaModelProvider](ollama-model-provider.md) - The provider that creates OllamaModel instances
- [Architecture](../architecture.md) - For details on response format conversion
- [MultiModelProvider](multi-model-provider.md) - For using Ollama models with automatic routing
