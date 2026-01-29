# Message Format Mapping

This document defines the mapping between A2A messages and OpenAI Chat Completions format.

## Core Concept: Direct A2A → OpenAI Conversion

A2A messages are converted directly to OpenAI Chat Completions format without intermediate steps. The conversion extracts data from A2A message parts and restructures it into OpenAI format.

## Mapping Table

| A2A Structure | OpenAI Chat Completions |
|--------------|------------------------|
| **Message** (role: user, text only) | `{"role": "user", "content": "..."}` |
| **Message** (role: agent, text only) | `{"role": "assistant", "content": "..."}` |
| **Message** (role: agent, with tool_calls) | `{"role": "assistant", "content": "", "tool_calls": [...]}` |
| **Message** (role: user, with tool_results) | `{"role": "tool", "tool_call_id": "...", "content": "..."}` (multiple messages) |

## Conversion Examples

### Example 1: User Text Message

**A2A Message:**
```python
{
    "role": "user",
    "parts": [
        {"kind": "text", "text": "What's the weather?"}
    ]
}
```

**OpenAI Chat Completions:**
```python
{
    "role": "user",
    "content": "What's the weather?"
}
```

### Example 2: Assistant Message with Tool Calls

**A2A Message:**
```python
{
    "role": "agent",
    "parts": [
        {
            "kind": "data",
            "data": {
                "tool_calls": [
                    {
                        "call_id": "call_abc123",
                        "name": "get_weather",
                        "arguments": {"location": "Oakland"}
                    }
                ]
            }
        }
    ]
}
```

**OpenAI Chat Completions:**
```python
{
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\": \"Oakland\"}"
            }
        }
    ]
}
```

### Example 3: Tool Results

**A2A Message:**
```python
{
    "role": "user",
    "parts": [
        {
            "kind": "data",
            "data": {
                "tool_results": [
                    {
                        "call_id": "call_abc123",
                        "name": "get_weather",
                        "output": "Sunny, 72°F"
                    }
                ]
            }
        }
    ]
}
```

**OpenAI Chat Completions:**
```python
{
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": "Sunny, 72°F"
}
```

## Implementation Notes

1. **Direct Conversion**: A2A messages are converted directly to OpenAI format using `convert_a2a_message_to_openai()`
2. **Part Extraction**: The conversion extracts:
   - Text content from `parts[].text` (TextPart)
   - Tool calls from `parts[].data.tool_calls` (DataPart)
   - Tool results from `parts[].data.tool_results` (DataPart)
3. **Complete Messages Only**: Since memory store only contains complete messages (no streaming deltas), the conversion handles final messages only
4. **Tool Messages**: Tool results are converted to separate tool messages (one per tool call result)

## Conversion Function

The `convert_a2a_message_to_openai()` function:
- Takes an A2A Message object (dict or object)
- Returns a tuple of `(openai_message, tool_messages)` where:
  - `openai_message`: The converted user/assistant message (or None if only tool results)
  - `tool_messages`: List of tool messages (empty if none)
