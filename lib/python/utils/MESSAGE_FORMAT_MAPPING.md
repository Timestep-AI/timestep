# Message Format Mapping

This document defines the canonical mapping between A2A messages/events, AG-UI events, OpenAI Chat Completions messages, and OpenAI Responses API events.

## Core Concept: AG-UI as Canonical Format

AG-UI (Agent User Interaction Protocol) serves as the canonical format for message/event representation. Each A2A message stores a `canonical_type` field in its metadata pointing to the corresponding AG-UI event type. The message structure itself provides all data needed to construct the full AG-UI event, enabling conversion to other formats without redundancy.

### AG-UI Event Types

- **Run**: Corresponds to an A2A Task (one conversation rollout)
- **Step**: Individual agent processing step within a run
- **Text Messages**: User input and assistant responses
- **Tool Calls**: Function invocations and results
- **State**: Task state transitions

## Mapping Table

| A2A Structure | AG-UI Event Type | OpenAI Chat Completions | OpenAI Responses API | Metadata Field |
|---------------|------------------|------------------------|---------------------|----------------|
| **Task (initial)** | `RunStartedEvent` | N/A (not a message) | `event: run.started` | `canonical_type: "RunStartedEvent"` |
| **TaskStatusUpdateEvent** (state: created) | `RunStartedEvent` | N/A | `event: run.started` | `canonical_type: "RunStartedEvent"` |
| **TaskStatusUpdateEvent** (state: working) | `StepStartedEvent` | N/A | `event: run.step.started` | `canonical_type: "StepStartedEvent"` |
| **Message** (role: user, text only) | `TextMessageContentEvent` | `{"role": "user", "content": "..."}` | `event: message.delta` | `canonical_type: "TextMessageContentEvent"` |
| **Message** (role: agent, text only, streaming chunk) | `TextMessageChunkEvent` | N/A (intermediate) | `event: message.delta` | `canonical_type: "TextMessageChunkEvent"` |
| **Message** (role: agent, text only, final) | `TextMessageEndEvent` | `{"role": "assistant", "content": "..."}` | `event: message.done` | `canonical_type: "TextMessageEndEvent"` |
| **Message** (role: agent, with tool_calls, incremental/streaming) | `ToolCallArgsEvent` | N/A (intermediate) | `event: message.delta` (with tool_calls) | `canonical_type: "ToolCallArgsEvent"` |
| **Message** (role: agent, with tool_calls, final) | `ToolCallStartEvent` | `{"role": "assistant", "content": "", "tool_calls": [...]}` | `event: message.delta` (with tool_calls) | `canonical_type: "ToolCallStartEvent"` |
| **Message** (role: user, with tool_results) | `ToolCallResultEvent` | `{"role": "tool", "tool_call_id": "...", "content": "..."}` (multiple) | `event: message.delta` (with tool results) | `canonical_type: "ToolCallResultEvent"` |
| **TaskStatusUpdateEvent** (state: input-required) | `StepFinishedEvent` | N/A | `event: run.step.done` | `canonical_type: "StepFinishedEvent"` |
| **TaskStatusUpdateEvent** (state: completed) | `RunFinishedEvent` | N/A | `event: run.done` | `canonical_type: "RunFinishedEvent"` |
| **TaskStatusUpdateEvent** (state: failed/canceled) | `RunErrorEvent` | N/A | `event: run.error` | `canonical_type: "RunErrorEvent"` |

## Metadata Structure

Each A2A message stores in `metadata`:

```python
{
    "canonical_type": "TextMessageContentEvent",  # AG-UI event type
    # Optional: Additional fields if needed for mapping
    # The message structure itself provides all other data
}
```

## Self-Referential Mapping

The message structure provides all data needed to construct the full AG-UI event:

- **TextMessageContentEvent**: Extract from `message.parts[].text`
- **TextMessageChunkEvent**: Extract from `message.parts[].text` (intermediate chunk)
- **TextMessageEndEvent**: Extract from `message.parts[].text` (final message)
- **ToolCallStartEvent**: Extract from `message.parts[].data.tool_calls[]`
- **ToolCallResultEvent**: Extract from `message.parts[].data.tool_results[]`
- **RunStartedEvent**: Extract from `task.id`, `task.context_id`, `task.status.state`
- **StepStartedEvent/StepFinishedEvent**: Extract from `task.status.state`

## Conversion Examples

### Example 1: User Text Message

**A2A Message:**
```python
{
    "role": "user",
    "parts": [
        {"kind": "text", "text": "What's the weather?"}
    ],
    "metadata": {
        "canonical_type": "TextMessageContentEvent"
    }
}
```

**AG-UI Event:**
```python
{
    "type": "text-message-content",
    "message": {
        "role": "user",
        "content": "What's the weather?"
    }
}
```

**OpenAI Chat Completions:**
```python
{
    "role": "user",
    "content": "What's the weather?"
}
```

**OpenAI Responses API:**
```python
{
    "event": "message.delta",
    "data": {
        "role": "user",
        "content": "What's the weather?"
    }
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
    ],
    "metadata": {
        "canonical_type": "ToolCallStartEvent"
    }
}
```

**AG-UI Event:**
```python
{
    "type": "tool-call-start",
    "toolCall": {
        "id": "call_abc123",
        "name": "get_weather",
        "arguments": {"location": "Oakland"}
    }
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

**OpenAI Responses API:**
```python
{
    "event": "message.delta",
    "data": {
        "role": "assistant",
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
    ],
    "metadata": {
        "canonical_type": "ToolCallResultEvent"
    }
}
```

**AG-UI Event:**
```python
{
    "type": "tool-call-result",
    "toolCallId": "call_abc123",
    "result": "Sunny, 72°F"
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

**OpenAI Responses API:**
```python
{
    "event": "message.delta",
    "data": {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": "Sunny, 72°F"
    }
}
```

## Implementation Notes

1. **Canonical Type Only**: Store `canonical_type` string in metadata, not full event (avoids redundancy)
2. **Self-Referential**: Message structure provides all data needed to construct full AG-UI event
3. **AG-UI as Intermediate**: All format conversions go through AG-UI format
4. **Metadata on Messages**: Only messages get `canonical_type`; TaskStatusUpdateEvents determine type from state
5. **Streaming Handling**: Use `TextMessageChunkEvent` for intermediate chunks, `TextMessageEndEvent` for final messages

## Event Compaction

When processing task history, streaming events are compacted into final events using a map-reduce-map pattern:

1. **Map**: Convert A2A messages to AG-UI events using `convert_a2a_message_to_agui_event()`
2. **Reduce**: Compact streaming events using `compact_events()`:
   - `TextMessageChunkEvent` chunks are skipped if followed by `TextMessageEndEvent`
   - `ToolCallArgsEvent` chunks are merged into the final `ToolCallStartEvent` (arguments are merged)
3. **Map**: Convert compacted AG-UI events to OpenAI format using `convert_agui_event_to_openai_chat()`

This ensures that:
- All content from streaming events is preserved in the final messages
- Only final, complete messages are sent to OpenAI Chat Completions
- Tool call arguments are properly merged from incremental updates
