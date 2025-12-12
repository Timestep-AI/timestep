"""Streaming agent core using OpenAI streaming API."""

import json
import os
from typing import Any

from openai import AsyncOpenAI, pydantic_function_tool
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .agent_events import AgentEventEmitter
from .tools import call_function


def merge_tool_calls(
    existing_tool_calls: list[dict[str, Any]],
    delta_tool_calls: list[Any],
) -> list[dict[str, Any]]:
    """Merge incremental tool call deltas into complete tool calls.

    Args:
        existing_tool_calls: List of existing tool call dictionaries
        delta_tool_calls: List of tool call deltas from streaming API

    Returns:
        Updated list of tool call dictionaries
    """
    # Convert existing tool calls to a map keyed by index for easier merging
    tool_calls_map = {i: tc.copy() for i, tc in enumerate(existing_tool_calls)}

    for delta_tc in delta_tool_calls:
        index = delta_tc.index

        if index not in tool_calls_map:
            # New tool call
            tool_calls_map[index] = {
                "id": delta_tc.id if hasattr(delta_tc, 'id') and delta_tc.id else "",
                "type": delta_tc.type if hasattr(delta_tc, 'type') else "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            }

        # Merge function name
        if hasattr(delta_tc, 'function') and delta_tc.function:
            if hasattr(delta_tc.function, 'name') and delta_tc.function.name:
                tool_calls_map[index]["function"]["name"] = delta_tc.function.name

            # Merge arguments (they come as incremental strings)
            if hasattr(delta_tc.function, 'arguments') and delta_tc.function.arguments:
                tool_calls_map[index]["function"]["arguments"] += delta_tc.function.arguments

    # Convert back to list, sorted by index
    return [tool_calls_map[i] for i in sorted(tool_calls_map.keys())]


def serialize(result: Any) -> str:
    """Serialize tool execution result to JSON string.

    Args:
        result: Tool execution result (any type)

    Returns:
        JSON string representation
    """
    try:
        if isinstance(result, str):
            # Try to parse as JSON first, if it fails return as-is
            try:
                json.loads(result)
            except (json.JSONDecodeError, TypeError):
                # Not valid JSON, wrap in JSON string
                return json.dumps(result)
            else:
                return result
        else:
            return json.dumps(result)
    except Exception:
        # Fallback: convert to string and wrap
        return json.dumps(str(result))


async def run_agent(  # noqa: PLR0915
    messages: list[ChatCompletionMessageParam],
    tools: list[type[BaseModel]] | None = None,
    model: str = "gpt-4.1",
    api_key: str | None = None,
    event_emitter: AgentEventEmitter | None = None,
    context_id: str | None = None,
) -> str:
    """Run agent with streaming OpenAI API and tool support.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        tools: Optional list of Pydantic models
        model: OpenAI model name (default: "gpt-4.1")
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        event_emitter: Optional event emitter for agent events
        context_id: Optional context ID for handoff tool

    Returns:
        Final assistant response as string
    """
    # Initialize OpenAI client
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key parameter.")

    client = AsyncOpenAI(api_key=api_key)

    while True:
        # 1. STREAM ASSISTANT OUTPUT
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[pydantic_function_tool(tool) for tool in tools] if tools else None,  # type: ignore[arg-type]
            stream=True,
        )

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": "",
        }
        tool_calls: list[dict[str, Any]] = []

        async for event in stream:  # type: ignore[union-attr]
            # Check if this is an assistant message delta event
            if hasattr(event, 'choices') and event.choices and len(event.choices) > 0:
                choice = event.choices[0]
                if hasattr(choice, 'delta'):
                    delta = choice.delta

                    # Handle content delta
                    if hasattr(delta, 'content') and delta.content:
                        assistant_msg["content"] += delta.content
                        if event_emitter:
                            event_emitter.emit("delta", {"content": delta.content})

                    # Handle tool_calls delta
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        tool_calls = merge_tool_calls(
                            tool_calls,
                            delta.tool_calls,
                        )
                        if event_emitter:
                            event_emitter.emit("delta", {"tool_calls": delta.tool_calls})

        # Only include tool_calls if there are any (OpenAI doesn't allow empty arrays)
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        # Append assistant message to conversation
        messages.append(assistant_msg)  # type: ignore[arg-type]

        # Emit assistant message event
        if event_emitter:
            try:
                await event_emitter.emit_async("assistant-message", {"message": assistant_msg})
            except Exception as e:
                print(f"[run_agent] Error in assistant-message event: {e}")

        # 2. MULTIPLE TOOL CALLS
        if tool_calls:
            tool_messages = []

            for tool_call in tool_calls:
                # --- HUMAN APPROVAL ---
                approved = False
                if event_emitter:
                    # Create a future that will be resolved by the event handler
                    import asyncio
                    future = asyncio.Future()
                    
                    event_emitter.emit("tool-approval-required", {
                        "tool_call": tool_call,
                        "resolve": lambda approved: future.set_result(approved),
                    })
                    
                    approved = await future
                else:
                    # Default: auto-approve if no event emitter provided
                    approved = True

                if approved:
                    # EXECUTE TOOL
                    # We only support function tool calls
                    tool_call_type = tool_call.get("type", "function")
                    if tool_call_type != "function":
                        raise ValueError(f"Unsupported tool call type: {tool_call_type}")
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                    
                    # For handoff tool, create callbacks from event emitter
                    on_approval_required_callback = None
                    on_child_message_callback = None
                    
                    if tool_name == "handoff" and event_emitter:
                        # Create approval callback that uses the event emitter
                        async def approval_callback(tool_call: dict[str, Any]) -> bool:
                            import asyncio
                            future = asyncio.Future()
                            event_emitter.emit("tool-approval-required", {
                                "tool_call": tool_call,
                                "resolve": lambda approved: future.set_result(approved),
                            })
                            return await future
                        
                        on_approval_required_callback = approval_callback
                        
                        # Create child message callback that emits events
                        def child_message_callback(message: dict[str, Any]) -> None:
                            event_emitter.emit("child-message", {"message": message})
                        
                        on_child_message_callback = child_message_callback
                    
                    result = await call_function(
                        tool_name,
                        args,
                        on_approval_required=on_approval_required_callback,
                        source_context_id=context_id,
                        on_child_message=on_child_message_callback,
                    )
                    # Handle structured results with _meta
                    if isinstance(result, dict) and "content" in result:
                        content = result["content"]
                    else:
                        content = result
                    serialized_result = serialize(content)
                    
                    # Publish tool result via event emitter (handoff handles its own events)
                    if event_emitter and tool_name != "handoff":
                        event_emitter.emit("tool-result", {
                            "tool_call_id": tool_call["id"],
                            "tool_name": tool_name,
                            "result": serialized_result,
                        })
                    
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": serialized_result,
                    })
                else:
                    # SYNTHETIC ERROR TOOL MSG
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": serialize({"error": "Human rejected tool call"}),
                    })

            # Append ALL tool messages in order
            messages.extend(tool_messages)  # type: ignore[arg-type]
            continue

        # 3. NO TOOLS → FINAL ANSWER
        # Content can be string, array, or None - we ensure it's a string
        content = assistant_msg["content"]
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # If content is an array, extract text from text parts
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "".join(text_parts)
        else:
            return ""

