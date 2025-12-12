"""Unified message format conversion between OpenAI and A2A formats."""

import uuid
from typing import Any


def extract_text_from_parts(parts: list[dict[str, Any]] | list[Any]) -> str:
    """Extract text content from A2A message parts.
    
    Handles both dictionary parts and Pydantic model parts.
    """
    text_parts = []
    for part in parts:
        # Handle both dict and Pydantic model objects
        if isinstance(part, dict):
            if part.get("kind") == "text" and "text" in part:
                text_parts.append(part["text"])
        else:
            # Pydantic model - use attribute access
            kind = getattr(part, "kind", None)
            text = getattr(part, "text", None)
            if kind == "text" and text:
                text_parts.append(text)
    return "".join(text_parts)


def openai_to_a2a(
    messages: list[dict[str, Any]],
    context_id: str,
    task_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert OpenAI messages to A2A format for display.
    
    Args:
        messages: List of OpenAI message dictionaries
        context_id: Context ID for A2A messages
        task_id: Optional task ID for A2A messages
        
    Returns:
        List of A2A message dictionaries
    """
    a2a_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "system":
            # Skip system messages in display
            continue

        if msg.get("role") == "user":
            # User message
            a2a_messages.append({
                "kind": "message",
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [
                    {
                        "kind": "text",
                        "text": msg.get("content", ""),
                    },
                ],
                "contextId": context_id,
                "taskId": task_id,
                "timestamp": None,  # Will be set by server if needed
            })
        elif msg.get("role") == "assistant":
            # Assistant message - check for tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                # This assistant message has tool calls
                a2a_messages.append({
                    "kind": "message",
                    "role": "agent",
                    "messageId": str(uuid.uuid4()),
                    "parts": (
                        [{"kind": "text", "text": msg.get("content", "")}]
                        if msg.get("content")
                        else []
                    ),
                    "contextId": context_id,
                    "taskId": task_id,
                    "timestamp": None,
                    "tool_calls": msg["tool_calls"],
                })
            else:
                # Regular assistant message
                content = msg.get("content", "")
                if content:
                    a2a_messages.append({
                        "kind": "message",
                        "role": "agent",
                        "messageId": str(uuid.uuid4()),
                        "parts": [{"kind": "text", "text": content}],
                        "contextId": context_id,
                        "taskId": task_id,
                        "timestamp": None,
                    })
        elif msg.get("role") == "tool":
            # Tool message - include it in the output
            tool_content = msg.get("content", "")
            if isinstance(tool_content, dict):
                tool_content = str(tool_content)
            tool_message: dict[str, Any] = {
                "kind": "message",
                "role": "tool",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": tool_content}],
                "contextId": context_id,
                "taskId": task_id,
                "timestamp": None,
            }
            # Store tool_call_id for matching
            if "tool_call_id" in msg:
                tool_message["tool_call_id"] = msg["tool_call_id"]
            a2a_messages.append(tool_message)

    return a2a_messages


def a2a_to_openai(task_history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert A2A message history to OpenAI message format.
    
    Args:
        task_history: List of A2A message dictionaries
        
    Returns:
        List of OpenAI message dictionaries
    """
    messages: list[dict[str, Any]] = []

    if not task_history:
        return messages

    for msg in task_history:
        role = msg.get("role", "user")
        parts = msg.get("parts", [])
        text = extract_text_from_parts(parts)

        # Map A2A roles to OpenAI roles
        if role == "agent":
            openai_role = "assistant"
            # Handle tool calls in assistant messages (text can be empty)
            if "tool_calls" in msg or "toolCalls" in msg:
                tool_calls = msg.get("tool_calls") or msg.get("toolCalls")
                messages.append({
                    "role": openai_role,
                    "content": text if text else None,
                    "tool_calls": tool_calls,
                })
                continue
        elif role == "user":
            openai_role = "user"
        elif role == "tool":
            # Tool messages
            tool_call_id = msg.get("tool_call_id") or msg.get("toolCallId", "")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": text,
            })
            continue
        else:
            continue

        # Skip messages with no text content (unless they have tool calls, handled above)
        if not text:
            continue

        # Skip approval responses - they should not be sent to the model
        is_approval_response = text.lower().strip() in ("approve", "reject")
        if is_approval_response:
            continue

        messages.append({
            "role": openai_role,
            "content": text,
        })

    return messages


# Legacy exports for backward compatibility
convert_openai_to_a2a = openai_to_a2a
convert_a2a_messages_to_openai = a2a_to_openai

