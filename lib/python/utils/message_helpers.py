"""Message conversion helpers for A2A, MCP, and OpenAI formats."""

import json
from typing import Dict, List, Any, Tuple, Optional
from a2a.types import Message, Part, DataPart, Role
from mcp.types import Tool

# DataPart payload keys for tool routing
TOOL_CALLS_KEY = "tool_calls"
TOOL_RESULTS_KEY = "tool_results"


def extract_user_text_and_tool_results(message: Message) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract text and tool results from an A2A user message.

    Mapping:
      - TextPart -> OpenAI user message
      - DataPart(tool_results=...) -> OpenAI tool messages
    """
    text_content = ""
    tool_results: List[Dict[str, Any]] = []

    if message.parts:
        for part in message.parts:
            part_data = part.root if hasattr(part, "root") else part
            if hasattr(part_data, "kind") and part_data.kind == "text" and hasattr(part_data, "text"):
                text_content += part_data.text
            elif hasattr(part_data, "kind") and part_data.kind == "data" and hasattr(part_data, "data"):
                if isinstance(part_data.data, dict):
                    results = part_data.data.get(TOOL_RESULTS_KEY)
                    if isinstance(results, list):
                        tool_results.extend(results)

    return text_content, tool_results


def get_message_id(message: Message) -> Optional[str]:
    """Extract message ID from A2A message (handles both dict and object formats).
    
    Args:
        message: A2A Message object (can be dict or object)
        
    Returns:
        Message ID string, or None if not found
    """
    if isinstance(message, dict):
        return message.get("message_id") or message.get("messageId")
    else:
        return getattr(message, "message_id", None) or getattr(message, "messageId", None)


def convert_mcp_tool_to_openai(mcp_tool: Tool) -> Dict[str, Any]:
    """Convert MCP Tool to OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema if mcp_tool.inputSchema else {},
        }
    }


def convert_openai_tool_call_to_mcp(tool_call: Any) -> Dict[str, Any]:
    """Convert OpenAI tool call to MCP format.
    
    Args:
        tool_call: OpenAI tool call (can be dict or object with id, function.name, function.arguments)
        
    Returns:
        Dict with call_id, name, and arguments
    """
    # Handle both dict and object formats
    if isinstance(tool_call, dict):
        call_id = tool_call.get("id", "")
        name = tool_call.get("function", {}).get("name", "")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
    else:
        call_id = tool_call.id if hasattr(tool_call, "id") else ""
        name = tool_call.function.name if hasattr(tool_call, "function") else ""
        arguments_str = tool_call.function.arguments if hasattr(tool_call, "function") else "{}"
    
    # Parse arguments
    if isinstance(arguments_str, str):
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {}
    else:
        arguments = arguments_str
    
    return {
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def get_agui_event_type_from_a2a_message(message: Message) -> str:
    """Determine AG-UI event type from A2A message structure.
    
    Checks message metadata for canonical_type first, then infers from message structure.
    
    Args:
        message: A2A Message object (can be dict or object)
        
    Returns:
        AG-UI event type string (e.g., "TextMessageContentEvent", "ToolCallStartEvent")
    """
    # Check metadata for canonical_type
    if isinstance(message, dict):
        metadata = message.get("metadata", {})
        if isinstance(metadata, dict) and "canonical_type" in metadata:
            return metadata["canonical_type"]
        message_role = message.get("role")
        message_parts = message.get("parts", [])
    else:
        metadata = getattr(message, "metadata", None)
        if metadata and isinstance(metadata, dict) and "canonical_type" in metadata:
            return metadata["canonical_type"]
        message_role = getattr(message, "role", None)
        message_parts = getattr(message, "parts", []) if hasattr(message, "parts") else []
    
    # Infer from message structure
    has_text = False
    has_tool_calls = False
    has_tool_results = False
    
    for part in message_parts:
        part_data = part.root if hasattr(part, "root") else part
        if isinstance(part_data, dict):
            part_kind = part_data.get("kind")
            if part_kind == "text":
                has_text = True
            elif part_kind == "data":
                data = part_data.get("data", {})
                if isinstance(data, dict):
                    if TOOL_CALLS_KEY in data:
                        has_tool_calls = True
                    if TOOL_RESULTS_KEY in data:
                        has_tool_results = True
        elif hasattr(part_data, "kind"):
            if part_data.kind == "text":
                has_text = True
            elif part_data.kind == "data" and hasattr(part_data, "data"):
                if isinstance(part_data.data, dict):
                    if TOOL_CALLS_KEY in part_data.data:
                        has_tool_calls = True
                    if TOOL_RESULTS_KEY in part_data.data:
                        has_tool_results = True
    
    # Determine type based on role and content
    if message_role == Role.user or (isinstance(message_role, str) and message_role == "user"):
        if has_tool_results:
            return "ToolCallResultEvent"
        elif has_text:
            return "TextMessageContentEvent"
    elif message_role == Role.agent or (isinstance(message_role, str) and message_role == "agent"):
        if has_tool_calls:
            return "ToolCallStartEvent"
        elif has_text:
            # Default to TextMessageEndEvent for final messages
            # TextMessageChunkEvent would be determined by context (streaming)
            return "TextMessageEndEvent"
    
    # Default fallback
    return "TextMessageContentEvent"


def convert_a2a_message_to_agui_event(message: Message) -> Dict[str, Any]:
    """Convert A2A message to full AG-UI event structure.
    
    Args:
        message: A2A Message object (can be dict or object)
        
    Returns:
        AG-UI event dictionary
    """
    event_type = get_agui_event_type_from_a2a_message(message)
    
    # Extract message data
    if isinstance(message, dict):
        message_role = message.get("role")
        message_parts = message.get("parts", [])
        message_id = message.get("message_id") or message.get("messageId")
    else:
        message_role = getattr(message, "role", None)
        message_parts = getattr(message, "parts", []) if hasattr(message, "parts") else []
        message_id = getattr(message, "message_id", None) or getattr(message, "messageId", None)
    
    # Extract content from parts
    text_content = ""
    tool_calls = []
    tool_results = []
    
    for part in message_parts:
        part_data = part.root if hasattr(part, "root") else part
        if isinstance(part_data, dict):
            part_kind = part_data.get("kind")
            if part_kind == "text":
                text_content += part_data.get("text", "") or ""
            elif part_kind == "data":
                data = part_data.get("data", {})
                if isinstance(data, dict):
                    if TOOL_CALLS_KEY in data:
                        calls = data.get(TOOL_CALLS_KEY, [])
                        if isinstance(calls, list):
                            tool_calls.extend(calls)
                    if TOOL_RESULTS_KEY in data:
                        results = data.get(TOOL_RESULTS_KEY, [])
                        if isinstance(results, list):
                            tool_results.extend(results)
        elif hasattr(part_data, "kind"):
            if part_data.kind == "text" and hasattr(part_data, "text"):
                text_content += part_data.text or ""
            elif part_data.kind == "data" and hasattr(part_data, "data"):
                if isinstance(part_data.data, dict):
                    if TOOL_CALLS_KEY in part_data.data:
                        calls = part_data.data.get(TOOL_CALLS_KEY, [])
                        if isinstance(calls, list):
                            tool_calls.extend(calls)
                    if TOOL_RESULTS_KEY in part_data.data:
                        results = part_data.data.get(TOOL_RESULTS_KEY, [])
                        if isinstance(results, list):
                            tool_results.extend(results)
    
    # Build AG-UI event based on type
    if event_type == "TextMessageContentEvent" or event_type == "TextMessageEndEvent" or event_type == "TextMessageChunkEvent":
        # Convert "TextMessageContentEvent" -> "text-message-content"
        # Note: The normalization to handle "textmessage" vs "text-message" is done in consumer functions
        # to handle both formats correctly
        return {
            "type": event_type.lower().replace("event", "").replace("message", "message-"),
            "message": {
                "role": "user" if (message_role == Role.user or (isinstance(message_role, str) and message_role == "user")) else "assistant",
                "content": text_content,
            }
        }
    elif event_type == "ToolCallStartEvent" or event_type == "ToolCallArgsEvent":
        # AG-UI format for tool calls (both start and args use same structure)
        agui_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                agui_tool_calls.append({
                    "id": tc.get("call_id", ""),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", {}),
                })
            else:
                agui_tool_calls.append({
                    "id": getattr(tc, "call_id", ""),
                    "name": getattr(tc, "name", ""),
                    "arguments": getattr(tc, "arguments", {}),
                })
        if event_type == "ToolCallArgsEvent":
            return {
                "type": "tool-call-args",
                "toolCalls": agui_tool_calls,
            }
        else:
            return {
                "type": "tool-call-start",
                "toolCalls": agui_tool_calls,
            }
    elif event_type == "ToolCallResultEvent":
        # AG-UI format for tool results
        agui_results = []
        for tr in tool_results:
            if isinstance(tr, dict):
                agui_results.append({
                    "toolCallId": tr.get("call_id", ""),
                    "result": tr.get("output", ""),
                })
            else:
                agui_results.append({
                    "toolCallId": getattr(tr, "call_id", ""),
                    "result": getattr(tr, "output", ""),
                })
        return {
            "type": "tool-call-result",
            "results": agui_results,
        }
    
    # Default fallback
    return {
        "type": "text-message-content",
        "message": {
            "role": "user" if (message_role == Role.user or (isinstance(message_role, str) and message_role == "user")) else "assistant",
            "content": text_content,
        }
    }


def convert_agui_event_to_openai_chat(agui_event: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert AG-UI event to OpenAI Chat Completions format.
    
    Args:
        agui_event: AG-UI event dictionary
        
    Returns:
        Tuple of (openai_message, tool_messages) where:
        - openai_message: The converted message (user or assistant), or None
        - tool_messages: List of tool messages (empty if none)
    """
    event_type = agui_event.get("type", "")
    
    # Handle both "text-message-content" and "textmessage-content" formats (normalize)
    normalized_type = event_type.replace("textmessage", "text-message")
    
    if normalized_type in ["text-message-content", "text-message-end"]:
        message = agui_event.get("message", {})
        role = message.get("role", "user")
        content = message.get("content", "")
        return {"role": role, "content": content}, []
    
    elif normalized_type == "text-message-chunk":
        # Intermediate chunks are not included in OpenAI Chat Completions
        return None, []
    
    elif event_type == "tool-call-args":
        # Incremental tool call argument updates are not included in OpenAI Chat Completions
        return None, []
    
    elif event_type == "tool-call-start":
        tool_calls = agui_event.get("toolCalls", [])
        openai_tool_calls = []
        for tc in tool_calls:
            openai_tool_calls.append({
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("arguments", {})) if isinstance(tc.get("arguments"), dict) else str(tc.get("arguments", "")),
                }
            })
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": openai_tool_calls,
        }, []
    
    elif event_type == "tool-call-result":
        results = agui_event.get("results", [])
        tool_messages = []
        for result in results:
            tool_messages.append({
                "role": "tool",
                "tool_call_id": result.get("toolCallId", ""),
                "content": str(result.get("result", "")),
            })
        return None, tool_messages
    
    return None, []


def convert_agui_event_to_responses_api(agui_event: Dict[str, Any]) -> Dict[str, Any]:
    """Convert AG-UI event to OpenAI Responses API format.
    
    Args:
        agui_event: AG-UI event dictionary
        
    Returns:
        OpenAI Responses API event dictionary
    """
    event_type = agui_event.get("type", "")
    
    if event_type in ["text-message-content", "text-message-chunk", "text-message-end"]:
        message = agui_event.get("message", {})
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if event_type == "text-message-end":
            return {
                "event": "message.done",
                "data": {
                    "role": role,
                    "content": content,
                }
            }
        else:
            return {
                "event": "message.delta",
                "data": {
                    "role": role,
                    "content": content,
                }
            }
    
    elif event_type == "tool-call-args":
        # Incremental tool call argument updates
        tool_calls = agui_event.get("toolCalls", [])
        openai_tool_calls = []
        for tc in tool_calls:
            openai_tool_calls.append({
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("arguments", {})) if isinstance(tc.get("arguments"), dict) else str(tc.get("arguments", "")),
                }
            })
        return {
            "event": "message.delta",
            "data": {
                "role": "assistant",
                "tool_calls": openai_tool_calls,
            }
        }
    
    elif event_type == "tool-call-start":
        tool_calls = agui_event.get("toolCalls", [])
        openai_tool_calls = []
        for tc in tool_calls:
            openai_tool_calls.append({
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("arguments", {})) if isinstance(tc.get("arguments"), dict) else str(tc.get("arguments", "")),
                }
            })
        return {
            "event": "message.delta",
            "data": {
                "role": "assistant",
                "tool_calls": openai_tool_calls,
            }
        }
    
    elif event_type == "tool-call-result":
        results = agui_event.get("results", [])
        # Responses API sends tool results as message deltas
        tool_messages = []
        for result in results:
            tool_messages.append({
                "role": "tool",
                "tool_call_id": result.get("toolCallId", ""),
                "content": str(result.get("result", "")),
            })
        return {
            "event": "message.delta",
            "data": tool_messages[0] if tool_messages else {},
        }
    
    return {}


def convert_a2a_message_to_openai(message: Message) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert an A2A message from task history to OpenAI format.
    
    Uses AG-UI as intermediate format for conversion.
    
    Args:
        message: A2A Message object (can be dict or object)
        
    Returns:
        Tuple of (openai_message, tool_messages) where:
        - openai_message: The converted message (user or assistant), or None if it was a tool_results-only message
        - tool_messages: List of tool messages converted from tool_results (empty if no tool_results)
    """
    if not message:
        return None, []
    
    # Convert through AG-UI as intermediate format
    agui_event = convert_a2a_message_to_agui_event(message)
    return convert_agui_event_to_openai_chat(agui_event)


def convert_memory_history_to_openai_messages(memory_history: List[Message]) -> List[Dict[str, Any]]:
    """Convert list of A2A messages from memory store to OpenAI Chat Completions format.
    
    Since memory store only contains complete messages (no streaming deltas),
    we can convert directly without compaction.
    
    Args:
        memory_history: List of A2A Message objects from memory store
        
    Returns:
        Flat list of OpenAI Chat Completions messages (user, assistant, tool)
    """
    messages = []
    for msg in memory_history:
        openai_msg, tool_messages = convert_a2a_message_to_openai(msg)
        if openai_msg:
            messages.append(openai_msg)
        if tool_messages:
            messages.extend(tool_messages)
    return messages
