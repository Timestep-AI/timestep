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


def convert_a2a_message_to_openai(message: Message) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert an A2A message directly to OpenAI Chat Completions format.
    
    Args:
        message: A2A Message object (can be dict or object)
        
    Returns:
        Tuple of (openai_message, tool_messages) where:
        - openai_message: The converted message (user or assistant), or None if it was a tool_results-only message
        - tool_messages: List of tool messages converted from tool_results (empty if no tool_results)
    """
    if not message:
        return None, []
    
    # Extract message data
    if isinstance(message, dict):
        message_role = message.get("role")
        message_parts = message.get("parts", [])
    else:
        message_role = getattr(message, "role", None)
        message_parts = getattr(message, "parts", []) if hasattr(message, "parts") else []
    
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
    
    # Determine role
    is_user = message_role == Role.user or (isinstance(message_role, str) and message_role == "user")
    is_agent = message_role == Role.agent or (isinstance(message_role, str) and message_role == "agent")
    
    # Convert to OpenAI format based on content
    openai_message = None
    tool_messages = []
    
    # Handle tool results (always from user messages)
    if tool_results:
        for tr in tool_results:
            if isinstance(tr, dict):
                call_id = tr.get("call_id", "")
                output = tr.get("output", "")
            else:
                call_id = getattr(tr, "call_id", "")
                output = getattr(tr, "output", "")
            
            if call_id:
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(output),
                })
    
    # Handle tool calls (always from agent messages)
    if tool_calls:
        openai_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                call_id = tc.get("call_id", "")
                name = tc.get("name", "")
                arguments = tc.get("arguments", {})
            else:
                call_id = getattr(tc, "call_id", "")
                name = getattr(tc, "name", "")
                arguments = getattr(tc, "arguments", {})
            
            openai_tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                }
            })
        
        openai_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": openai_tool_calls,
        }
    # Handle text messages
    elif text_content:
        role = "user" if is_user else "assistant"
        openai_message = {
            "role": role,
            "content": text_content,
        }
    
    return openai_message, tool_messages


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
