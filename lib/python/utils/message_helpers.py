"""Message conversion helpers for A2A, MCP, and OpenAI formats."""

import json
from typing import Dict, List, Any, Tuple, Optional
from a2a.types import Message, Part, DataPart, Role
from a2a.client.helpers import create_text_message_object
from mcp.types import Tool, CallToolResult
from openai.types.chat import ChatCompletionMessageToolCall

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
        tool_call: OpenAI tool call (can be ChatCompletionMessageToolCall or dict)
        
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
