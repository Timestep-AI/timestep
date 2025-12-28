"""Utility functions for message and tool processing."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from timestep.utils.constants import ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_TOOL, ROLE_USER
from timestep.utils.types import ChatMessage


def convert_message_to_openai_format(message: ChatMessage) -> Dict[str, Any]:
    """Convert internal ChatMessage format to OpenAI API format.
    
    Args:
        message: Internal chat message dictionary
        
    Returns:
        OpenAI-formatted message dictionary
    """
    openai_msg: Dict[str, Any] = {
        "role": message.get("role", ROLE_USER),
        "content": message.get("content", "")
    }
    
    # Add optional fields
    if "name" in message:
        openai_msg["name"] = message["name"]
    if "tool_call_id" in message:
        openai_msg["tool_call_id"] = message["tool_call_id"]
    
    return openai_msg


def convert_messages_to_openai_format(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert list of internal ChatMessage format to OpenAI API format.
    
    Args:
        messages: List of internal chat message dictionaries
        
    Returns:
        List of OpenAI-formatted message dictionaries
    """
    return [convert_message_to_openai_format(msg) for msg in messages]


def parse_tool_call_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Parse tool call arguments from OpenAI format.
    
    Args:
        tool_call: Tool call dictionary with function.arguments as JSON string
        
    Returns:
        Parsed arguments dictionary
        
    Raises:
        ValueError: If arguments cannot be parsed as JSON
    """
    arguments_str = tool_call.get("function", {}).get("arguments", "{}")
    try:
        return json.loads(arguments_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse tool call arguments: {e}") from e


def extract_tool_name_from_call(tool_call: Dict[str, Any]) -> str:
    """Extract tool name from tool call dictionary.
    
    Args:
        tool_call: Tool call dictionary
        
    Returns:
        Tool name string
    """
    return tool_call.get("function", {}).get("name", "")


def create_tool_message(
    content: Dict[str, Any],
    tool_call_id: str
) -> ChatMessage:
    """Create a tool message for the session.
    
    Args:
        content: Tool result content (will be JSON-encoded)
        tool_call_id: ID of the tool call this message responds to
        
    Returns:
        Tool message dictionary
    """
    return {
        "role": ROLE_TOOL,
        "content": json.dumps(content),
        "tool_call_id": tool_call_id
    }

