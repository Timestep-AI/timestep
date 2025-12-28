"""Type definitions for multi-agent system."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, TypedDict

# ChatMessage format: {"role": "system"|"user"|"assistant"|"tool", "content": str, ...}
# Optional fields: "name" (for tool messages), "tool_call_id" (for tool messages),
#                  "tool_calls" (for assistant messages)
ChatMessage = Dict[str, Any]

# Tool is a callable function that takes args dict and returns result
Tool = Callable[[Dict[str, Any]], Any]


class AgentConfig(TypedDict, total=False):
    """Agent configuration dictionary.
    
    Required fields: name, model, instructions
    Optional fields: tools, handoffs, guardrails
    """
    name: str
    model: str
    instructions: str
    tools: List[Tool]
    handoffs: List["AgentConfig"]
    guardrails: List[Any]


class Event(TypedDict, total=False):
    """Execution event dictionary.
    
    All events have a "type" field. Other fields depend on event type.
    """
    type: str  # Event type (content_delta, tool_call, message, error, etc.)
    content: str  # For content_delta and message events
    tool: str  # For tool_call, tool_result, tool_error events
    args: Dict[str, Any]  # For tool_call events
    result: Dict[str, Any]  # For tool_result events
    error: str  # For error and tool_error events





