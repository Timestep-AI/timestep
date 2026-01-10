"""Core types for the agent-environment loop."""

from typing import Any, Callable, Dict, List

# Core type aliases
JSON = Dict[str, Any]
Message = Dict[str, Any]

# Agent harness: function that takes messages and context, returns assistant message
AgentFn = Callable[[List[Message], JSON], Message]  # (messages, context) -> assistant message

# Tool function: deterministic function that takes args and returns result
ToolFn = Callable[[JSON], Any]  # (args) -> result

__all__ = ["JSON", "Message", "AgentFn", "ToolFn"]
