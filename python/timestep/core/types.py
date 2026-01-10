"""Core types for the agent-environment loop."""

from typing import Any, AsyncIterator, Callable, Dict, List, Union

# Core type aliases
JSON = Dict[str, Any]
Message = Dict[str, Any]

# Agent harness: function that takes messages and context, returns assistant message
AgentFn = Callable[[List[Message], JSON], Message]  # (messages, context) -> assistant message

# Streaming agent harness: function that takes messages and context, yields chunks
# Chunk format: {type: "content", delta: str} | {type: "tool_call", delta: {...}} | {type: "done"} | {type: "error", error: str}
StreamingAgentFn = Callable[[List[Message], JSON], AsyncIterator[Dict[str, Any]]]  # (messages, context) -> async iterator of chunks

# Tool function: deterministic function that takes args and returns result
ToolFn = Callable[[JSON], Any]  # (args) -> result

__all__ = ["JSON", "Message", "AgentFn", "StreamingAgentFn", "ToolFn"]
