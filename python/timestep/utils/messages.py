"""Message utilities for OpenAI chat protocol."""

from typing import List, Dict, Any
from .hashing import stable_hash


def is_assistant_message(msg: Dict[str, Any]) -> bool:
    """Check if message is from assistant."""
    return msg.get("role") == "assistant"


def is_tool_message(msg: Dict[str, Any]) -> bool:
    """Check if message is a tool result."""
    return msg.get("role") == "tool"


def last_assistant_content(messages: List[Dict[str, Any]]) -> str:
    """Get content of last assistant message."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return str(m.get("content", "") or "")
    return ""


def ensure_task_id(task: Dict[str, Any]) -> str:
    """Ensure task has an ID, generating one if missing."""
    if "id" not in task or not str(task["id"]).strip():
        task["id"] = stable_hash(task)
    return str(task["id"])
