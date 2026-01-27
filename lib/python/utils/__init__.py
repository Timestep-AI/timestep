"""Utility functions for message conversion and helpers."""

# Import event_helpers first (no external dependencies beyond a2a)
from timestep.utils.event_helpers import (
    extract_event_data,
    extract_task_from_event,
    extract_task_from_tuple,
)

# Import message_helpers lazily to avoid requiring 'mcp' for clients that don't need it
# These will be imported on-demand when accessed
try:
    from timestep.utils.message_helpers import (
        extract_user_text_and_tool_results,
        convert_mcp_tool_to_openai,
        convert_openai_tool_call_to_mcp,
    )
except ImportError:
    # If mcp is not available, these functions won't be importable
    # This is fine for test clients that only need event_helpers
    extract_user_text_and_tool_results = None
    convert_mcp_tool_to_openai = None
    convert_openai_tool_call_to_mcp = None

__all__ = [
    "extract_event_data",
    "extract_task_from_event",
    "extract_task_from_tuple",
    "extract_user_text_and_tool_results",
    "convert_mcp_tool_to_openai",
    "convert_openai_tool_call_to_mcp",
]
