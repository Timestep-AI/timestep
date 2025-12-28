"""Constants for agent execution system."""

# Event types emitted during agent execution
EVENT_CONTENT_DELTA = "content_delta"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_TOOL_ERROR = "tool_error"
EVENT_MESSAGE = "message"
EVENT_ERROR = "error"

# Message roles
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

# Default values
DEFAULT_MAX_ITERATIONS = 50

