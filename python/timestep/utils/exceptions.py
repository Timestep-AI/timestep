"""Custom exceptions for agent execution system."""


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""
    pass


class ToolExecutionError(AgentError):
    """Raised when tool execution fails."""
    def __init__(self, tool_name: str, message: str, original_error: Exception | None = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' execution failed: {message}")


class AgentConfigError(AgentError):
    """Raised when agent configuration is invalid."""
    pass


class HandoffError(AgentError):
    """Raised when agent handoff fails."""
    def __init__(self, target_agent: str, message: str, original_error: Exception | None = None):
        self.target_agent = target_agent
        self.original_error = original_error
        super().__init__(f"Handoff to '{target_agent}' failed: {message}")

