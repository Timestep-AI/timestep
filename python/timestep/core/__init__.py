"""Core agent-environment loop - the foundation of Timestep AI Agents SDK."""

from .types import JSON, Message, AgentFn, StreamingAgentFn, ToolFn
from .agent import agent_builtin_echo, create_agent
from .episode import run_episode, stream_episode, EpisodeInfo
from .tools import (
    tool_calc,
    tool_echo,
    DEFAULT_TOOLS,
    build_tools_schema,
    index_tool_calls,
    ToolCallRecord,
)

__all__ = [
    # Types
    "JSON",
    "Message",
    "AgentFn",
    "StreamingAgentFn",
    "ToolFn",
    # Agent harness
    "agent_builtin_echo",
    "create_agent",
    # Episode runner
    "run_episode",
    "stream_episode",
    "EpisodeInfo",
    # Tools
    "tool_calc",
    "tool_echo",
    "DEFAULT_TOOLS",
    "build_tools_schema",
    "index_tool_calls",
    "ToolCallRecord",
]
