"""Eval framework - universal evaluation harness for agents."""

from .agent import AgentFn, agent_builtin_echo, agent_cmd_factory
from .episode import run_episode, EpisodeInfo
from .tools import ToolFn, tool_calc, tool_echo, DEFAULT_TOOLS, build_tools_schema, index_tool_calls, ToolCallRecord
from .graders import (
    Grader,
    FinalRegex,
    FinalContains,
    FinalJSON,
    ForbiddenTools,
    MaxToolCalls,
    ToolCallSequence,
    ToolResultJSON,
    BUILTIN_GRADERS,
    parse_grader_spec,
    aggregate_grades,
)
from .suite import run_suite, report

__all__ = [
    # Agent
    "AgentFn",
    "agent_builtin_echo",
    "agent_cmd_factory",
    # Episode
    "run_episode",
    "EpisodeInfo",
    # Tools
    "ToolFn",
    "tool_calc",
    "tool_echo",
    "DEFAULT_TOOLS",
    "build_tools_schema",
    "index_tool_calls",
    "ToolCallRecord",
    # Graders
    "Grader",
    "FinalRegex",
    "FinalContains",
    "FinalJSON",
    "ForbiddenTools",
    "MaxToolCalls",
    "ToolCallSequence",
    "ToolResultJSON",
    "BUILTIN_GRADERS",
    "parse_grader_spec",
    "aggregate_grades",
    # Suite
    "run_suite",
    "report",
]
