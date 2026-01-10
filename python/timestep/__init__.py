"""Timestep AI Agents SDK - Core agent-environment loop with evaluation harness."""

# Core agent-environment loop
from .core import (
    # Types
    JSON,
    Message,
    AgentFn,
    ToolFn,
    # Agent harness
    agent_builtin_echo,
    agent_cmd_factory,
    # Episode runner
    run_episode,
    EpisodeInfo,
    # Tools
    tool_calc,
    tool_echo,
    DEFAULT_TOOLS,
    build_tools_schema,
    index_tool_calls,
    ToolCallRecord,
)

# Evaluation harness
from .eval import (
    run_suite,
    report,
    Grader,
    FinalRegex,
    FinalContains,
    FinalJSON,
    TranscriptContains,
    TranscriptRegex,
    ForbiddenTools,
    MaxToolCalls,
    MinToolCalls,
    ToolCallSequence,
    ToolCallOrder,
    ToolResultJSON,
    OutcomeVerifier,
    LLMJudge,
    BUILTIN_GRADERS,
    parse_grader_spec,
    aggregate_grades,
)

__all__ = [
    # Core types
    "JSON",
    "Message",
    "AgentFn",
    "ToolFn",
    # Core agent harness
    "agent_builtin_echo",
    "agent_cmd_factory",
    # Core episode runner
    "run_episode",
    "EpisodeInfo",
    # Core tools
    "tool_calc",
    "tool_echo",
    "DEFAULT_TOOLS",
    "build_tools_schema",
    "index_tool_calls",
    "ToolCallRecord",
    # Evaluation harness
    "run_suite",
    "report",
    # Graders
    "Grader",
    "FinalRegex",
    "FinalContains",
    "FinalJSON",
    "TranscriptContains",
    "TranscriptRegex",
    "ForbiddenTools",
    "MaxToolCalls",
    "MinToolCalls",
    "ToolCallSequence",
    "ToolCallOrder",
    "ToolResultJSON",
    "OutcomeVerifier",
    "LLMJudge",
    "BUILTIN_GRADERS",
    "parse_grader_spec",
    "aggregate_grades",
]
