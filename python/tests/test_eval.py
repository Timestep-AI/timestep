"""Tests for Timestep AI Agents SDK - core and eval modules."""

import json
import tempfile
from pathlib import Path

import pytest

from timestep import (
    run_episode,
    agent_builtin_echo,
    DEFAULT_TOOLS,
    tool_calc,
    tool_echo,
)
from timestep import (
    FinalContains,
    FinalRegex,
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
    parse_grader_spec,
    aggregate_grades,
)
from timestep.core.tools import ToolCallRecord
from timestep.core.episode import EpisodeInfo
from timestep.utils import read_jsonl, write_jsonl, ensure_task_id


def test_agent_builtin_echo():
    """Test builtin echo agent harness."""
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    result = agent_builtin_echo(messages, {})
    assert result["role"] == "assistant"
    assert "Hello" in result["content"]


def test_tool_calc():
    """Test calc tool."""
    result = tool_calc({"expr": "2+2"})
    assert result["value"] == 4


def test_tool_echo():
    """Test echo tool."""
    result = tool_echo({"test": "value"})
    assert result["echo"]["test"] == "value"


def test_run_episode_simple():
    """Test running a simple episode using core agent-environment loop."""
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    result_messages, info = run_episode(
        initial_messages=messages,
        agent=agent_builtin_echo,
        tools=DEFAULT_TOOLS,
        tools_allowed=None,
        limits={"max_steps": 5},
        task_meta={"id": "test"},
        seed=0,
    )
    assert len(result_messages) == 2  # user + assistant
    assert info.terminated_reason == "final_answer"
    assert info.steps == 1
    assert info.input_tokens == 0  # Echo agent doesn't provide usage
    assert info.output_tokens == 0


def test_run_episode_with_tokens():
    """Test episode with token usage info."""
    def agent_with_usage(messages, context):
        return {
            "role": "assistant",
            "content": "Hello",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
    
    messages = [{"role": "user", "content": "Hello"}]
    result_messages, info = run_episode(
        initial_messages=messages,
        agent=agent_with_usage,
        tools=DEFAULT_TOOLS,
        tools_allowed=None,
        limits={"max_steps": 5},
        task_meta={"id": "test"},
        seed=0,
    )
    assert info.input_tokens == 10
    assert info.output_tokens == 5
    assert info.total_tokens == 15


def test_grader_final_contains():
    """Test FinalContains grader."""
    grader = FinalContains(substring="Hello")
    messages = [
        {"role": "user", "content": "Say Hello"},
        {"role": "assistant", "content": "Hello world"}
    ]
    info = EpisodeInfo("test", 1, 0, 1, 0, 1.0, "final_answer")
    result = grader.grade(messages, [], {}, info)
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_grader_final_regex():
    """Test FinalRegex grader."""
    grader = FinalRegex(pattern="^\\d+$")
    messages = [
        {"role": "user", "content": "Say a number"},
        {"role": "assistant", "content": "123"}
    ]
    info = EpisodeInfo("test", 1, 0, 1, 0, 1.0, "final_answer")
    result = grader.grade(messages, [], {}, info)
    assert result["passed"] is True


def test_grader_transcript_contains():
    """Test TranscriptContains grader."""
    grader = TranscriptContains(substring="Hello")
    messages = [
        {"role": "user", "content": "Say Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Say Hello again"},
        {"role": "assistant", "content": "Hello"}
    ]
    info = EpisodeInfo("test", 1, 0, 2, 0, 1.0, "final_answer")
    result = grader.grade(messages, [], {}, info)
    assert result["passed"] is True


def test_grader_transcript_regex():
    """Test TranscriptRegex grader."""
    grader = TranscriptRegex(pattern="\\d+")
    messages = [
        {"role": "user", "content": "Say a number"},
        {"role": "assistant", "content": "The answer is 42"}
    ]
    info = EpisodeInfo("test", 1, 0, 1, 0, 1.0, "final_answer")
    result = grader.grade(messages, [], {}, info)
    assert result["passed"] is True


def test_grader_min_tool_calls():
    """Test MinToolCalls grader."""
    grader = MinToolCalls(min_calls=2)
    messages = [
        {"role": "user", "content": "Use calc twice"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "calc", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "2", "content": "{}"},
        {"role": "assistant", "content": "Done"}
    ]
    tool_index = [
        ToolCallRecord("1", "calc", "{}", {}, "{}", {}, None),
        ToolCallRecord("2", "calc", "{}", {}, "{}", {}, None),
    ]
    info = EpisodeInfo("test", 1, 0, 3, 2, 1.0, "final_answer")
    result = grader.grade(messages, tool_index, {}, info)
    assert result["passed"] is True


def test_grader_tool_call_order():
    """Test ToolCallOrder grader."""
    grader = ToolCallOrder(expected_sequence=["calc", "echo"])
    messages = [
        {"role": "user", "content": "Use calc then echo"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "echo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "2", "content": "{}"},
        {"role": "assistant", "content": "Done"}
    ]
    tool_index = [
        ToolCallRecord("1", "calc", "{}", {}, "{}", {}, None),
        ToolCallRecord("2", "echo", "{}", {}, "{}", {}, None),
    ]
    info = EpisodeInfo("test", 1, 0, 3, 2, 1.0, "final_answer")
    result = grader.grade(messages, tool_index, {}, info)
    assert result["passed"] is True


def test_grader_outcome_verifier():
    """Test OutcomeVerifier grader."""
    def verifier(messages, tool_index, task):
        # Simple verifier: check if calc was called
        return any(r.name == "calc" for r in tool_index)
    
    grader = OutcomeVerifier(verifier_fn=verifier)
    messages = [
        {"role": "user", "content": "Use calc"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
        {"role": "assistant", "content": "Done"}
    ]
    tool_index = [
        ToolCallRecord("1", "calc", "{}", {}, "{}", {}, None),
    ]
    info = EpisodeInfo("test", 1, 0, 2, 1, 1.0, "final_answer")
    result = grader.grade(messages, tool_index, {}, info)
    assert result["passed"] is True


def test_parse_grader_spec():
    """Test parsing grader specs."""
    grader = parse_grader_spec("FinalContains:Hello")
    assert isinstance(grader, FinalContains)
    assert grader.substring == "Hello"
    
    grader2 = parse_grader_spec("MinToolCalls:2")
    assert isinstance(grader2, MinToolCalls)
    assert grader2.min_calls == 2
    
    grader3 = parse_grader_spec("ToolCallOrder:calc,echo")
    assert isinstance(grader3, ToolCallOrder)
    assert grader3.expected_sequence == ["calc", "echo"]


def test_aggregate_grades():
    """Test grade aggregation."""
    grades = [
        {"name": "Test1", "passed": True, "score": 1.0},
        {"name": "Test2", "passed": False, "score": 0.5},
    ]
    result = aggregate_grades(grades)
    assert result["passed"] is False
    assert result["score"] == 0.75


def test_jsonl_io():
    """Test JSONL I/O."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        path = Path(f.name)
    
    try:
        data = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]
        write_jsonl(path, data)
        
        read_data = list(read_jsonl(path))
        assert len(read_data) == 2
        assert read_data[0]["id"] == 1
    finally:
        path.unlink()


def test_ensure_task_id():
    """Test task ID generation."""
    task = {"messages": [{"role": "user", "content": "test"}]}
    task_id = ensure_task_id(task)
    assert "id" in task
    assert task_id == task["id"]
    
    # Should be stable
    task2 = {"messages": [{"role": "user", "content": "test"}]}
    task_id2 = ensure_task_id(task2)
    assert task_id == task_id2
