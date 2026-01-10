"""Tests for eval framework."""

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
from timestep.graders import (
    FinalContains,
    FinalRegex,
    ForbiddenTools,
    MaxToolCalls,
    ToolCallSequence,
    parse_grader_spec,
    aggregate_grades,
)
from timestep.utils import read_jsonl, write_jsonl, ensure_task_id


def test_agent_builtin_echo():
    """Test builtin echo agent."""
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
    """Test running a simple episode."""
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


def test_grader_final_contains():
    """Test FinalContains grader."""
    grader = FinalContains(substring="Hello")
    messages = [
        {"role": "user", "content": "Say Hello"},
        {"role": "assistant", "content": "Hello world"}
    ]
    result = grader.grade(messages, [], {}, None)
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_grader_final_regex():
    """Test FinalRegex grader."""
    grader = FinalRegex(pattern="^\\d+$")
    messages = [
        {"role": "user", "content": "Say a number"},
        {"role": "assistant", "content": "123"}
    ]
    result = grader.grade(messages, [], {}, None)
    assert result["passed"] is True


def test_parse_grader_spec():
    """Test parsing grader specs."""
    grader = parse_grader_spec("FinalContains:Hello")
    assert isinstance(grader, FinalContains)
    assert grader.substring == "Hello"


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
