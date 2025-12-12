"""Tests for A2A AgentExecutor implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Task, TaskState

from timestep.a2a.agent_executor import TimestepAgentExecutor
from timestep.a2a.message_converter import (
    a2a_to_openai as _convert_a2a_messages_to_openai,
    extract_text_from_parts as _extract_text_from_parts,
)


def test_extract_text_from_parts():
    """Test extracting text from A2A message parts."""
    parts = [
        {"kind": "text", "text": "Hello"},
        {"kind": "text", "text": " World"},
        {"kind": "other", "data": "ignored"},
    ]
    result = _extract_text_from_parts(parts)
    assert result == "Hello World"


def test_convert_a2a_messages_to_openai():
    """Test converting A2A messages to OpenAI format."""
    a2a_messages = [
        {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello"}],
        },
        {
            "role": "agent",
            "parts": [{"kind": "text", "text": "Hi there"}],
        },
    ]
    result = _convert_a2a_messages_to_openai(a2a_messages)
    
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "Hi there"


def test_convert_a2a_messages_with_tool():
    """Test converting A2A messages with tool calls."""
    a2a_messages = [
        {
            "role": "user",
            "parts": [{"kind": "text", "text": "What's the weather?"}],
        },
        {
            "role": "agent",
            "parts": [{"kind": "text", "text": ""}],
            "tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "Oakland"}'}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "parts": [{"kind": "text", "text": "Sunny"}],
        },
    ]
    result = _convert_a2a_messages_to_openai(a2a_messages)
    
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "What's the weather?"
    assert result[1]["role"] == "assistant"
    assert "tool_calls" in result[1]
    assert result[2]["role"] == "tool"
    assert result[2]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_agent_executor_initialization():
    """Test TimestepAgentExecutor initialization."""
    executor = TimestepAgentExecutor()
    assert executor.tools is not None
    assert len(executor.tools) == 2  # GetWeather and WebSearch
    assert executor.model == "gpt-4.1"


@pytest.mark.asyncio
async def test_agent_executor_execute_creates_task():
    """Test that executor creates a task when none exists."""
    executor = TimestepAgentExecutor()
    
    # Create proper mock message object
    from a2a.types import TextPart
    mock_message = Message(
        kind="message",
        role="user",
        messageId="test-msg-1",
        parts=[TextPart(kind="text", text="Hello")],
        contextId="test-context-id",
    )
    
    mock_context = MagicMock(spec=RequestContext)
    mock_context.current_task = None
    mock_context.message = mock_message
    mock_context.get_user_input = MagicMock(return_value="Hello")
    mock_context.context_id = "test-context-id"
    
    mock_event_queue = MagicMock(spec=EventQueue)
    mock_event_queue.enqueue_event = AsyncMock()
    
    # Mock run_agent to return immediately
    with patch("timestep.a2a.agent_executor.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Test response"
        
        await executor.execute(mock_context, mock_event_queue)
        
        # Verify task was created
        assert mock_event_queue.enqueue_event.call_count >= 2  # Task creation + working + completed
        # First call should be task creation
        first_call = mock_event_queue.enqueue_event.call_args_list[0]
        assert first_call is not None


@pytest.mark.asyncio
async def test_agent_executor_execute_with_existing_task():
    """Test executor with existing task."""
    executor = TimestepAgentExecutor()
    
    # Create mock task
    mock_task = MagicMock(spec=Task)
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_task.history = []
    
    mock_message = MagicMock()
    mock_message.parts = [MagicMock(kind="text", text="Hello")]
    mock_message.role = "user"
    
    mock_context = MagicMock(spec=RequestContext)
    mock_context.current_task = mock_task
    mock_context.message = mock_message
    mock_context.get_user_input = MagicMock(return_value="Hello")
    
    mock_event_queue = MagicMock(spec=EventQueue)
    mock_event_queue.enqueue_event = AsyncMock()
    
    with patch("timestep.a2a.agent_executor.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Test response"
        
        await executor.execute(mock_context, mock_event_queue)
        
        # Verify run_agent was called
        mock_run.assert_called_once()
        # Verify events were published
        assert mock_event_queue.enqueue_event.call_count >= 2


@pytest.mark.asyncio
async def test_agent_executor_cancel():
    """Test executor cancellation."""
    executor = TimestepAgentExecutor()
    
    mock_task = MagicMock(spec=Task)
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    
    mock_context = MagicMock(spec=RequestContext)
    mock_context.current_task = mock_task
    
    mock_event_queue = MagicMock(spec=EventQueue)
    mock_event_queue.enqueue_event = AsyncMock()
    
    await executor.cancel(mock_context, mock_event_queue)
    
    # Verify cancellation event was published
    mock_event_queue.enqueue_event.assert_called_once()
    call_args = mock_event_queue.enqueue_event.call_args[0][0]
    assert call_args.status.state == TaskState.canceled
    assert call_args.final is True


@pytest.mark.asyncio
async def test_agent_executor_error_handling():
    """Test executor error handling."""
    executor = TimestepAgentExecutor()
    
    from a2a.types import TextPart
    mock_message = Message(
        kind="message",
        role="user",
        messageId="test-msg-1",
        parts=[TextPart(kind="text", text="Hello")],
        contextId="test-context-id",
    )
    
    mock_context = MagicMock(spec=RequestContext)
    mock_context.current_task = None
    mock_context.message = mock_message
    mock_context.get_user_input = MagicMock(return_value="Hello")
    mock_context.context_id = "test-context-id"
    
    mock_event_queue = MagicMock(spec=EventQueue)
    mock_event_queue.enqueue_event = AsyncMock()
    
    # Mock run_agent to raise an error
    with patch("timestep.a2a.agent_executor.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = Exception("Test error")
        
        # Execute should not raise - it catches and publishes error event
        await executor.execute(mock_context, mock_event_queue)
        
        # Verify error event was published
        call_args_list = mock_event_queue.enqueue_event.call_args_list
        # Find the error event (should have state "failed")
        error_event = None
        for call in call_args_list:
            event = call[0][0]
            if hasattr(event, "status") and event.status.state == TaskState.failed:
                error_event = event
                break
        
        assert error_event is not None
        assert error_event.final is True


@pytest.mark.asyncio
async def test_agent_executor_publishes_working_status():
    """Test that executor publishes working status update."""
    executor = TimestepAgentExecutor()
    
    from a2a.types import TextPart
    mock_message = Message(
        kind="message",
        role="user",
        messageId="test-msg-1",
        parts=[TextPart(kind="text", text="Hello")],
        contextId="test-context-id",
    )
    
    mock_context = MagicMock(spec=RequestContext)
    mock_context.current_task = None
    mock_context.message = mock_message
    mock_context.get_user_input = MagicMock(return_value="Hello")
    mock_context.context_id = "test-context-id"
    
    mock_event_queue = MagicMock(spec=EventQueue)
    mock_event_queue.enqueue_event = AsyncMock()
    
    with patch("timestep.a2a.agent_executor.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Test response"
        
        await executor.execute(mock_context, mock_event_queue)
        
        # Find working status update
        call_args_list = mock_event_queue.enqueue_event.call_args_list
        working_update = None
        for call in call_args_list:
            event = call[0][0]
            if hasattr(event, "status") and event.status.state == TaskState.working:
                working_update = event
                break
        
        assert working_update is not None
        assert working_update.final is False


@pytest.mark.asyncio
async def test_agent_executor_publishes_completed_status():
    """Test that executor publishes completed status on success."""
    executor = TimestepAgentExecutor()
    
    from a2a.types import TextPart
    mock_message = Message(
        kind="message",
        role="user",
        messageId="test-msg-1",
        parts=[TextPart(kind="text", text="Hello")],
        contextId="test-context-id",
    )
    
    mock_context = MagicMock(spec=RequestContext)
    mock_context.current_task = None
    mock_context.message = mock_message
    mock_context.get_user_input = MagicMock(return_value="Hello")
    mock_context.context_id = "test-context-id"
    
    mock_event_queue = MagicMock(spec=EventQueue)
    mock_event_queue.enqueue_event = AsyncMock()
    
    with patch("timestep.a2a.agent_executor.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Test response"
        
        await executor.execute(mock_context, mock_event_queue)
        
        # Find completed status update
        call_args_list = mock_event_queue.enqueue_event.call_args_list
        completed_update = None
        for call in call_args_list:
            event = call[0][0]
            if hasattr(event, "status") and event.status.state == TaskState.completed:
                completed_update = event
                break
        
        assert completed_update is not None
        assert completed_update.final is True


@pytest.mark.asyncio
async def test_agent_executor_tool_approval_requirement():
    """Test tool approval requirement handling."""
    executor = TimestepAgentExecutor()
    
    from a2a.types import TextPart
    mock_message = Message(
        kind="message",
        role="user",
        messageId="test-msg-1",
        parts=[TextPart(kind="text", text="What's the weather?")],
        contextId="test-context-id",
    )
    
    mock_context = MagicMock(spec=RequestContext)
    mock_context.current_task = None
    mock_context.message = mock_message
    mock_context.get_user_input = MagicMock(return_value="What's the weather?")
    mock_context.context_id = "test-context-id"
    
    mock_event_queue = MagicMock(spec=EventQueue)
    mock_event_queue.enqueue_event = AsyncMock()
    
    import asyncio
    from timestep.core.agent_events import AgentEventEmitter
    
    # Mock run_agent to trigger tool approval via event emitter
    async def mock_run_agent_with_approval(*args, **kwargs):
        event_emitter = kwargs.get("event_emitter")
        if event_emitter:
            tool_call = {
                "id": "test-tool-call-id",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Oakland"}',
                },
            }
            # Emit tool-approval-required event - this will create a Future and wait for it
            future = asyncio.Future()
            # Emit the event - this will trigger the handler which is async
            await event_emitter.emit_async("tool-approval-required", {
                "tool_call": tool_call,
                "resolve": lambda approved: future.set_result(approved),
            })
            # Give it a moment to set up the resolver and publish the event
            await asyncio.sleep(0.1)
            # Manually resolve the approval by finding the resolver and calling it
            if executor._pending_approvals:
                # Get the first (and only) approval
                approval = next(iter(executor._pending_approvals.values()))
                approval["resolve"](True)  # Auto-approve for test
            # Wait for approval to complete (with timeout)
            try:
                await asyncio.wait_for(future, timeout=1.0)
            except asyncio.TimeoutError:
                # If timeout, just return
                pass
        return "Test response"
    
    with patch("timestep.a2a.agent_executor.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = mock_run_agent_with_approval
        
        await executor.execute(mock_context, mock_event_queue)
        
        # Verify input-required status was published
        call_args_list = mock_event_queue.enqueue_event.call_args_list
        input_required_update = None
        for call in call_args_list:
            event = call[0][0]
            if hasattr(event, "status") and event.status.state == TaskState.input_required:
                input_required_update = event
                break
        
        assert input_required_update is not None
        assert input_required_update.final is False  # Should be False to keep stream open

