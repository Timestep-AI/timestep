"""Integration tests for A2A functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from timestep.a2a.agent_executor import TimestepAgentExecutor
from timestep.a2a.server import create_agent_card, create_server


@pytest.mark.asyncio
async def test_full_execution_flow():
    """Test the full execution flow from message to response."""
    executor = TimestepAgentExecutor()
    
    # Create proper mock message object
    from a2a.types import TextPart, Message
    mock_message = Message(
        kind="message",
        role="user",
        messageId="test-msg-1",
        parts=[TextPart(kind="text", text="What's 2+2?")],
        contextId="test-context-id",
    )
    
    mock_context = MagicMock()
    mock_context.current_task = None
    mock_context.message = mock_message
    mock_context.get_user_input = MagicMock(return_value="What's 2+2?")
    mock_context.context_id = "test-context-id"
    
    mock_event_queue = MagicMock()
    mock_event_queue.enqueue_event = AsyncMock()
    
    # Mock run_agent to simulate a successful response
    with patch("timestep.a2a.agent_executor.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "2 + 2 = 4"
        
        await executor.execute(mock_context, mock_event_queue)
        
        # Verify run_agent was called
        mock_run.assert_called_once()
        
        # Verify multiple events were published (task creation, working, completed)
        assert mock_event_queue.enqueue_event.call_count >= 3
        
        # Verify final event is completed
        all_calls = mock_event_queue.enqueue_event.call_args_list
        final_call = all_calls[-1]
        final_event = final_call[0][0]
        assert final_event.status.state.value == "completed"  # TaskState enum
        assert final_event.final is True


def test_agent_card_structure():
    """Test that agent card has all required fields."""
    from timestep.a2a.postgres_agent_store import Agent
    
    # Create a test agent
    test_agent = Agent(
        id="test-agent",
        name="Test Agent",
        description="Test agent for testing",
        tools=["get_weather", "web_search"],
        model="gpt-4.1",
    )
    
    card = create_agent_card(test_agent)
    
    # Required fields
    assert card.name is not None
    assert card.url is not None
    assert card.version is not None
    assert card.default_input_modes is not None
    assert card.default_output_modes is not None
    assert card.capabilities is not None
    
    # Skills should be present
    assert card.skills is not None
    assert len(card.skills) > 0
    
    # Each skill should have required fields
    for skill in card.skills:
        assert skill.id is not None
        assert skill.name is not None
        assert skill.description is not None


def test_server_creation():
    """Test that server can be created without errors."""
    app = create_server(host="127.0.0.1", port=8080)
    assert app is not None
    
    # Verify it's a Starlette app (has routes attribute)
    assert hasattr(app, "routes") or hasattr(app, "router") or hasattr(app, "mount")

