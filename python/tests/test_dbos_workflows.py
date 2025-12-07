"""Tests for DBOS workflow functionality."""

import pytest
import os
from timestep import (
    configure_dbos,
    ensure_dbos_launched,
    run_agent_workflow,
    queue_agent_workflow,
    create_scheduled_agent_workflow,
)
from timestep._vendored_imports import (
    Agent,
    OpenAIConversationsSession,
    ModelSettings,
    TResponseInputItem,
)


@pytest.fixture(scope="module")
def setup_dbos():
    """Set up DBOS for testing."""
    configure_dbos(name="timestep-test")
    ensure_dbos_launched()
    yield
    # Cleanup if needed


@pytest.mark.asyncio
async def test_configure_dbos(setup_dbos):
    """Test that DBOS can be configured."""
    # Configuration is done in fixture
    assert True  # If we get here, configuration worked


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
async def test_run_agent_workflow_basic(setup_dbos):
    """Test basic durable workflow execution."""
    agent = Agent(
        instructions="You are a helpful assistant. Answer concisely.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name="Test Assistant",
    )
    session = OpenAIConversationsSession()
    
    input_items: list[TResponseInputItem] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say 'hello' and nothing else."}]}
    ]
    
    result = await run_agent_workflow(
        agent=agent,
        input_items=input_items,
        session=session,
        stream=False,
        workflow_id="test-workflow-1"
    )
    
    assert result is not None
    assert hasattr(result, 'output')
    assert "hello" in result.output.lower()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
async def test_queue_agent_workflow(setup_dbos):
    """Test queued workflow execution."""
    agent = Agent(
        instructions="You are a helpful assistant. Answer concisely.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name="Test Assistant",
    )
    session = OpenAIConversationsSession()
    
    input_items: list[TResponseInputItem] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say 'queued' and nothing else."}]}
    ]
    
    handle = await queue_agent_workflow(
        agent=agent,
        input_items=input_items,
        session=session,
        stream=False,
        priority=1,
        deduplication_id="test-queue-1"
    )
    
    result = await handle.get_result()
    assert result is not None
    assert hasattr(result, 'output')
    assert "queued" in result.output.lower()


@pytest.mark.asyncio
async def test_create_scheduled_workflow(setup_dbos):
    """Test that scheduled workflows can be created."""
    agent = Agent(
        instructions="You are a helpful assistant.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name="Test Assistant",
    )
    session = OpenAIConversationsSession()
    
    input_items: list[TResponseInputItem] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
    ]
    
    # This should not raise an error
    await create_scheduled_agent_workflow(
        crontab="0 * * * *",  # Every hour
        agent=agent,
        input_items=input_items,
        session=session,
        stream=False
    )
    
    # If we get here, the workflow was created successfully
    assert True

