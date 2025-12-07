"""Tests for DBOS workflow functionality."""

import pytest
import pytest_asyncio
import os
import asyncio
import uuid
from timestep import (
    configure_dbos,
    ensure_dbos_launched,
    cleanup_dbos,
    run_agent_workflow,
    queue_agent_workflow,
    create_scheduled_agent_workflow,
    register_generic_workflows,
)
from timestep._vendored_imports import (
    Agent,
    OpenAIConversationsSession,
    ModelSettings,
    TResponseInputItem,
)
from timestep.agent_store import save_agent
from timestep.session_store import save_session
from timestep.db_connection import DatabaseConnection
from timestep.dbos_config import get_dbos_connection_string


@pytest_asyncio.fixture(scope="function")
async def setup_dbos():
    """Set up DBOS for testing."""
    import uuid
    await configure_dbos(name=f"timestep-test-{uuid.uuid4().hex[:8]}")
    # Register generic workflows before DBOS launch (required by DBOS)
    register_generic_workflows()
    await ensure_dbos_launched()
    yield
    # Cleanup
    await cleanup_dbos()


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
    # Create agent and session
    import uuid
    agent = Agent(
        instructions="You are a helpful assistant. Answer concisely.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name=f"Test Assistant Basic {uuid.uuid4().hex[:8]}",
    )
    session = OpenAIConversationsSession()
    
    # Save agent and session to database
    connection_string = get_dbos_connection_string()
    if not connection_string:
        pytest.skip("DBOS connection string not available")
    
    db = DatabaseConnection(connection_string=connection_string)
    await db.connect()
    try:
        agent_id = await save_agent(agent, db)
        session_id = await save_session(session, db)
    finally:
        await db.disconnect()
    
    input_items: list[TResponseInputItem] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say 'hello' and nothing else."}]}
    ]
    
    result = await run_agent_workflow(
        agent_id=agent_id,
        input_items=input_items,
        session_id=session_id,
        stream=False,
        workflow_id="test-workflow-1"
    )
    
    assert result is not None
    # Result is now a dict, not an object
    assert 'output' in result
    assert result['output'] is not None, f"Result output is None. Full result: {result}"
    
    # Extract text from output array if needed
    output_text = result['output']
    if isinstance(output_text, list):
        # Extract text from message items
        text_parts = []
        for item in output_text:
            if item.get('type') == 'message' and item.get('role') == 'assistant' and item.get('content'):
                for block in item['content']:
                    if block.get('type') == 'output_text' and block.get('text'):
                        text_parts.append(block['text'])
        output_text = ' '.join(text_parts)
    assert "hello" in str(output_text).lower(), f"Output text '{output_text}' does not contain 'hello'"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
async def test_queue_agent_workflow(setup_dbos):
    """Test queued workflow execution."""
    # Create agent and session
    import uuid
    agent = Agent(
        instructions="You are a helpful assistant. Answer concisely.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name=f"Test Assistant Queue {uuid.uuid4().hex[:8]}",
    )
    session = OpenAIConversationsSession()
    
    # Save agent and session to database
    connection_string = get_dbos_connection_string()
    if not connection_string:
        pytest.skip("DBOS connection string not available")
    
    db = DatabaseConnection(connection_string=connection_string)
    await db.connect()
    try:
        agent_id = await save_agent(agent, db)
        session_id = await save_session(session, db)
    finally:
        await db.disconnect()
    
    input_items: list[TResponseInputItem] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say 'queued' and nothing else."}]}
    ]
    
    dedup_id = f"test-queue-{uuid.uuid4().hex[:8]}"
    handle = await queue_agent_workflow(
        agent_id=agent_id,
        input_items=input_items,
        session_id=session_id,
        stream=False,
        deduplication_id=dedup_id
    )
    
    # Poll for workflow completion before getting result
    # Use same pattern as the working simplified test
    import time
    max_wait = 90  # Increase timeout
    start_time = time.time()
    status = None
    last_status = None
    
    while time.time() - start_time < max_wait:
        status_obj = handle.get_status()
        status = status_obj.status if hasattr(status_obj, 'status') else str(status_obj)
        
        # Log status changes for debugging
        if status != last_status:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Status: {last_status} -> {status}")
            last_status = status
        
        if str(status) in ['SUCCESS', 'FAILED', 'ERROR']:
            break
        await asyncio.sleep(1)  # Check every second instead of 0.5s
    
    if status is None or str(status) not in ['SUCCESS', 'FAILED', 'ERROR']:
        pytest.fail(f"Workflow did not complete after {max_wait} seconds. Status: {status}")
    
    # get_result() is synchronous and blocks until workflow completes
    result = handle.get_result()
    assert result is not None
    # Result is now a dict, not an object
    assert 'output' in result
    assert result['output'] is not None, f"Result output is None. Full result: {result}"
    
    # Extract text from output array if needed
    output_text = result['output']
    if isinstance(output_text, list):
        # Extract text from message items
        text_parts = []
        for item in output_text:
            if item.get('type') == 'message' and item.get('role') == 'assistant' and item.get('content'):
                for block in item['content']:
                    if block.get('type') == 'output_text' and block.get('text'):
                        text_parts.append(block['text'])
        output_text = ' '.join(text_parts)
    assert "queued" in str(output_text).lower(), f"Output text '{output_text}' does not contain 'queued'"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
async def test_create_scheduled_workflow(setup_dbos):
    """Test that scheduled workflows must be created before DBOS launch."""
    # This test verifies that scheduled workflows must be created before DBOS launch.
    # Since DBOS is launched in the fixture, this test should fail with an appropriate error.
    import uuid
    agent = Agent(
        instructions="You are a helpful assistant.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name=f"Test Assistant Scheduled {uuid.uuid4().hex[:8]}",
    )
    session = OpenAIConversationsSession()
    
    # Save agent and session to database
    connection_string = get_dbos_connection_string()
    if not connection_string:
        pytest.skip("DBOS connection string not available")
    
    db = DatabaseConnection(connection_string=connection_string)
    await db.connect()
    try:
        agent_id = await save_agent(agent, db)
        session_id = await save_session(session, db)
    finally:
        await db.disconnect()
    
    input_items: list[TResponseInputItem] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
    ]
    
    # This should raise an error because DBOS is already launched
    with pytest.raises(RuntimeError, match="Cannot create scheduled workflow after DBOS launch"):
        await create_scheduled_agent_workflow(
            crontab="0 * * * *",  # Every hour
            agent_id=agent_id,
            input_items=input_items,
            session_id=session_id,
            stream=False
        )

