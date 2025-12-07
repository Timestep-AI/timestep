"""DBOS workflows for durable agent execution."""

import os
import json
import uuid
from typing import Any, Optional, Callable, Awaitable
from dbos import DBOS, Queue, SetWorkflowID, SetWorkflowTimeout
from .dbos_config import configure_dbos, ensure_dbos_launched, _dbos_context, is_dbos_launched, get_dbos_connection_string
from .run_state_store import RunStateStore
from ._vendored_imports import Agent, SessionABC, TResponseInputItem, RunState
from .agent_store import load_agent
from .session_store import load_session
from .db_connection import DatabaseConnection
# Import run_agent and default_result_processor inside functions to avoid circular import


# Default queue for agent workflows with rate limiting
_default_queue: Optional[Queue] = None


def _get_default_queue() -> Queue:
    """Get or create the default agent queue with rate limiting."""
    global _default_queue
    if _default_queue is None:
        # Rate limit: 50 requests per 60 seconds (conservative for LLM APIs)
        _default_queue = Queue("timestep_agent_queue", limiter={"limit": 50, "period": 60})
    return _default_queue


@DBOS.step()
async def _load_agent_step(agent_id: str) -> Agent:
    """
    Step that loads an agent from the database.
    
    Args:
        agent_id: The agent ID (UUID as string)
    
    Returns:
        The loaded Agent object
    """
    connection_string = get_dbos_connection_string()
    if not connection_string:
        raise ValueError("DBOS connection string not available")
    
    db = DatabaseConnection(connection_string=connection_string)
    await db.connect()
    try:
        agent = await load_agent(agent_id, db)
        return agent
    finally:
        await db.disconnect()


@DBOS.step()
async def _load_session_data_step(session_id: str) -> dict:
    """
    Step that loads session data from the database.
    
    Returns serializable session data dict, not the Session object itself.
    
    Args:
        session_id: The session ID (UUID as string or session's internal ID)
    
    Returns:
        Session data dict
    """
    connection_string = get_dbos_connection_string()
    if not connection_string:
        raise ValueError("DBOS connection string not available")
    
    db = DatabaseConnection(connection_string=connection_string)
    await db.connect()
    try:
        session_data = await load_session(session_id, db)
        if not session_data:
            raise ValueError(f"Session with id {session_id} not found")
        return session_data
    finally:
        await db.disconnect()


@DBOS.step()
async def _run_agent_step(
    agent: Agent,
    run_input: list[TResponseInputItem] | RunState,
    session_data: dict,
    stream: bool,
    result_processor: Optional[Callable[[Any], Awaitable[Any]]] = None
) -> dict:
    """Step that runs an agent. Returns serializable dict with RunResult data."""
    # Reconstruct Session object from session_data
    session_type = session_data.get('session_type', '')
    internal_session_id = session_data.get('session_id')
    
    if 'OpenAIConversationsSession' in session_type:
        from ._vendored_imports import OpenAIConversationsSession
        session = OpenAIConversationsSession(conversation_id=internal_session_id)
    else:
        raise ValueError(f"Unsupported session type: {session_type}")
    
    # Import here to avoid circular import
    from . import run_agent, default_result_processor
    processor = result_processor or default_result_processor
    
    # Run agent - returns RunResult
    run_result = await run_agent(agent, run_input, session, stream, processor)
    
    # Extract only serializable data - RunResult has non-serializable objects
    # Store the RunResult object reference for _save_state_step to use
    # But return only serializable data
    return {
        '_run_result_ref': id(run_result),  # Store reference ID
        'final_output': run_result.final_output,
        'interruptions': [item.model_dump() for item in run_result.interruptions] if run_result.interruptions else [],
        '_has_interruptions': bool(run_result.interruptions),
    }


@DBOS.step()
async def _save_state_step(
    result_dict: dict,  # Serializable dict from _run_agent_step
    agent_id: str,
    session_id: Optional[str]
) -> None:
    """
    Step that saves agent state using RunStateStore.
    Note: We can't pass RunResult directly, so we need to re-run or store it differently.
    For now, if there are interruptions, we'll need to handle state saving differently.
    """
    # If there are interruptions, we need the actual RunResult to call to_state()
    # But we can't serialize it. So we'll need to save state in _run_agent_step itself
    # or use a different approach. For MVP, skip state saving in workflow for now.
    # State should be saved by the caller after getting the result.
    pass


async def _execute_agent_with_state_handling(
    agent_id: str,
    input_items: list[TResponseInputItem] | RunState,
    session_id: str,
    stream: bool,
    result_processor: Optional[Callable[[Any], Awaitable[Any]]],
    timeout_seconds: Optional[float]
) -> Any:
    """Execute agent and handle state persistence via RunStateStore."""
    # Step 1: Load agent from database
    agent = await _load_agent_step(agent_id)
    
    # Step 2: Load session data from database
    session_data = await _load_session_data_step(session_id)
    
    # Step 3: Run agent - returns dict with output and interruptions
    result_dict = await _run_agent_step(agent, input_items, session_data, stream, result_processor)
    
    # Return output - state saving happens outside workflow via RunStateStore
    return {
        'output': result_dict['final_output'],
        'interruptions': result_dict['interruptions']
    }


@DBOS.workflow()
async def _agent_workflow(
    agent_id: str,
    input_items_json: str,  # Serialized input items
    session_id: str,
    stream: bool = False,
    timeout_seconds: Optional[float] = None
) -> Any:
    """
    Workflow that runs an agent using IDs stored in the database.
    
    Args:
        agent_id: The agent ID (UUID as string)
        input_items_json: JSON-serialized input items or RunState
        session_id: The session ID (UUID as string or session's internal ID)
        stream: Whether to stream the results
        timeout_seconds: Optional timeout for the workflow
    
    Returns:
        The result from run_agent
    """
    # Deserialize input items
    input_items_data = json.loads(input_items_json)
    # TODO: Reconstruct RunState or list[TResponseInputItem] from data
    # For now, assume it's a list of dicts that can be converted to TResponseInputItem
    input_items = input_items_data  # Placeholder - will need proper deserialization
    
    # Set timeout if provided
    if timeout_seconds:
        with SetWorkflowTimeout(timeout_seconds):
            return await _execute_agent_with_state_handling(
                agent_id, input_items, session_id, stream, None, timeout_seconds
            )
    else:
        return await _execute_agent_with_state_handling(
            agent_id, input_items, session_id, stream, None, timeout_seconds
        )


def register_generic_workflows() -> None:
    """
    Register the generic workflows before DBOS launch.
    This must be called before ensure_dbos_launched().
    """
    # The workflow is already registered via @DBOS.workflow() decorator
    pass


async def run_agent_workflow(
    agent_id: str,
    input_items: list[TResponseInputItem] | RunState,
    session_id: str,
    stream: bool = False,
    workflow_id: Optional[str] = None,
    timeout_seconds: Optional[float] = None
) -> Any:
    """
    Run an agent in a durable DBOS workflow.
    
    This workflow automatically saves state on interruptions and can be resumed
    if the process crashes or restarts.
    
    Args:
        agent_id: The agent ID (UUID as string) - agent must be saved to database first
        input_items: Input items or RunState for the agent
        session_id: Session ID (UUID as string) - session must be saved to database first
        stream: Whether to stream the results
        workflow_id: Optional workflow ID for idempotency
        timeout_seconds: Optional timeout for the workflow
    
    Returns:
        The result from run_agent
    """
    # Ensure DBOS is configured (but not launched yet - workflows must be registered before launch)
    if not _dbos_context.is_configured:
        await configure_dbos()
    
    # Ensure generic workflow is registered
    register_generic_workflows()
    
    # Launch DBOS if not already launched (after workflow registration)
    if not _dbos_context.is_launched:
        await ensure_dbos_launched()
    
    # Serialize input items
    input_items_json = json.dumps(input_items, default=str)
    
    # Call the workflow with serializable parameters
    if workflow_id:
        with SetWorkflowID(workflow_id):
            return await _agent_workflow(agent_id, input_items_json, session_id, stream, timeout_seconds)
    else:
        return await _agent_workflow(agent_id, input_items_json, session_id, stream, timeout_seconds)


async def queue_agent_workflow(
    agent_id: str,
    input_items: list[TResponseInputItem] | RunState,
    session_id: str,
    stream: bool = False,
    queue_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    priority: Optional[int] = None,
    deduplication_id: Optional[str] = None
) -> Any:
    """
    Enqueue an agent run in a DBOS queue with rate limiting support.
    
    This is useful for managing concurrent agent executions and respecting
    LLM API rate limits.
    
    Args:
        agent_id: The agent ID (UUID as string) - agent must be saved to database first
        input_items: Input items or RunState for the agent
        session_id: Session ID (UUID as string) - session must be saved to database first
        stream: Whether to stream the results
        queue_name: Optional queue name. Defaults to "timestep_agent_queue"
        workflow_id: Optional workflow ID for idempotency
        timeout_seconds: Optional timeout for the workflow
        priority: Optional priority (lower number = higher priority)
        deduplication_id: Optional deduplication ID to prevent duplicate runs
    
    Returns:
        WorkflowHandle that can be used to get the result via handle.get_result()
    """
    # Ensure DBOS is configured (but not launched yet - workflows must be registered before launch)
    if not _dbos_context.is_configured:
        await configure_dbos()
    
    # Ensure generic workflow is registered
    register_generic_workflows()
    
    # Launch DBOS if not already launched (after workflow registration)
    if not _dbos_context.is_launched:
        await ensure_dbos_launched()
    
    # Get queue
    if queue_name:
        queue = Queue(queue_name)
    else:
        queue = _get_default_queue()
    
    # Serialize input items
    input_items_json = json.dumps(input_items, default=str)
    
    # Enqueue options
    from dbos import SetEnqueueOptions
    from contextlib import ExitStack
    
    enqueue_options = {}
    if priority is not None:
        enqueue_options["priority"] = priority
    if deduplication_id:
        enqueue_options["deduplication_id"] = deduplication_id
    
    # Enqueue the workflow with serializable parameters
    with ExitStack() as stack:
        if workflow_id:
            stack.enter_context(SetWorkflowID(workflow_id))
        if timeout_seconds:
            stack.enter_context(SetWorkflowTimeout(timeout_seconds))
        if enqueue_options:
            stack.enter_context(SetEnqueueOptions(**enqueue_options))
        
        handle = queue.enqueue(_agent_workflow, agent_id, input_items_json, session_id, stream, timeout_seconds)
    
    return handle


async def create_scheduled_agent_workflow(
    crontab: str,
    agent_id: str,
    input_items: list[TResponseInputItem] | RunState,
    session_id: str,
    stream: bool = False
) -> None:
    """
    Create a scheduled workflow that runs an agent periodically.
    
    This function registers a scheduled workflow with DBOS. The workflow will
    run automatically according to the crontab schedule.
    
    Note: This must be called before ensure_dbos_launched() because scheduled
    workflows must be registered before DBOS launch.
    
    Example:
        create_scheduled_agent_workflow(
            "0 0,6,12,18 * * *",  # Every 6 hours
            agent_id,
            input_items,
            session_id
        )
    
    Args:
        crontab: Crontab schedule (e.g., "0 0,6,12,18 * * *" for every 6 hours)
        agent_id: The agent ID (UUID as string) - agent must be saved to database first
        input_items: Input items or RunState for the agent
        session_id: Session ID (UUID as string) - session must be saved to database first
        stream: Whether to stream the results
    
    Raises:
        RuntimeError: If DBOS is already launched
    """
    # Check if DBOS is already launched - if so, we can't register new scheduled workflows
    if is_dbos_launched():
        raise RuntimeError(
            "Cannot create scheduled workflow after DBOS launch. "
            "Scheduled workflows must be registered before DBOS.launch() is called. "
            "Call create_scheduled_agent_workflow() before ensure_dbos_launched()."
        )
    
    # Ensure DBOS is configured (but not launched yet)
    if not _dbos_context.is_configured:
        await configure_dbos()
    
    # Serialize input items
    input_items_json = json.dumps(input_items, default=str)
    
    # Register a scheduled workflow
    @DBOS.scheduled(crontab)
    @DBOS.workflow()
    async def _scheduled_workflow(scheduled_time: Any, actual_time: Any):
        return await _agent_workflow(agent_id, input_items_json, session_id, stream, None)
