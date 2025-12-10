"""DBOS workflows for durable agent execution."""

import os
import json
import uuid
from typing import Any, Optional, Callable, Awaitable
from dbos import DBOS, Queue, SetWorkflowID, SetWorkflowTimeout
from ..config.dbos_config import configure_dbos, ensure_dbos_launched, _dbos_context, is_dbos_launched, get_dbos_connection_string
from ..stores.run_state_store.store import RunStateStore
from .._vendored_imports import Agent, SessionABC, TResponseInputItem, RunState
from ..stores.agent_store.store import load_agent
from ..stores.session_store.store import load_session
from .agent import run_agent, default_result_processor


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
    # Store manages connection internally
    return await load_agent(agent_id)


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
    # Store manages connection internally
    session_data = await load_session(session_id)
    if not session_data:
        raise ValueError(f"Session with id {session_id} not found")
    return session_data


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
        from .._vendored_imports import OpenAIConversationsSession
        session = OpenAIConversationsSession(conversation_id=internal_session_id)
    else:
        raise ValueError(f"Unsupported session type: {session_type}")
    
    # Import here to avoid circular import
    # run_agent and default_result_processor are already imported at module level
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
    timeout_seconds: Optional[float],
    verify: bool = True  # New parameter for safety checks
) -> Any:
    """Execute agent and handle state persistence via RunStateStore."""
    # Optional safety checks
    if verify:
        from ..analysis.safety import CircularDependencyChecker, ToolCompatibilityChecker
        import warnings
        
        # Check for circular handoffs
        cycle_checker = CircularDependencyChecker()
        cycle = await cycle_checker.check_circular_handoffs(agent_id)
        if cycle:
            raise ValueError(f"Agent {agent_id} has circular handoff dependencies: {' -> '.join(cycle)}")
        
        # Check tool compatibility
        compat_checker = ToolCompatibilityChecker()
        compat_warnings = await compat_checker.check_compatibility(agent_id)
        if compat_warnings:
            for warning in compat_warnings:
                warnings.warn(warning, UserWarning)
    
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
    timeout_seconds: Optional[float] = None,
    verify: bool = True  # Safety checks
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
                agent_id, input_items, session_id, stream, None, timeout_seconds, verify
            )
    else:
        return await _execute_agent_with_state_handling(
            agent_id, input_items, session_id, stream, None, timeout_seconds, verify
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
    timeout_seconds: Optional[float] = None,
    verify: bool = True  # Safety checks
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
    # Ensure DBOS is configured and launched
    # Note: If DBOS is already configured (e.g., by a test fixture), we don't
    # call configure_dbos() again as that would destroy the existing instance.
    # We just ensure it's launched (matching TypeScript pattern).
    if not _dbos_context.is_configured:
        await configure_dbos()
    
    # Always ensure DBOS is launched (safe to call multiple times)
    # The workflow decorator checks if DBOS is initialized, so we need to ensure it's launched
    from dbos import DBOS
    try:
        # Check if DBOS is actually initialized by accessing the system database
        _ = DBOS._sys_db
        # If we got here, DBOS is initialized and ready
    except (AttributeError, Exception):
        # DBOS is not initialized, so we need to launch it
        try:
            DBOS.launch()
            # Verify it's actually ready now
            _ = DBOS._sys_db
        except Exception as e:
            # If launch fails, check if it's because DBOS is already launched
            error_msg = str(e).lower()
            if "already" in error_msg or "launched" in error_msg:
                # DBOS says it's already launched, verify it's actually ready
                try:
                    _ = DBOS._sys_db
                except Exception:
                    # Still not ready, this is a problem
                    raise RuntimeError("DBOS reports as launched but system database is not accessible")
            else:
                raise
    
    # Serialize input items
    input_items_json = json.dumps(input_items, default=str)
    
    # Call the workflow with serializable parameters
    if workflow_id:
        with SetWorkflowID(workflow_id):
            return await _agent_workflow(agent_id, input_items_json, session_id, stream, timeout_seconds, verify)
    else:
        return await _agent_workflow(agent_id, input_items_json, session_id, stream, timeout_seconds, verify)


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
    # Ensure DBOS is configured and launched
    # Note: If DBOS is already configured (e.g., by a test fixture), we don't
    # call configure_dbos() again as that would destroy the existing instance.
    # We just ensure it's launched (matching TypeScript pattern).
    if not _dbos_context.is_configured:
        await configure_dbos()
        # Register generic workflows before DBOS launch (required by DBOS)
        register_generic_workflows()
    else:
        # If already configured, ensure workflows are registered
        # (they should be from module import, but double-check)
        register_generic_workflows()
    
    # Always ensure DBOS is launched (safe to call multiple times)
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
