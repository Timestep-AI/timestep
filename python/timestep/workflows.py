"""DBOS workflows for durable agent execution."""

import os
from typing import Any, Optional, Callable, Awaitable
from dbos import DBOS, Queue, SetWorkflowID, SetWorkflowTimeout
from .dbos_config import configure_dbos, ensure_dbos_launched
from .run_state_store import RunStateStore
from ._vendored_imports import Agent, SessionABC, TResponseInputItem, RunState
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
async def _run_agent_step(
    agent: Agent,
    run_input: list[TResponseInputItem] | RunState,
    session: SessionABC,
    stream: bool,
    result_processor: Optional[Callable[[Any], Awaitable[Any]]] = None
) -> Any:
    """
    Step that runs an agent. This must be a step because it's non-deterministic.
    
    Args:
        agent: The agent to run
        run_input: Input items or RunState for the agent
        session: Session for managing conversation state
        stream: Whether to stream the results
        result_processor: Optional result processor
    
    Returns:
        The result from run_agent
    """
    # Import here to avoid circular import
    from . import run_agent, default_result_processor
    processor = result_processor or default_result_processor
    return await run_agent(agent, run_input, session, stream, processor)


@DBOS.step()
async def _save_state_step(
    result: Any,
    state_store: RunStateStore
) -> None:
    """
    Step that saves agent state. This must be a step because it accesses the database.
    
    Args:
        result: The result from run_agent
        state_store: The RunStateStore instance
    """
    if hasattr(result, 'to_state'):
        state = result.to_state()
        await state_store.save(state)
    elif hasattr(result, 'state'):
        await state_store.save(result.state)
    else:
        raise ValueError("Result does not have to_state() or state attribute")


@DBOS.workflow()
async def run_agent_workflow(
    agent: Agent,
    input_items: list[TResponseInputItem] | RunState,
    session: SessionABC,
    stream: bool = False,
    result_processor: Optional[Callable[[Any], Awaitable[Any]]] = None,
    state_store: Optional[RunStateStore] = None,
    session_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    timeout_seconds: Optional[float] = None
) -> Any:
    """
    Run an agent in a durable DBOS workflow.
    
    This workflow automatically saves state on interruptions and can be resumed
    if the process crashes or restarts.
    
    Args:
        agent: The agent to run
        input_items: Input items or RunState for the agent
        session: Session for managing conversation state
        stream: Whether to stream the results
        result_processor: Optional function to process the result
        state_store: Optional RunStateStore instance. If not provided, one will be created
        session_id: Optional session ID for state persistence
        workflow_id: Optional workflow ID for idempotency
        timeout_seconds: Optional timeout for the workflow
    
    Returns:
        The result from run_agent
    """
    # Ensure DBOS is configured and launched
    ensure_dbos_launched()
    
    # Set workflow ID if provided
    if workflow_id:
        with SetWorkflowID(workflow_id):
            return await _run_agent_workflow_impl(
                agent, input_items, session, stream, result_processor,
                state_store, session_id, timeout_seconds
            )
    else:
        return await _run_agent_workflow_impl(
            agent, input_items, session, stream, result_processor,
            state_store, session_id, timeout_seconds
        )


async def _run_agent_workflow_impl(
    agent: Agent,
    input_items: list[TResponseInputItem] | RunState,
    session: SessionABC,
    stream: bool,
    result_processor: Optional[Callable[[Any], Awaitable[Any]]],
    state_store: Optional[RunStateStore],
    session_id: Optional[str],
    timeout_seconds: Optional[float]
) -> Any:
    """Internal implementation of run_agent_workflow."""
    # Create state store if not provided
    if state_store is None:
        if session_id is None:
            # Try to get session ID from session
            if hasattr(session, '_get_session_id'):
                session_id = await session._get_session_id()
            elif hasattr(session, 'get_session_id'):
                session_id = await session.get_session_id()
        
        state_store = RunStateStore(agent=agent, session_id=session_id)
    
    # Set timeout if provided
    if timeout_seconds:
        with SetWorkflowTimeout(timeout_seconds):
            return await _execute_agent_with_state_handling(
                agent, input_items, session, stream, result_processor, state_store
            )
    else:
        return await _execute_agent_with_state_handling(
            agent, input_items, session, stream, result_processor, state_store
        )


async def _execute_agent_with_state_handling(
    agent: Agent,
    input_items: list[TResponseInputItem] | RunState,
    session: SessionABC,
    stream: bool,
    result_processor: Optional[Callable[[Any], Awaitable[Any]]],
    state_store: RunStateStore
) -> Any:
    """Execute agent and handle state persistence."""
    # Step 1: Run agent (non-deterministic, must be a step)
    result = await _run_agent_step(agent, input_items, session, stream, result_processor)
    
    # Step 2: Handle interruptions and save state if needed
    if hasattr(result, 'interruptions') and result.interruptions:
        await _save_state_step(result, state_store)
    
    return result


def queue_agent_workflow(
    agent: Agent,
    input_items: list[TResponseInputItem] | RunState,
    session: SessionABC,
    stream: bool = False,
    result_processor: Optional[Callable[[Any], Awaitable[Any]]] = None,
    state_store: Optional[RunStateStore] = None,
    session_id: Optional[str] = None,
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
        agent: The agent to run
        input_items: Input items or RunState for the agent
        session: Session for managing conversation state
        stream: Whether to stream the results
        result_processor: Optional function to process the result
        state_store: Optional RunStateStore instance
        session_id: Optional session ID for state persistence
        queue_name: Optional queue name. Defaults to "timestep_agent_queue"
        workflow_id: Optional workflow ID for idempotency
        timeout_seconds: Optional timeout for the workflow
        priority: Optional priority (lower number = higher priority)
        deduplication_id: Optional deduplication ID to prevent duplicate runs
    
    Returns:
        WorkflowHandle that can be used to get the result via handle.get_result()
    """
    ensure_dbos_launched()
    
    # Get queue
    if queue_name:
        queue = Queue(queue_name)
    else:
        queue = _get_default_queue()
    
    # Enqueue options
    from dbos import SetEnqueueOptions
    
    enqueue_options = {}
    if priority is not None:
        enqueue_options["priority"] = priority
    if deduplication_id:
        enqueue_options["deduplication_id"] = deduplication_id
    
    # Enqueue the workflow directly (run_agent_workflow is already a registered workflow)
    # Build context managers for options
    from contextlib import ExitStack
    
    with ExitStack() as stack:
        if workflow_id:
            stack.enter_context(SetWorkflowID(workflow_id))
        if timeout_seconds:
            stack.enter_context(SetWorkflowTimeout(timeout_seconds))
        if enqueue_options:
            stack.enter_context(SetEnqueueOptions(**enqueue_options))
        
        # Note: workflow_id and timeout_seconds are handled by context managers above
        # Pass None for those parameters since they're set via context
        handle = queue.enqueue(
            run_agent_workflow,
            agent, input_items, session, stream, result_processor,
            state_store, session_id, None, None
        )
    
    return handle


def create_scheduled_agent_workflow(
    crontab: str,
    agent: Agent,
    input_items: list[TResponseInputItem] | RunState,
    session: SessionABC,
    stream: bool = False,
    result_processor: Optional[Callable[[Any], Awaitable[Any]]] = None,
    state_store: Optional[RunStateStore] = None,
    session_id: Optional[str] = None
) -> None:
    """
    Create a scheduled workflow that runs an agent periodically.
    
    This function registers a scheduled workflow with DBOS. The workflow will
    run automatically according to the crontab schedule.
    
    Example:
        create_scheduled_agent_workflow(
            "0 */6 * * *",  # Every 6 hours
            agent,
            input_items,
            session
        )
    
    Args:
        crontab: Crontab schedule (e.g., "0 */6 * * *" for every 6 hours)
        agent: The agent to run
        input_items: Input items or RunState for the agent
        session: Session for managing conversation state
        stream: Whether to stream the results
        result_processor: Optional function to process the result
        state_store: Optional RunStateStore instance
        session_id: Optional session ID for state persistence
    """
    ensure_dbos_launched()
    
    @DBOS.scheduled(crontab)
    @DBOS.workflow()
    async def _scheduled_workflow(scheduled_time: Any, actual_time: Any):
        return await _run_agent_workflow_impl(
            agent, input_items, session, stream, result_processor,
            state_store, session_id, None
        )

