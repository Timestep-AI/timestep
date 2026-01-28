"""Helper functions for handling A2A client events."""

from typing import Any, Union, Tuple, Dict, Optional

# Import a2a types - Task may not always be directly importable
try:
    from a2a.types import TaskStatusUpdateEvent, Task, Message, TaskState
except ImportError:
    # Fallback if Task is not directly importable
    Task = Any
    TaskStatusUpdateEvent = Any
    Message = Any
    TaskState = Any


def extract_event_data(event: Union[Tuple[Any, Any], Any]) -> Any:
    """Extract event data from a client event.
    
    Client.send_message() may return either:
    - A tuple of (Task, TaskStatusUpdateEvent) where Task is the full task object
    - A direct event object (TaskStatusUpdateEvent, Task, etc.)
    
    This function normalizes both cases to return the event data.
    For tuples, it returns the TaskStatusUpdateEvent (second element).
    
    Args:
        event: Either a tuple (Task, TaskStatusUpdateEvent) or a direct event object
        
    Returns:
        The event data object (TaskStatusUpdateEvent, Task, or similar)
    """
    if isinstance(event, tuple):
        # Unpack tuple - typically (Task, TaskStatusUpdateEvent)
        # Return the TaskStatusUpdateEvent (second element) for event processing
        if len(event) >= 2:
            return event[1]  # TaskStatusUpdateEvent
        else:
            return event[0]  # Fallback to first element
    else:
        # Direct event object
        return event


def extract_task_from_tuple(event: Union[Tuple[Any, Any], Any]) -> Any:
    """Extract the Task object from a client event tuple.
    
    Client.send_message() returns tuples as (Task, TaskStatusUpdateEvent).
    This function extracts the Task object (first element) which contains
    the full task state including history.
    
    Args:
        event: Either a tuple (Task, TaskStatusUpdateEvent) or a direct event object
        
    Returns:
        The Task object (first element of tuple) or None if not a tuple
    """
    if isinstance(event, tuple) and len(event) >= 1:
        return event[0]  # Task object
    return None


def extract_task_from_event(event: Union[Tuple[Any, Any], Any]) -> Any:
    """Extract task from a client event.
    
    Args:
        event: Either a tuple (event_type, event_data) or a direct event object
        
    Returns:
        Task object extracted from the event, or None if not found
    """
    event_data = extract_event_data(event)
    
    # Handle different event structures
    if hasattr(event_data, 'model_dump'):
        event_dict = event_data.model_dump()
    elif hasattr(event_data, '__dict__'):
        event_dict = dict(event_data)
    elif isinstance(event_data, dict):
        event_dict = event_data
    else:
        event_dict = {}
    
    # Try to extract task from various possible structures
    # Check if event_data itself is a task-like object
    is_task_like = (
        isinstance(event_data, dict) 
        or hasattr(event_data, 'id') 
        or hasattr(event_data, 'status')
        or hasattr(event_data, 'task_id')
    )
    
    task_data = (
        event_dict.get('result') 
        or event_dict.get('task')
        or getattr(event_data, 'result', None)
        or getattr(event_data, 'task', None)
        or (event_data if is_task_like else None)
    )
    
    return task_data


def get_agui_event_type_from_task_status_update(event: TaskStatusUpdateEvent) -> str:
    """Determine AG-UI event type from TaskStatusUpdateEvent.
    
    Args:
        event: TaskStatusUpdateEvent object (can be dict or object)
        
    Returns:
        AG-UI event type string (e.g., "RunStartedEvent", "StepFinishedEvent")
    """
    # Extract state from event
    if isinstance(event, dict):
        status = event.get("status", {})
        if isinstance(status, dict):
            state = status.get("state", {})
            if isinstance(state, dict):
                state_value = state.get("value")
            else:
                state_value = state
        else:
            state_value = getattr(status, "state", None) if hasattr(status, "state") else None
            if state_value:
                if hasattr(state_value, "value"):
                    state_value = state_value.value
    else:
        status = getattr(event, "status", None)
        if status:
            state = getattr(status, "state", None) if hasattr(status, "state") else None
            if state:
                if hasattr(state, "value"):
                    state_value = state.value
                else:
                    state_value = state
            else:
                state_value = None
        else:
            state_value = None
    
    # Map A2A task states to AG-UI event types
    if state_value == "created":
        return "RunStartedEvent"
    elif state_value == "working":
        return "StepStartedEvent"
    elif state_value == "input-required":
        return "StepFinishedEvent"
    elif state_value == "completed":
        return "RunFinishedEvent"
    elif state_value in ["failed", "canceled", "rejected"]:
        return "RunErrorEvent"
    
    # Default fallback
    return "RunStartedEvent"


def convert_task_status_update_to_agui_event(event: TaskStatusUpdateEvent, task: Task) -> Dict[str, Any]:
    """Convert TaskStatusUpdateEvent to AG-UI event.
    
    Args:
        event: TaskStatusUpdateEvent object (can be dict or object)
        task: Task object (can be dict or object) for additional context
        
    Returns:
        AG-UI event dictionary
    """
    event_type = get_agui_event_type_from_task_status_update(event)
    
    # Extract task information
    if isinstance(task, dict):
        task_id = task.get("id")
        context_id = task.get("context_id")
    else:
        task_id = getattr(task, "id", None)
        context_id = getattr(task, "context_id", None)
    
    # Build AG-UI event based on type
    if event_type == "RunStartedEvent":
        return {
            "type": "run-started",
            "runId": task_id or "",
        }
    elif event_type == "StepStartedEvent":
        return {
            "type": "step-started",
            "runId": task_id or "",
        }
    elif event_type == "StepFinishedEvent":
        return {
            "type": "step-finished",
            "runId": task_id or "",
        }
    elif event_type == "RunFinishedEvent":
        return {
            "type": "run-finished",
            "runId": task_id or "",
        }
    elif event_type == "RunErrorEvent":
        return {
            "type": "run-error",
            "runId": task_id or "",
        }
    
    # Default fallback
    return {
        "type": "run-started",
        "runId": task_id or "",
    }


def add_canonical_type_to_message(message: Message, agui_type: str) -> Message:
    """Add canonical_type to message metadata.
    
    Args:
        message: A2A Message object (can be dict or object)
        agui_type: AG-UI event type string (e.g., "TextMessageContentEvent")
        
    Returns:
        Message with canonical_type added to metadata
    """
    if isinstance(message, dict):
        if "metadata" not in message:
            message["metadata"] = {}
        if not isinstance(message["metadata"], dict):
            message["metadata"] = {}
        message["metadata"]["canonical_type"] = agui_type
    else:
        if not hasattr(message, "metadata") or message.metadata is None:
            message.metadata = {}
        if not isinstance(message.metadata, dict):
            message.metadata = {}
        message.metadata["canonical_type"] = agui_type
    
    return message
