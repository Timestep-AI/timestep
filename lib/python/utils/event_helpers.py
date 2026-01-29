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
