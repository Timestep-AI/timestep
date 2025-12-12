"""Event emitter for agent execution events."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from openai.types.chat import ChatCompletionMessageParam


class AgentDeltaEvent(TypedDict, total=False):
    """Delta event for streaming updates."""
    content: str
    tool_calls: list[Any]


class ToolApprovalRequiredEvent(TypedDict):
    """Event for tool approval requirement."""
    tool_call: dict[str, Any]
    resolve: Callable[[bool], None]


class ToolResultEvent(TypedDict):
    """Event for tool execution result."""
    tool_call_id: str
    tool_name: str
    result: str


class AssistantMessageEvent(TypedDict):
    """Event for assistant message."""
    message: ChatCompletionMessageParam


class ChildMessageEvent(TypedDict, total=False):
    """Event for child message from handoff."""
    kind: str
    role: str
    message_id: str
    parts: list[dict[str, Any]]
    context_id: str
    task_id: str | None
    tool_name: str | None
    tool_calls: list[Any] | None


class AgentEventEmitter:
    """Event emitter for agent execution events."""

    def __init__(self) -> None:
        """Initialize the event emitter."""
        self._listeners: dict[str, list[Callable[..., Any]]] = {}

    def on(
        self, event: str, listener: Callable[..., Any]
    ) -> "AgentEventEmitter":
        """Register an event listener.

        Args:
            event: Event name
            listener: Listener function

        Returns:
            Self for chaining
        """
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)
        return self

    def off(self, event: str, listener: Callable[..., Any]) -> "AgentEventEmitter":
        """Remove an event listener.

        Args:
            event: Event name
            listener: Listener function to remove

        Returns:
            Self for chaining
        """
        if event in self._listeners:
            self._listeners[event].remove(listener)
        return self

    def emit(self, event: str, *args: Any, **kwargs: Any) -> Any:
        """Emit an event.

        Args:
            event: Event name
            *args: Positional arguments for listeners
            **kwargs: Keyword arguments for listeners

        Returns:
            Result from listeners (if any)
        """
        if event not in self._listeners:
            return None

        results = []
        for listener in self._listeners[event]:
            try:
                result = listener(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    # For async listeners, we'll need to handle them differently
                    # For now, just collect them
                    results.append(result)
                else:
                    results.append(result)
            except Exception as e:
                print(f"Error in event listener for {event}: {e}")

        return results[0] if len(results) == 1 else results

    async def emit_async(self, event: str, *args: Any, **kwargs: Any) -> Any:
        """Emit an event and await async listeners.

        Args:
            event: Event name
            *args: Positional arguments for listeners
            **kwargs: Keyword arguments for listeners

        Returns:
            Result from listeners (if any)
        """
        if event not in self._listeners:
            return None

        results = []
        for listener in self._listeners[event]:
            try:
                result = listener(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                results.append(result)
            except Exception as e:
                print(f"Error in async event listener for {event}: {e}")

        return results[0] if len(results) == 1 else results

