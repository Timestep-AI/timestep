"""A2A AgentExecutor implementation for Timestep agent."""

import json
import uuid
from typing import Any

try:
    from typing import override  # type: ignore[attr-defined]
except ImportError:
    # Python < 3.12 doesn't have override, use a no-op decorator
    from collections.abc import Callable
    from typing import TypeVar
    T = TypeVar('T')
    def override(func: Callable[..., T]) -> Callable[..., T]:
        return func

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task
from openai.types.chat import ChatCompletionMessageParam

from ..core import AgentEventEmitter, GetWeather, WebSearch, run_agent
from .message_converter import a2a_to_openai, extract_text_from_parts
from .postgres_task_store import PostgresTaskStore


class TimestepAgentExecutor(AgentExecutor):
    """AgentExecutor that wraps Timestep's run_agent function."""

    def __init__(
        self,
        tools: list[type] | None = None,
        model: str = "gpt-4.1",
        task_store: PostgresTaskStore | None = None,
    ):
        """Initialize the TimestepAgentExecutor.

        Args:
            tools: List of tool classes to use. If None, uses default tools (GetWeather, WebSearch).
            model: OpenAI model name to use.
            task_store: Optional PostgresTaskStore for saving OpenAI messages.
        """
        self.tools = tools if tools is not None else [GetWeather, WebSearch]
        self.model = model
        self.task_store = task_store
        self._pending_approvals: dict[str, dict[str, Any]] = {}  # Maps approval_key to {tool_call, resolve}

    def _convert_message_to_dict(self, msg: Any) -> dict[str, Any]:
        """Convert a message object to dictionary format."""
        if hasattr(msg, "model_dump"):
            result = msg.model_dump()
            if isinstance(result, dict):
                return result
        if hasattr(msg, "dict"):
            result = msg.dict()
            if isinstance(result, dict):
                return result
        if isinstance(msg, dict):
            return msg
        return {
            "role": getattr(msg, "role", "user"),
            "parts": getattr(msg, "parts", []),
        }

    def _build_task_history(self, task: Any, context: RequestContext) -> list[dict[str, Any]]:
        """Build task history from task and context."""
        task_history: list[dict[str, Any]] = []
        if hasattr(task, "history") and task.history:
            for msg in task.history:
                task_history.append(self._convert_message_to_dict(msg))

        if context.message:
            current_msg_dict = self._convert_message_to_dict(context.message)
            if not any(
                msg.get("messageId") == current_msg_dict.get("messageId")
                for msg in task_history
            ):
                task_history.append(current_msg_dict)

        return task_history

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent task.

        Args:
            context: Request context containing message and task information.
            event_queue: Event queue for publishing A2A events.
        """
        user_message = context.message
        existing_task = context.current_task

        # Extract user input first to check if it's an approval response
        user_input = context.get_user_input() if user_message else None

        # Check if this is an approval response (BEFORE determining taskId/contextId)
        # Approval responses are simple "approve" or "reject" messages
        is_approval_response = user_input and (
            user_input.lower().strip() == "approve"
            or user_input.lower().strip() == "reject"
        )

        # If it's an approval response, try to find a pending approval resolver
        if is_approval_response and user_message:
            message_context_id = getattr(user_message, "context_id", None)
            message_task_id = getattr(user_message, "task_id", None)

            approved = user_input.lower().strip() == "approve"
            
            # Try to find pending approval - first exact match, then by contextId
            approval_key: str | None = None
            if message_context_id and message_task_id:
                approval_key = f"{message_context_id}:{message_task_id}"
                if approval_key not in self._pending_approvals:
                    approval_key = None
            
            # If exact match failed, try to find any pending approval for this context
            if not approval_key and message_context_id:
                for key in self._pending_approvals.keys():
                    if key.startswith(f"{message_context_id}:"):
                        approval_key = key
                        break
            
            if approval_key:
                pending_approval = self._pending_approvals.get(approval_key)
                if pending_approval:
                    print("Resolving approval", {"approval_key": approval_key, "approved": approved})
                    # Delete BEFORE resolving to avoid race conditions
                    self._pending_approvals.pop(approval_key, None)
                    try:
                        pending_approval["resolve"](approved)
                        print("Approval resolved successfully")
                    except Exception as e:
                        print(f"Error resolving approval: {e}")
                    return  # Don't process as a new message - original agent execution will continue
            
            # If no approval found, log warning and return
            print(
                "Approval response received but no pending approval found",
                {
                    "message_context_id": message_context_id,
                    "message_task_id": message_task_id,
                    "pending_keys": list(self._pending_approvals.keys()),
                },
            )
            return  # Don't process as a new message

        # 1. Task Creation
        # Use taskId from message if provided (for approval responses), otherwise use existing task or create new
        if user_message:
            # Get task_id and context_id from message, but ensure they're strings (not MagicMock objects)
            msg_task_id = getattr(user_message, "task_id", None)
            msg_context_id = getattr(user_message, "context_id", None)
            
            # Convert to string if not None and not already a string (handles MagicMock in tests)
            if msg_task_id is not None and isinstance(msg_task_id, str):
                task_id = msg_task_id
            elif existing_task:
                task_id = existing_task.id
            else:
                task_id = str(uuid.uuid4())
            
            if msg_context_id is not None and isinstance(msg_context_id, str):
                context_id = msg_context_id
            elif existing_task:
                context_id = existing_task.context_id
            else:
                context_id = str(uuid.uuid4())
        elif existing_task:
            task_id = existing_task.id
            context_id = existing_task.context_id
        else:
            raise ValueError("No message or task provided in context")

        if not user_input:
            failure_update = TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message(
                        "No user input found in context.",
                        context_id,
                        task_id,
                    ),
                ),
                final=True,
            )
            await event_queue.enqueue_event(failure_update)
            return

        # Only create new task if this is not an approval response
        if not existing_task and user_message:
            initial_task = new_task(user_message)
            await event_queue.enqueue_event(initial_task)

        # 2. Working State
        working_status_update = TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            status=TaskStatus(
                state=TaskState.working,
                message=new_agent_text_message(
                    "Processing your request...",
                    context_id,
                    task_id,
                ),
            ),
            final=False,
        )
        await event_queue.enqueue_event(working_status_update)

        # 4. History Conversion
        # Load full conversation history from context (all previous tasks)
        openai_messages: list[dict[str, Any]] = []

        if self.task_store:
            try:
                # Get all OpenAI messages from previous tasks in this context
                context_messages = (
                    await self.task_store.get_openai_messages_by_context_id(context_id)
                )
                openai_messages = list(context_messages)
                print(
                    "Loaded context history",
                    {"context_id": context_id, "message_count": len(openai_messages)},
                )
            except Exception as error:
                print(
                    "Error loading context history, falling back to task history",
                    {"error": error, "context_id": context_id},
                )
                # Fallback to task history if loading context messages fails
                task_history = self._build_task_history(existing_task, context)
                openai_messages = a2a_to_openai(task_history)
        else:
            # Fallback: use task history if no taskStore available
            task_history = self._build_task_history(existing_task, context)
            openai_messages = a2a_to_openai(task_history)

        # Add the current user message if it's not an approval response
        if user_message and not is_approval_response:
            message_text = extract_text_from_parts(
                getattr(user_message, "parts", [])
            )
            if message_text:
                openai_messages.append({"role": "user", "content": message_text})
                print(
                    "Added current user message to history",
                    {"message_text": message_text[:50]},
                )

        # Add system message if not present
        if not any(msg.get("role") == "system" for msg in openai_messages):
            openai_messages.insert(
                0, {"role": "system", "content": "You are a helpful AI assistant."}
            )

        # 5. Agent Execution with streaming and tool approval
        accumulated_content = ""
        streaming_events: list[TaskStatusUpdateEvent] = []
        tool_result_events: list[TaskStatusUpdateEvent] = []

        # Create event emitter for agent events
        event_emitter = AgentEventEmitter()

        # Handle delta events (streaming updates)
        def handle_delta(delta: dict[str, Any]) -> None:
            """Handle delta events."""
            nonlocal accumulated_content
            if "content" in delta:
                accumulated_content += delta["content"]
                # Create event for this chunk (will be published later)
                event = TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.working,
                        message=new_agent_text_message(
                            accumulated_content,
                            context_id,
                            task_id,
                        ),
                    ),
                    final=False,
                )
                streaming_events.append(event)

        event_emitter.on("delta", handle_delta)

        # Handle tool approval required events
        async def handle_tool_approval_required(event: dict[str, Any]) -> None:
            """Handle tool approval required event."""
            tool_call = event["tool_call"]
            resolve = event["resolve"]
            
            # ChatCompletionMessageToolCall can be function or custom, check type
            tool_name = (
                tool_call.get("function", {}).get("name", "unknown")
                if "function" in tool_call
                else "unknown"
            )
            tool_args = (
                tool_call.get("function", {}).get("arguments", "{}")
                if "function" in tool_call
                else "{}"
            )

            tool_args_dict: dict[str, Any] = {}
            try:
                tool_args_dict = (
                    json.loads(tool_args)
                    if isinstance(tool_args, str)
                    else tool_args
                )
            except json.JSONDecodeError:
                tool_args_dict = {}

            approval_message = (
                f"Tool call requires approval:\n"
                f"Tool: {tool_name}\n"
                f"Arguments: {json.dumps(tool_args_dict, indent=2)}"
            )

            approval_update = TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.input_required,
                    message=new_agent_text_message(
                        approval_message,
                        context_id,
                        task_id,
                    ),
                ),
                final=False,  # Don't close the stream - we need to continue after approval
            )
            await event_queue.enqueue_event(approval_update)

            # Store approval request with resolver - use simple key that frontend can match
            approval_key = f"{context_id}:{task_id}"
            print(
                "Tool approval required",
                {
                    "approval_key": approval_key,
                    "tool_name": tool_name,
                    "task_id": task_id,
                    "context_id": context_id,
                },
            )

            # Store the resolver - will be resolved when we get approval response
            self._pending_approvals[approval_key] = {
                "tool_call": tool_call,
                "resolve": resolve,
            }
            print(
                "Waiting for approval",
                {
                    "approval_key": approval_key,
                    "pending_count": len(self._pending_approvals),
                },
            )

        event_emitter.on("tool-approval-required", handle_tool_approval_required)

        # Handle tool result events
        def handle_tool_result(event: dict[str, Any]) -> None:
            """Handle tool result event."""
            tool_call_id = event["tool_call_id"]
            tool_name = event["tool_name"]
            result = event["result"]
            
            print(
                "[tool-result] Called",
                {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "result_length": len(result),
                    "task_id": task_id,
                    "context_id": context_id,
                },
            )
            tool_result_event = TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=new_agent_text_message(
                        f"Tool {tool_name} executed. Result: {result}",
                        context_id,
                        task_id,
                    ),
                ),
                final=False,
            )
            print("Publishing tool result event", {"task_id": task_id, "context_id": context_id})
            # Add to list to be published later (can't await in sync callback)
            tool_result_events.append(tool_result_event)
            print("Tool result event added to list")

        event_emitter.on("tool-result", handle_tool_result)

        # Handle child message events from handoff
        def handle_child_message(event: dict[str, Any]) -> None:
            """Handle child message event from handoff."""
            message = event["message"]
            # Convert child A2A message to OpenAI format and add to parent's messages
            text_parts = extract_text_from_parts(message.get("parts", []))
            
            if message.get("role") == "agent":
                # Don't save "Processing your request..." messages
                if text_parts == "Processing your request...":
                    print('[on_child_message] Skipping "Processing your request..." message')
                    return
                
                # Child agent message - check if it has tool_calls
                msg_with_tool_calls = message
                if msg_with_tool_calls.get("tool_calls") or msg_with_tool_calls.get("toolCalls"):
                    tool_calls = msg_with_tool_calls.get("tool_calls") or msg_with_tool_calls.get("toolCalls")
                    # Convert to OpenAI format and add to parent's messages
                    openai_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": text_parts if text_parts else None,
                        "tool_calls": tool_calls,
                    }
                    openai_messages.append(openai_message)
                    print(
                        "[on_child_message] Added child assistant message with tool calls to parent messages",
                        {
                            "tool_calls_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
                            "total_messages": len(openai_messages),
                        },
                    )
                    
                    # Save immediately
                    if self.task_store:
                        asyncio.create_task(
                            self.task_store.save_openai_messages(task_id, openai_messages)
                        )
                elif text_parts and text_parts.strip():
                    # Regular assistant message without tool calls
                    openai_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": text_parts,
                    }
                    openai_messages.append(openai_message)
                    print("[on_child_message] Added child assistant message to parent messages")
                    
                    # Save immediately
                    if self.task_store:
                        asyncio.create_task(
                            self.task_store.save_openai_messages(task_id, openai_messages)
                        )
            elif message.get("role") == "tool" and message.get("toolName"):
                # Child tool message - find the corresponding tool_call_id from the previous assistant message
                tool_call_id: str | None = None
                for i in range(len(openai_messages) - 1, -1, -1):
                    prev_msg = openai_messages[i]
                    if prev_msg.get("role") == "assistant":
                        if prev_msg.get("tool_calls"):
                            matching_tool_call = next(
                                (
                                    tc
                                    for tc in prev_msg["tool_calls"]
                                    if isinstance(tc, dict)
                                    and tc.get("function", {}).get("name") == message.get("toolName")
                                ),
                                None,
                            )
                            if matching_tool_call:
                                tool_call_id = matching_tool_call.get("id")
                                break
                
                if tool_call_id:
                    # Parse content - it might be a JSON string
                    content: str = text_parts
                    try:
                        if content.startswith('"') and content.endswith('"'):
                            content = json.loads(content)
                        elif content.startswith("{") or content.startswith("["):
                            content = json.loads(content)
                    except json.JSONDecodeError:
                        # Keep as string if parsing fails
                        pass
                    
                    openai_message: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,
                    }
                    openai_messages.append(openai_message)
                    print(
                        "[on_child_message] Added child tool message to parent messages",
                        {"tool_name": message.get("toolName"), "tool_call_id": tool_call_id},
                    )
                    
                    # Save immediately
                    if self.task_store:
                        asyncio.create_task(
                            self.task_store.save_openai_messages(task_id, openai_messages)
                        )
                else:
                    print(
                        "[on_child_message] Could not find tool_call_id for child tool message",
                        {"tool_name": message.get("toolName")},
                    )
            
            # Also forward via tool-result event for UI display
            if text_parts:
                child_tool_name = (
                    f"child:tool:{message.get('toolName')}"
                    if message.get("role") == "tool" and message.get("toolName")
                    else f"child:{message.get('role')}"
                )
                event_emitter.emit("tool-result", {
                    "tool_call_id": message.get("messageId", ""),
                    "tool_name": child_tool_name,
                    "result": text_parts,
                })

        event_emitter.on("child-message", handle_child_message)

        # Handle assistant message events
        async def handle_assistant_message(event: dict[str, Any]) -> None:
            """Handle assistant message event."""
            message = event["message"]
            # Don't save "Processing your request..." messages - they're not real messages
            content = message.get("content", "")
            if content == "Processing your request...":
                print('Skipping save for "Processing your request..." message')
                return

            if self.task_store:
                try:
                    # Save all messages up to and including this assistant message
                    await self.task_store.save_openai_messages(task_id, openai_messages)
                    print(
                        "Saved assistant message immediately",
                        {"task_id": task_id, "message_count": len(openai_messages)},
                    )
                except Exception as error:
                    print("Error saving assistant message", {"error": error, "task_id": task_id})
                    # Don't fail - continue execution

        event_emitter.on("assistant-message", handle_assistant_message)

        try:
            print(
                "Starting run_agent",
                {
                    "task_id": task_id,
                    "context_id": context_id,
                    "message_count": len(openai_messages),
                },
            )
            # Type cast to satisfy mypy - the dicts are compatible with ChatCompletionMessageParam
            typed_messages: list[Any] = openai_messages
            # Run the agent with event emitter
            final_response = await run_agent(
                typed_messages,
                tools=self.tools,
                model=self.model,
                event_emitter=event_emitter,  # Pass event emitter instead of callbacks
                context_id=context_id,  # Pass context_id for handoff tool (future support)
            )
            print(
                "run_agent completed",
                {
                    "task_id": task_id,
                    "context_id": context_id,
                    "response_length": len(final_response) if final_response else 0,
                },
            )

            # Publish streaming events (publish every 5th event to avoid too many updates)
            # But also publish the last event if there are any
            if streaming_events:
                for i, event in enumerate(streaming_events):
                    if i % 5 == 0 or i == len(streaming_events) - 1:
                        await event_queue.enqueue_event(event)

            # Publish tool result events
            for event in tool_result_events:
                await event_queue.enqueue_event(event)

            # 8. Completion - Publish final response
            # If we have accumulated content from streaming, use that; otherwise use finalResponse
            final_text = accumulated_content or final_response or ""
            print(
                "Preparing final response",
                {
                    "task_id": task_id,
                    "context_id": context_id,
                    "text_length": len(final_text),
                    "accumulated_length": len(accumulated_content),
                    "final_response_length": len(final_response) if final_response else 0,
                },
            )
            final_update = TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=new_agent_text_message(
                        final_text,
                        context_id,
                        task_id,
                    ),
                ),
                final=True,
            )
            print(
                "Publishing final response",
                {
                    "task_id": task_id,
                    "context_id": context_id,
                    "text_length": len(final_text),
                    "event_kind": final_update.kind,
                },
            )
            await event_queue.enqueue_event(final_update)
            print("Final response event published")

            # Save OpenAI messages after execution completes
            # The messages array is modified in place by run_agent (tool messages are appended)
            if self.task_store:
                try:
                    await self.task_store.save_openai_messages(task_id, typed_messages)
                    print(
                        "Saved OpenAI messages",
                        {"task_id": task_id, "message_count": len(typed_messages)},
                    )
                except Exception as save_error:
                    print(
                        "Error saving OpenAI messages",
                        {"error": save_error, "task_id": task_id},
                    )
                    # Don't fail the execution if saving messages fails

        except Exception as e:
            # Publish error status
            print("Error in agent execution", {"error": e, "task_id": task_id, "context_id": context_id})
            error_update = TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message(
                        f"Error: {e!s}",
                        context_id,
                        task_id,
                    ),
                ),
                final=True,
            )
            await event_queue.enqueue_event(error_update)
            # Don't rethrow - let the execution complete gracefully

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel the current task.

        Args:
            context: Request context.
            event_queue: Event queue for publishing cancellation event.
        """
        task = context.current_task
        if task:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(state=TaskState.canceled),
                    final=True,
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )

