"""Loop class - AgentExecutor that uses MCP client to get system prompt and tools."""

import json
import logging
import os
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskStatusUpdateEvent,
    TaskStatus,
    TaskState,
    Role,
    Part,
    DataPart,
    Message,
)
from a2a.client.helpers import create_text_message_object
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

from timestep.utils.message_helpers import (
    extract_user_text_and_tool_results,
    convert_mcp_tool_to_openai,
    convert_openai_tool_call_to_mcp,
    TOOL_CALLS_KEY,
    TOOL_RESULTS_KEY,
)


def _convert_a2a_message_to_openai(message: Message) -> Optional[Dict[str, Any]]:
    """Convert an A2A message from task history to OpenAI format.
    
    Returns:
        OpenAI message dict (user, assistant, or tool) or None if message is empty/invalid
    """
    if not message:
        return None
    
    # Handle both dict and object access for message
    if isinstance(message, dict):
        message_parts = message.get("parts", [])
        message_role = message.get("role")
    else:
        message_parts = getattr(message, "parts", []) if hasattr(message, "parts") else []
        message_role = getattr(message, "role", None) if hasattr(message, "role") else None
    
    if not message_parts:
        return None
    
    # Extract text content
    text_content = ""
    tool_calls = []
    tool_results = []
    
    for part in message_parts:
        part_data = part.root if hasattr(part, "root") else part
        # Handle both dict and object access for part_data
        if isinstance(part_data, dict):
            part_kind = part_data.get("kind")
            if part_kind == "text":
                text_content += part_data.get("text", "") or ""
            elif part_kind == "data":
                part_data_dict = part_data.get("data")
                if isinstance(part_data_dict, dict):
                    if TOOL_CALLS_KEY in part_data_dict:
                        calls = part_data_dict[TOOL_CALLS_KEY]
                        if isinstance(calls, list):
                            tool_calls.extend(calls)
                    if TOOL_RESULTS_KEY in part_data_dict:
                        results = part_data_dict[TOOL_RESULTS_KEY]
                        if isinstance(results, list):
                            tool_results.extend(results)
        elif hasattr(part_data, "kind"):
            if part_data.kind == "text" and hasattr(part_data, "text"):
                text_content += part_data.text or ""
            elif part_data.kind == "data" and hasattr(part_data, "data"):
                if isinstance(part_data.data, dict):
                    if TOOL_CALLS_KEY in part_data.data:
                        calls = part_data.data[TOOL_CALLS_KEY]
                        if isinstance(calls, list):
                            tool_calls.extend(calls)
                    if TOOL_RESULTS_KEY in part_data.data:
                        results = part_data.data[TOOL_RESULTS_KEY]
                        if isinstance(results, list):
                            tool_results.extend(results)
    
    # Determine role from message role
    if message_role is None:
        role = Role.user
    else:
        role = message_role
    
    # Convert based on role and content
    if role == Role.user:
        # User message - can have text or tool results
        if tool_results:
            # This is a tool results message - convert to tool messages
            # (This shouldn't happen in history, but handle it)
            return None  # Tool results are handled separately
        if text_content:
            return {"role": "user", "content": text_content}
    elif role == Role.agent:
        # Assistant message - can have text and/or tool calls
        openai_msg: Dict[str, Any] = {"role": "assistant", "content": text_content or ""}
        if tool_calls:
            # Convert MCP tool calls to OpenAI format
            openai_tool_calls = []
            for mcp_tc in tool_calls:
                call_id = mcp_tc.get("call_id", "") if isinstance(mcp_tc, dict) else getattr(mcp_tc, "call_id", "")
                name = mcp_tc.get("name", "") if isinstance(mcp_tc, dict) else getattr(mcp_tc, "name", "")
                arguments = mcp_tc.get("arguments", {}) if isinstance(mcp_tc, dict) else getattr(mcp_tc, "arguments", {})
                openai_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                    }
                })
            openai_msg["tool_calls"] = openai_tool_calls
        return openai_msg
    
    return None


class Loop(AgentExecutor):
    """Loop is an AgentExecutor that uses MCP client to get system prompt and tools from Environment.
    
    The Loop:
    1. Receives A2A RequestContext with context_id
    2. Uses MCP client to locate Environment by context_id
    3. Gets system prompt (FastMCP prompt) and tools from Environment
    4. Invokes OpenAI model (async/streaming)
    5. Emits tool calls in A2A format (client handles execution)
    6. Returns A2A Task
    """
    
    def __init__(
        self,
        agent_id: str,
        model: str,
        context_id_to_environment_uri: Dict[str, str],
        human_in_loop: bool = False,
    ):
        self.agent_id = agent_id
        self.model = model
        self.context_id_to_environment_uri = context_id_to_environment_uri
        self.human_in_loop = human_in_loop
        
        # Initialize AsyncOpenAI client
        self.openai_client = AsyncOpenAI()
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute agent task using OpenAI and MCP client for system prompt/tools."""
        task_id = context.task_id
        context_id = context.context_id
        
        # Get Environment URI from context_id - fail fast if not found
        environment_uri = self.context_id_to_environment_uri.get(context_id)
        
        # Single-environment mode: if only one environment is configured,
        # use it for any context_id (explicit configuration, not fallback)
        if not environment_uri and len(self.context_id_to_environment_uri) == 1:
            environment_uri = next(iter(self.context_id_to_environment_uri.values()))
        
        # Fail fast if still not found
        if not environment_uri:
            available_context_ids = list(self.context_id_to_environment_uri.keys())
            raise ValueError(
                f"No environment found for context_id: {context_id}. "
                f"Available context_ids: {available_context_ids}. "
                f"If you only have one environment, it will be used for all context_ids."
            )
        
        # Get system prompt and tools from Environment via HTTP client
        # Fail fast if prompt not found or other errors occur
        async with streamable_http_client(environment_uri) as (read, write, _):
            async with ClientSession(read, write) as mcp_session:
                await mcp_session.initialize()
                
                # Get system prompt (FastMCP prompt) - fail fast if not found
                # Human-in-the-loop: MCP Elicitation can happen here
                system_prompt_result = await mcp_session.get_prompt("system_prompt", {
                    "agent_name": self.agent_id
                })
                if not system_prompt_result.messages or not system_prompt_result.messages[0].content.text:
                    raise ValueError(f"System prompt 'system_prompt' returned empty result from environment at {environment_uri}")
                system_prompt = system_prompt_result.messages[0].content.text
                
                # Get available tools
                tools_result = await mcp_session.list_tools()
                tools = [
                    convert_mcp_tool_to_openai(tool)
                    for tool in tools_result.tools
                ]
        
        # Extract user message and tool results from incoming message
        user_text, tool_results = extract_user_text_and_tool_results(context.message)
        
        # Build messages list - always start with task history
        messages = []
        
        # Get task from context and convert full history to OpenAI format
        task = context.current_task
        if tool_results and (not task or not task.history):
            logger.warning(
                f"Tool results provided but task history is missing. "
                f"task={task}, has_history={task.history if task else False}, "
                f"task_id={task_id}, context_id={context_id}"
            )
        if task and task.history:
            # Track which tool_call_id sets we've seen (to filter incremental updates)
            seen_tool_call_sets = set()
            # Track the last assistant message without tool_calls (to filter incremental text updates)
            last_assistant_text = None
            
            # Go through history in reverse to keep the last (final) version of each assistant message
            for hist_message in reversed(task.history):
                openai_msg = _convert_a2a_message_to_openai(hist_message)
                if openai_msg:
                    if openai_msg.get("role") == "assistant" and openai_msg.get("tool_calls"):
                        # Only keep the last (first in reverse) assistant message for each set of tool_call_ids
                        tool_calls = openai_msg.get("tool_calls", [])
                        tool_call_ids = tuple(sorted(tc.get("id") for tc in tool_calls if tc.get("id")))
                        if tool_call_ids not in seen_tool_call_sets:
                            messages.insert(0, openai_msg)
                            seen_tool_call_sets.add(tool_call_ids)
                    elif openai_msg.get("role") == "assistant":
                        # For assistant messages without tool_calls, only keep the last one (filter incremental text updates)
                        if last_assistant_text is None:
                            last_assistant_text = openai_msg
                    else:
                        # Keep all other messages (user, tool)
                        messages.insert(0, openai_msg)
            
            # Add the last assistant text message (if any) at the appropriate position
            if last_assistant_text:
                # Insert after the last user message or at the beginning
                insert_pos = 0
                for i, msg in enumerate(messages):
                    if msg.get("role") == "user":
                        insert_pos = i + 1
                messages.insert(insert_pos, last_assistant_text)
        
        # Add the current incoming message
        if tool_results:
            # Add tool results as tool messages
            # The assistant message with matching tool_calls should already be in messages from history processing
            for tool_result in tool_results:
                tool_call_id = tool_result.get("call_id")
                if not tool_call_id:
                    raise ValueError("tool_result missing call_id")
                raw_result = tool_result.get("output")
                if raw_result is None:
                    raise ValueError("tool_result missing output")
                if isinstance(raw_result, (dict, list)):
                    content = json.dumps(raw_result)
                else:
                    content = str(raw_result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                })
        elif user_text:
            # Add user message
            messages.append({"role": "user", "content": user_text})
        
        # Ensure we have at least one message before calling OpenAI
        if not messages:
            raise ValueError(f"No messages to send to OpenAI. Task: {task_id}, Context: {context_id}")
        
        # Validate message structure: if we have tool messages, ensure there's a preceding assistant message with tool_calls
        if messages and any(msg.get("role") == "tool" for msg in messages):
            # Find the first tool message
            first_tool_idx = next((i for i, msg in enumerate(messages) if msg.get("role") == "tool"), None)
            if first_tool_idx is not None:
                # Check if there's an assistant message with tool_calls before this tool message
                has_assistant_with_tool_calls = False
                for i in range(first_tool_idx - 1, -1, -1):
                    if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
                        has_assistant_with_tool_calls = True
                        break
                
                if not has_assistant_with_tool_calls:
                    # This should never happen if task history is maintained correctly
                    raise ValueError(
                        f"Tool messages found at index {first_tool_idx} but no preceding assistant message with tool_calls. "
                        f"This indicates the task history is missing the assistant message that emitted these tool calls. "
                        f"Messages: {[msg.get('role') for msg in messages]}"
                    )
        
        # Variables for streaming
        assistant_content = ""
        tool_calls: List[Any] = []
        has_tool_calls = False
        emitted_streaming_updates = False
        previous_tool_calls_state: Optional[str] = None  # Track previous tool calls state to avoid duplicate emissions
        
        # Call OpenAI with streaming (include system prompt)
        openai_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages
        
        # Always use streaming - emit incremental status updates as chunks arrive
        stream = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            stream=True,
        )
        
        # Stream response chunks and emit incremental updates (async iterator)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    assistant_content += delta.content
                    # Only emit incremental updates if we don't have tool calls yet
                    # (tool calls will be handled in incremental status updates)
                    if not has_tool_calls:
                        emitted_streaming_updates = True
                        incremental_message = create_text_message_object(
                            role=Role.agent,
                            content=assistant_content
                        )
                        status_update = TaskStatusUpdateEvent(
                            task_id=task_id or "",
                            context_id=context_id or "",
                            status=TaskStatus(
                                state=TaskState.working,  # Use 'working' for streaming updates
                                message=incremental_message
                            ),
                            final=False,
                        )
                        await event_queue.enqueue_event(status_update)
                if delta.tool_calls:
                    has_tool_calls = True
                    for tool_call_delta in delta.tool_calls:
                        # Initialize tool call if needed
                        idx = tool_call_delta.index or 0
                        while len(tool_calls) <= idx:
                            tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": "",
                                }
                            })
                        
                        if tool_call_delta.id:
                            tool_calls[idx]["id"] = tool_call_delta.id
                        
                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                tool_calls[idx]["function"]["name"] = tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                tool_calls[idx]["function"]["arguments"] += tool_call_delta.function.arguments
                    
                    # Emit incremental tool call updates only when state actually changes
                    # Convert current tool calls to MCP format (only non-empty ones)
                    current_mcp_tool_calls = []
                    for tc in tool_calls:
                        if tc.get("id"):  # Only include tool calls that have an ID
                            mcp_tc = convert_openai_tool_call_to_mcp(tc)
                            current_mcp_tool_calls.append(mcp_tc)
                    
                    # Serialize current state to compare with previous
                    current_state = json.dumps(current_mcp_tool_calls, sort_keys=True) if current_mcp_tool_calls else ""
                    
                    # Only emit if state changed
                    if current_state != previous_tool_calls_state:
                        previous_tool_calls_state = current_state
                        
                        # Build A2A message with current tool call state
                        incremental_message = create_text_message_object(
                            role=Role.agent,
                            content=assistant_content
                        )
                        if current_mcp_tool_calls:
                            incremental_message.parts.append(Part(DataPart(data={TOOL_CALLS_KEY: current_mcp_tool_calls})))
                        
                        # Emit incremental status update with working state
                        status_update = TaskStatusUpdateEvent(
                            task_id=task_id or "",
                            context_id=context_id or "",
                            status=TaskStatus(
                                state=TaskState.working,  # Use 'working' for incremental tool call updates
                                message=incremental_message
                            ),
                            final=False,
                        )
                        await event_queue.enqueue_event(status_update)
                        emitted_streaming_updates = True
        
        # Filter out empty tool calls
        tool_calls = [tc for tc in tool_calls if tc.get("id")]
        
        # If tool calls, emit them in A2A format with input-required state
        # Client will handle tool execution
        if tool_calls:
            # Human-in-the-loop: A2A input-required state
            # (Client can pause here for human input)
            
            # Convert OpenAI tool calls to MCP format
            mcp_tool_calls = []
            for tc in tool_calls:
                mcp_tc = convert_openai_tool_call_to_mcp(tc)
                mcp_tool_calls.append(mcp_tc)
            
            # Build A2A message with tool calls
            a2a_message = create_text_message_object(
                role=Role.agent,
                content=assistant_content
            )
            a2a_message.parts.append(Part(DataPart(data={TOOL_CALLS_KEY: mcp_tool_calls})))
            
            # Emit input-required status (human-in-the-loop point)
            status_update = TaskStatusUpdateEvent(
                task_id=task_id or "",
                context_id=context_id or "",
                status=TaskStatus(
                    state=TaskState.input_required,
                    message=a2a_message
                ),
                final=False,
            )
            await event_queue.enqueue_event(status_update)
        else:
            # No tool calls, task complete
            # Only emit final message if we haven't been streaming (to avoid duplicate)
            # If we've been streaming, the last incremental update will be the final one
            if not emitted_streaming_updates:
                a2a_message = create_text_message_object(
                    role=Role.agent,
                    content=assistant_content
                )
                
                status_update = TaskStatusUpdateEvent(
                    task_id=task_id or "",
                    context_id=context_id or "",
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=a2a_message
                    ),
                    final=True,
                )
                await event_queue.enqueue_event(status_update)
            else:
                # Emit final status update marking completion (reusing last content)
                final_message = create_text_message_object(
                    role=Role.agent,
                    content=assistant_content
                )
                
                status_update = TaskStatusUpdateEvent(
                    task_id=task_id or "",
                    context_id=context_id or "",
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=final_message
                    ),
                    final=True,
                )
                await event_queue.enqueue_event(status_update)
    
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel an ongoing task."""
        task_id = context.task_id
        context_id = context.context_id
        
        # Publish canceled status
        status_update = TaskStatusUpdateEvent(
            task_id=task_id or "",
            context_id=context_id or "",
            status=TaskStatus(state=TaskState.canceled),
            final=True,
        )
        await event_queue.enqueue_event(status_update)
