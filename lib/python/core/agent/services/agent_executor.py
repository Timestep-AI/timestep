"""AgentExecutor class - AgentExecutor that uses MCP client to get system prompt and tools."""

import json
import logging
import os
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
from a2a.server.agent_execution.agent_executor import AgentExecutor as BaseAgentExecutor
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
    convert_a2a_message_to_openai,
    convert_a2a_message_to_agui_event,
    compact_events,
    convert_agui_event_to_openai_chat,
    get_agui_event_type_from_a2a_message,
    TOOL_CALLS_KEY,
    TOOL_RESULTS_KEY,
)
from timestep.utils.event_helpers import (
    get_agui_event_type_from_task_status_update,
    add_canonical_type_to_message,
)
from timestep.core.agent.stores.memory_store import MemoryStore

# Tracing not used
TRACING_AVAILABLE = False


class AgentExecutor(BaseAgentExecutor):
    """AgentExecutor is an AgentExecutor that uses MCP client to get system prompt and tools from Environment.
    
    The AgentExecutor:
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
        memory_store: Optional[MemoryStore] = None,
    ):
        self.agent_id = agent_id
        self.model = model
        self.context_id_to_environment_uri = context_id_to_environment_uri
        self.human_in_loop = human_in_loop
        self.memory_store = memory_store
        
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
        
        # === ABUNDANT LOGGING: Start of execute() ===
        logger.info("=" * 80)
        logger.info("=== EXECUTE() START ===")
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Context ID: {context_id}")
        
        # Log context.message
        logger.info("--- context.message ---")
        if isinstance(context.message, dict):
            msg_role = context.message.get("role", "unknown")
            msg_id = context.message.get("message_id") or context.message.get("messageId", "no-id")
            msg_parts = context.message.get("parts", [])
            logger.info(f"  Type: dict")
            logger.info(f"  Role: {msg_role}")
            logger.info(f"  Message ID: {msg_id}")
            logger.info(f"  Parts count: {len(msg_parts)}")
            for idx, part in enumerate(msg_parts):
                part_data = part.root if hasattr(part, "root") else part
                if isinstance(part_data, dict):
                    part_kind = part_data.get("kind", "unknown")
                    if part_kind == "text":
                        text_content = part_data.get("text", "")[:100]
                        logger.info(f"    Part[{idx}]: text - {text_content}...")
                    elif part_kind == "data":
                        data = part_data.get("data", {})
                        if TOOL_CALLS_KEY in data:
                            logger.info(f"    Part[{idx}]: data - tool_calls: {len(data.get(TOOL_CALLS_KEY, []))} calls")
                        if TOOL_RESULTS_KEY in data:
                            logger.info(f"    Part[{idx}]: data - tool_results: {len(data.get(TOOL_RESULTS_KEY, []))} results")
                else:
                    logger.info(f"    Part[{idx}]: {type(part_data).__name__}")
        else:
            msg_role = getattr(context.message, "role", "unknown")
            msg_id = getattr(context.message, "message_id", None) or getattr(context.message, "messageId", "no-id")
            msg_parts = getattr(context.message, "parts", []) if hasattr(context.message, "parts") else []
            logger.info(f"  Type: {type(context.message).__name__}")
            logger.info(f"  Role: {msg_role}")
            logger.info(f"  Message ID: {msg_id}")
            logger.info(f"  Parts count: {len(msg_parts)}")
        
        # Log context.current_task
        task = context.current_task
        logger.info("--- context.current_task ---")
        if task:
            logger.info(f"  Task ID: {task.task_id if hasattr(task, 'task_id') else 'unknown'}")
            logger.info(f"  Context ID: {task.context_id if hasattr(task, 'context_id') else 'unknown'}")
            logger.info(f"  State: {task.status.state.value if hasattr(task, 'status') and hasattr(task.status, 'state') else 'unknown'}")
            logger.info(f"  History count: {len(task.history) if task.history else 0}")
            
            # Log each message in task.history with full details
            if task.history:
                logger.info(f"  --- task.history ({len(task.history)} messages) ---")
                for hist_idx, hist_msg in enumerate(task.history):
                    if isinstance(hist_msg, dict):
                        hist_role = hist_msg.get("role", "unknown")
                        hist_id = hist_msg.get("message_id") or hist_msg.get("messageId", "no-id")
                        hist_parts = hist_msg.get("parts", [])
                        hist_metadata = hist_msg.get("metadata", {})
                        hist_canonical_type = hist_metadata.get("canonical_type", "none") if isinstance(hist_metadata, dict) else "none"
                    else:
                        hist_role = getattr(hist_msg, "role", "unknown")
                        hist_id = getattr(hist_msg, "message_id", None) or getattr(hist_msg, "messageId", "no-id")
                        hist_parts = getattr(hist_msg, "parts", []) if hasattr(hist_msg, "parts") else []
                        hist_metadata = getattr(hist_msg, "metadata", None)
                        hist_canonical_type = hist_metadata.get("canonical_type", "none") if isinstance(hist_metadata, dict) else "none"
                    
                    logger.info(f"    History[{hist_idx}]:")
                    logger.info(f"      Role: {hist_role}")
                    logger.info(f"      Message ID: {hist_id}")
                    logger.info(f"      Canonical Type: {hist_canonical_type}")
                    logger.info(f"      Parts count: {len(hist_parts)}")
                    for part_idx, part in enumerate(hist_parts):
                        part_data = part.root if hasattr(part, "root") else part
                        if isinstance(part_data, dict):
                            part_kind = part_data.get("kind", "unknown")
                            if part_kind == "text":
                                text_content = part_data.get("text", "")[:100]
                                logger.info(f"        Part[{part_idx}]: text - {text_content}...")
                            elif part_kind == "data":
                                data = part_data.get("data", {})
                                if TOOL_CALLS_KEY in data:
                                    tool_calls = data.get(TOOL_CALLS_KEY, [])
                                    logger.info(f"        Part[{part_idx}]: data - tool_calls: {len(tool_calls)} calls")
                                    for tc_idx, tc in enumerate(tool_calls[:2]):  # Show first 2
                                        tc_name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                                        logger.info(f"          ToolCall[{tc_idx}]: {tc_name}")
                                if TOOL_RESULTS_KEY in data:
                                    tool_results = data.get(TOOL_RESULTS_KEY, [])
                                    logger.info(f"        Part[{part_idx}]: data - tool_results: {len(tool_results)} results")
                        else:
                            logger.info(f"        Part[{part_idx}]: {type(part_data).__name__}")
            else:
                logger.info("  History: None or empty")
        else:
            logger.info("  Task: None")
        logger.info("=" * 80)
        
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
        logger.info("--- Extracting user_text and tool_results from incoming message ---")
        user_text, tool_results = extract_user_text_and_tool_results(context.message)
        logger.info(f"  user_text: {user_text[:100] if user_text else 'None'}...")
        logger.info(f"  tool_results count: {len(tool_results) if tool_results else 0}")
        if tool_results:
            for tr_idx, tr in enumerate(tool_results[:2]):  # Show first 2
                tr_call_id = tr.get("call_id", "unknown") if isinstance(tr, dict) else getattr(tr, "call_id", "unknown")
                logger.info(f"    ToolResult[{tr_idx}]: call_id={tr_call_id}")
        
        # Build messages list - always start with context-level memory history
        messages = []
        
        # Get context-level history from Agent's memory store
        if self.memory_store:
            memory_history = await self.memory_store.get_history(context_id)
            if memory_history:
                # Map-reduce-map pattern: A2A → AG-UI → compacted AG-UI → OpenAI
                logger.info(f"Processing context-level memory history: {len(memory_history)} messages")
                
                # Map: Convert A2A messages to AG-UI events
                logger.info("--- Converting A2A messages to AG-UI events ---")
                agui_events = []
                for msg_idx, msg in enumerate(memory_history):
                    agui_event = convert_a2a_message_to_agui_event(msg)
                    agui_events.append(agui_event)
                    event_type = agui_event.get("type", "unknown")
                    logger.info(f"  A2A Message[{msg_idx}] → AG-UI Event: type={event_type}")
                    if event_type in ["text-message-content", "text-message-end"]:
                        event_msg = agui_event.get("message", {})
                        event_role = event_msg.get("role", "unknown")
                        event_content = event_msg.get("content", "")[:100]
                        logger.info(f"    Role: {event_role}, Content: {event_content}...")
                    elif event_type in ["tool-call-start", "tool-call-args"]:
                        tool_calls = agui_event.get("toolCalls", [])
                        logger.info(f"    Tool calls: {len(tool_calls)}")
                        for tc_idx, tc in enumerate(tool_calls[:2]):  # Show first 2
                            tc_name = tc.get("name", "unknown")
                            logger.info(f"      ToolCall[{tc_idx}]: {tc_name}")
                    elif event_type == "tool-call-result":
                        results = agui_event.get("results", [])
                        logger.info(f"    Tool results: {len(results)}")
                logger.info(f"Converted to {len(agui_events)} AG-UI events")
                
                # Reduce: Compact streaming events into final events
                logger.info("--- Compacting AG-UI events ---")
                compacted_agui_events = compact_events(agui_events)
                logger.info(f"Compacted to {len(compacted_agui_events)} final AG-UI events")
                for comp_idx, comp_event in enumerate(compacted_agui_events):
                    comp_type = comp_event.get("type", "unknown")
                    logger.info(f"  Compacted[{comp_idx}]: type={comp_type}")
                
                # Map: Convert compacted AG-UI events to OpenAI format
                logger.info("--- Converting compacted AG-UI events to OpenAI format ---")
                for agui_idx, agui_event in enumerate(compacted_agui_events):
                    openai_msg, tool_messages = convert_agui_event_to_openai_chat(agui_event)
                    
                    if openai_msg:
                        openai_role = openai_msg.get("role", "unknown")
                        logger.info(f"  AG-UI Event[{agui_idx}] → OpenAI Message: role={openai_role}")
                        if openai_role == "user" or openai_role == "assistant":
                            content = openai_msg.get("content", "")[:100]
                            logger.info(f"    Content: {content}...")
                        if openai_msg.get("tool_calls"):
                            logger.info(f"    Tool calls: {len(openai_msg.get('tool_calls', []))}")
                    else:
                        logger.info(f"  AG-UI Event[{agui_idx}] → OpenAI Message: None (intermediate event)")
                    
                    if tool_messages:
                        logger.info(f"  AG-UI Event[{agui_idx}] → Tool Messages: {len(tool_messages)}")
                        for tm_idx, tm in enumerate(tool_messages[:2]):  # Show first 2
                            tm_tool_call_id = tm.get("tool_call_id", "unknown")
                            logger.info(f"    ToolMessage[{tm_idx}]: tool_call_id={tm_tool_call_id}")
                    
                    # Add the main message if it exists
                    if openai_msg:
                        messages.append(openai_msg)
                    # Add tool messages if any (from tool_results in user messages)
                    if tool_messages:
                        messages.extend(tool_messages)
        else:
            # Fallback: if no memory store, log warning and continue without history
            logger.warning("No memory store available, building messages without context history")
        
        # Add the current incoming message
        # Check if incoming message is already in memory (same message_id) to avoid duplicates
        incoming_message_id = None
        if isinstance(context.message, dict):
            incoming_message_id = context.message.get("message_id") or context.message.get("messageId")
        else:
            incoming_message_id = getattr(context.message, "message_id", None) or getattr(context.message, "messageId", None)
        
        # Check if this message is already in memory
        logger.info("--- Checking if incoming message is already in memory ---")
        logger.info(f"  Incoming message ID: {incoming_message_id}")
        message_already_in_memory = False
        if self.memory_store and incoming_message_id:
            memory_history = await self.memory_store.get_history(context_id)
            for hist_message in memory_history:
                if isinstance(hist_message, dict):
                    hist_msg_id = hist_message.get("message_id") or hist_message.get("messageId")
                else:
                    hist_msg_id = getattr(hist_message, "message_id", None) or getattr(hist_message, "messageId", None)
                if hist_msg_id == incoming_message_id:
                    message_already_in_memory = True
                    logger.info(f"  ✓ Incoming message (id: {incoming_message_id}) already in memory, skipping duplicate")
                    break
        if not message_already_in_memory:
            logger.info(f"  ✗ Incoming message not in memory, will add to messages array and memory")
        
        # Add incoming message to memory store if not already present
        # context.message should be a Message object from A2A SDK
        if not message_already_in_memory and self.memory_store:
            await self.memory_store.add_message(context_id, context.message)
            logger.info(f"  ✓ Added incoming message to memory store")
        
        if not message_already_in_memory:
            logger.info("--- Adding incoming message to messages array ---")
            if tool_results:
                # Add tool results as tool messages
                # The assistant message with matching tool_calls should already be in messages from history processing
                logger.info(f"  Adding {len(tool_results)} tool result(s) as tool messages")
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
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,
                    }
                    messages.append(tool_msg)
                    logger.info(f"    Added tool message: tool_call_id={tool_call_id}, content_length={len(content)}")
            elif user_text:
                # Add user message
                logger.info(f"  Adding user message: {user_text[:100]}...")
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
        
        # Log the incoming message
        logger.info(
            f"Incoming message: user_text={bool(user_text)}, "
            f"tool_results_count={len(tool_results) if tool_results else 0}"
        )
        
        # Log final messages array before OpenAI call
        messages_roles = [msg.get("role") for msg in messages]
        logger.info(f"Final messages array (before OpenAI call): roles={messages_roles}")
        
        # Log detailed structure for debugging
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                tool_call_ids = [tc.get("id") for tc in msg.get("tool_calls", []) if tc.get("id")]
                logger.info(f"Messages[{idx}]: assistant with tool_calls, ids={tool_call_ids}")
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                logger.info(f"Messages[{idx}]: tool message, tool_call_id={tool_call_id}")
            else:
                logger.info(f"Messages[{idx}]: {role}")
        
        # Call OpenAI with streaming (include system prompt)
        openai_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages
        
        # Log full messages array being sent to OpenAI
        logger.info("=== OpenAI Chat Completions Request ===")
        logger.info(f"Model: {self.model}")
        logger.info(f"Number of messages: {len(openai_messages)}")
        for idx, msg in enumerate(openai_messages):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "assistant" and msg.get("tool_calls"):
                tool_calls_summary = [f"{tc.get('function', {}).get('name', 'unknown')}({tc.get('function', {}).get('arguments', '')[:50]}...)" for tc in msg.get("tool_calls", [])]
                logger.info(f"  Message[{idx}]: {role} with tool_calls: {tool_calls_summary}")
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                content_preview = str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
                logger.info(f"  Message[{idx}]: {role} (tool_call_id={tool_call_id}): {content_preview}")
            else:
                content_preview = str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
                logger.info(f"  Message[{idx}]: {role}: {content_preview}")
        logger.info(f"Tools available: {len(tools) if tools else 0}")
        
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
                        # Add canonical_type for streaming chunk
                        incremental_message = add_canonical_type_to_message(incremental_message, "TextMessageChunkEvent")
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
                        # Add canonical_type for incremental tool call args (ToolCallArgsEvent for streaming updates)
                        incremental_message = add_canonical_type_to_message(incremental_message, "ToolCallArgsEvent")
                        
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
        
        # Log OpenAI response details
        logger.info("=== OpenAI Chat Completions Response ===")
        logger.info(f"Assistant content length: {len(assistant_content)}")
        if assistant_content:
            content_preview = assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content
            logger.info(f"Assistant content: {content_preview}")
        logger.info(f"Tool calls count: {len(tool_calls)}")
        if tool_calls:
            for idx, tc in enumerate(tool_calls):
                func_name = tc.get("function", {}).get("name", "unknown")
                func_args = tc.get("function", {}).get("arguments", "")
                func_args_preview = func_args[:100] + "..." if len(func_args) > 100 else func_args
                logger.info(f"  Tool call[{idx}]: {func_name}({func_args_preview})")
        
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
            # Add canonical_type for tool call start (final)
            a2a_message = add_canonical_type_to_message(a2a_message, "ToolCallStartEvent")
            
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
            
            # Add complete agent message (with tool calls) to memory store
            if self.memory_store:
                await self.memory_store.add_message(context_id, a2a_message)
                logger.info("  ✓ Added agent message (input-required with tool calls) to memory store")
        else:
            # No tool calls, task complete
            # Only emit final message if we haven't been streaming (to avoid duplicate)
            # If we've been streaming, the last incremental update will be the final one
            if not emitted_streaming_updates:
                a2a_message = create_text_message_object(
                    role=Role.agent,
                    content=assistant_content
                )
                # Add canonical_type for final text message
                a2a_message = add_canonical_type_to_message(a2a_message, "TextMessageEndEvent")
                
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
                
                # Add complete agent message (final response) to memory store
                if self.memory_store:
                    await self.memory_store.add_message(context_id, a2a_message)
                    logger.info("  ✓ Added agent message (completed, non-streaming) to memory store")
            else:
                # Emit final status update marking completion (reusing last content)
                final_message = create_text_message_object(
                    role=Role.agent,
                    content=assistant_content
                )
                # Add canonical_type for final text message after streaming
                final_message = add_canonical_type_to_message(final_message, "TextMessageEndEvent")
                
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
                
                # Add complete agent message (final response after streaming) to memory store
                if self.memory_store:
                    await self.memory_store.add_message(context_id, final_message)
                    logger.info("  ✓ Added agent message (completed, streaming) to memory store")
    
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
