"""Loop class - AgentExecutor that uses MCP client to get system prompt and tools."""

import json
import os
from typing import Dict, List, Any
from openai import OpenAI
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
)
from a2a.client.helpers import create_text_message_object
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

from timestep.utils.message_helpers import (
    extract_user_text_and_tool_results,
    convert_mcp_tool_to_openai,
    convert_openai_tool_call_to_mcp,
    build_tool_result_message,
)

# DataPart payload keys
TOOL_CALLS_KEY = "tool_calls"


class Loop(AgentExecutor):
    """Loop is an AgentExecutor that uses MCP client to get system prompt and tools from Environment.
    
    The Loop:
    1. Receives A2A RequestContext with context_id
    2. Uses MCP client to locate Environment by context_id
    3. Gets system prompt (FastMCP prompt) and tools from Environment
    4. Invokes OpenAI model (async/streaming)
    5. Executes tools via MCP client if needed
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
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
    
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
        
        # Extract user message and tool results
        user_text, tool_results = extract_user_text_and_tool_results(context.message)
        
        # Build OpenAI messages
        messages = []
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if tool_results:
            for tool_result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result["call_id"],
                    "content": str(tool_result["output"]),
                })
        
        # Variables for streaming
        assistant_content = ""
        tool_calls: List[Any] = []
        has_tool_calls = False
        emitted_streaming_updates = False
        
        # Call OpenAI with streaming
        openai_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages
        
        # Always use streaming - emit incremental status updates as chunks arrive
        stream = self.openai_client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            stream=True,
        )
        
        # Stream response chunks and emit incremental updates
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    assistant_content += delta.content
                    # Only emit incremental updates if we don't have tool calls yet
                    # (tool calls will be handled in final status update)
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
        
        # Filter out empty tool calls
        tool_calls = [tc for tc in tool_calls if tc.get("id")]
        
        # If tool calls, execute via MCP client
        # Note: We need to create a new session for tool execution since the previous
        # session was closed after getting the system prompt and tools.
        # The DELETE calls are normal MCP protocol cleanup when sessions close.
        if tool_calls:
            # Human-in-the-loop: A2A input-required state
            # (Client can pause here for human input)
            
            mcp_tool_calls = []
            for tc in tool_calls:
                mcp_tc = convert_openai_tool_call_to_mcp(tc)
                mcp_tool_calls.append(mcp_tc)
            
            # Execute tools via MCP client
            # Human-in-the-loop: MCP Sampling can happen here (for handoffs)
            # We create a new session here because we can't keep the previous one
            # open during the OpenAI API call (which happens asynchronously).
            # This creates a new session, which will send DELETE when it closes
            async with streamable_http_client(environment_uri) as (read, write, _):
                async with ClientSession(read, write) as mcp_session:
                    await mcp_session.initialize()
                    
                    tool_results = []
                    for tool_call in mcp_tool_calls:
                        result = await mcp_session.call_tool(
                            tool_call["name"],
                            tool_call["arguments"]
                        )
                        
                        # Extract result content
                        result_text = ""
                        if result.content:
                            for content_block in result.content:
                                if hasattr(content_block, "text"):
                                    result_text += content_block.text
                        
                        tool_results.append({
                            "call_id": tool_call["call_id"],
                            "name": tool_call["name"],
                            "output": result_text or None,
                        })
            
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
