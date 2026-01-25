"""Loop class - AgentExecutor that uses MCP client to get system prompt and tools."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
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
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from timestep.utils.message_helpers import (
    extract_user_text_and_tool_results,
    convert_mcp_tool_to_openai,
    convert_openai_tool_call_to_mcp,
    build_tool_result_message,
)
from timestep.observability.tracing import get_tracer

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
        trace_to_file: str = "traces.jsonl",
    ):
        self.agent_id = agent_id
        self.model = model
        self.context_id_to_environment_uri = context_id_to_environment_uri
        self.human_in_loop = human_in_loop
        self.trace_to_file = trace_to_file
        
        # Setup OpenTelemetry tracing
        from timestep.observability.tracing import setup_tracing
        setup_tracing(exporter="file", file_path=trace_to_file)
        self.tracer = get_tracer(f"timestep.loop.{agent_id}")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute agent task using OpenAI and MCP client for system prompt/tools."""
        print('--------------------------------')
        print(dir(context))
        print('--------------------------------')

        print('context.configuration:', context.configuration)
        print('context.context_id:', context.context_id)
        print('context.current_task:', context.current_task)
        print('context.get_user_input:', context.get_user_input)
        print('context.message:', context.message)
        print('context.metadata:', context.metadata)
        print('context.task_id:', context.task_id)

        task_id = context.task_id
        context_id = context.context_id
        
        # Get Environment URI from context_id
        environment_uri = self.context_id_to_environment_uri.get(context_id)
        
        # Handle missing context_id:
        # 1. Try default context_id (None or empty string)
        if not environment_uri:
            environment_uri = self.context_id_to_environment_uri.get(None) or \
                              self.context_id_to_environment_uri.get("")
        
        # 2. If still not found and only one environment, use it
        if not environment_uri and len(self.context_id_to_environment_uri) == 1:
            environment_uri = next(iter(self.context_id_to_environment_uri.values()))
        
        # 3. If still not found, raise error
        if not environment_uri:
            available_context_ids = list(self.context_id_to_environment_uri.keys())
            raise ValueError(
                f"No environment found for context_id: {context_id}. "
                f"Available context_ids: {available_context_ids}. "
                f"To use a default environment, configure context_id_to_environment_uri with None or '' as a key."
            )
        
        # Trace: Get system prompt from Environment
        with self.tracer.start_as_current_span("get_system_prompt") as span:
            span.set_attribute("environment_uri", str(environment_uri))
            span.set_attribute("context_id", context_id or "")
            span.set_attribute("agent_id", self.agent_id)
            
            # Check if using stdio transport
            if isinstance(environment_uri, str) and environment_uri.startswith("stdio://"):
                # Use stdio client - connect to subprocess
                env_name = environment_uri.replace("stdio://", "")
                # Find the stdio environment script
                # Try relative to current file first, then try absolute paths
                current_file = Path(__file__)
                # lib/python/core/loop.py -> lib/python/examples/personal_assistant/
                env_script = current_file.parent.parent / "examples" / "personal_assistant" / f"{env_name}_env_stdio.py"
                if not env_script.exists():
                    # Try from workspace root
                    env_script = current_file.parent.parent.parent.parent / "lib" / "python" / "examples" / "personal_assistant" / f"{env_name}_env_stdio.py"
                stdio_params = StdioServerParameters(
                    command="uv",
                    args=["run", str(env_script.resolve())],
                )
                async with stdio_client(stdio_params) as (read, write):
                    async with ClientSession(read, write) as mcp_session:
                        await mcp_session.initialize()
                        
                        # Get system prompt (FastMCP prompt)
                        try:
                            system_prompt_result = await mcp_session.get_prompt("system_prompt", {
                                "agent_name": self.agent_id
                            })
                            system_prompt = system_prompt_result.messages[0].content.text
                        except Exception as e:
                            system_prompt = f"You are {self.agent_id}."
                            span.set_attribute("system_prompt_fallback", True)
                        
                        # Get available tools
                        tools_result = await mcp_session.list_tools()
                        tools = [
                            convert_mcp_tool_to_openai(tool)
                            for tool in tools_result.tools
                        ]
            else:
                # Use HTTP client
                async with streamable_http_client(environment_uri) as (read, write, _):
                    async with ClientSession(read, write) as mcp_session:
                        await mcp_session.initialize()
                        
                        # Get system prompt (FastMCP prompt)
                        # Human-in-the-loop: MCP Elicitation can happen here
                        try:
                            system_prompt_result = await mcp_session.get_prompt("system_prompt", {
                                "agent_name": self.agent_id
                            })
                            system_prompt = system_prompt_result.messages[0].content.text
                        except Exception as e:
                            # Fallback if prompt not found
                            system_prompt = f"You are {self.agent_id}."
                            span.set_attribute("system_prompt_fallback", True)
                        
                        # Get available tools
                        tools_result = await mcp_session.list_tools()
                        tools = [
                            convert_mcp_tool_to_openai(tool)
                            for tool in tools_result.tools
                        ]
            
            span.set_attribute("system_prompt", system_prompt)
            span.set_attribute("tools_count", len(tools))
        
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
        
        # Trace: Call OpenAI
        with self.tracer.start_as_current_span("call_openai") as span:
            span.set_attribute("model", self.model)
            span.set_attribute("messages_count", len(messages))
            
            openai_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages
            
            # Always use streaming
            # Note: OpenAI streaming returns a generator, we collect chunks synchronously
            # but this is fine since we're already in an async function
            stream = self.openai_client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                stream=True,
            )
            
            assistant_content = ""
            tool_calls: List[Any] = []
            
            # Collect streaming response (stream is a generator)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        assistant_content += delta.content
                    if delta.tool_calls:
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
            
            span.set_attribute("assistant_content_length", len(assistant_content))
            span.set_attribute("tool_calls_count", len(tool_calls))
        
        # If tool calls, execute via MCP client
        if tool_calls:
            # Human-in-the-loop: A2A input-required state
            # (Client can pause here for human input)
            
            # Trace: Execute tools
            with self.tracer.start_as_current_span("execute_tools") as span:
                span.set_attribute("tools_count", len(tool_calls))
                
                mcp_tool_calls = []
                for tc in tool_calls:
                    mcp_tc = convert_openai_tool_call_to_mcp(tc)
                    mcp_tool_calls.append(mcp_tc)
                
                # Execute tools via MCP client
                # Human-in-the-loop: MCP Sampling can happen here (for handoffs)
                if isinstance(environment_uri, str) and environment_uri.startswith("stdio://"):
                    # Use stdio client - connect to subprocess
                    env_name = environment_uri.replace("stdio://", "")
                    # Find the stdio environment script
                    current_file = Path(__file__)
                    env_script = current_file.parent.parent / "examples" / "personal_assistant" / f"{env_name}_env_stdio.py"
                    if not env_script.exists():
                        env_script = current_file.parent.parent.parent.parent / "lib" / "python" / "examples" / "personal_assistant" / f"{env_name}_env_stdio.py"
                    stdio_params = StdioServerParameters(
                        command="uv",
                        args=["run", str(env_script.resolve())],
                    )
                    async with stdio_client(stdio_params) as (read, write):
                        async with ClientSession(read, write) as mcp_session:
                            await mcp_session.initialize()
                            
                            tool_results = []
                            for tool_call in mcp_tool_calls:
                                with self.tracer.start_as_current_span("execute_tool") as tool_span:
                                    tool_span.set_attribute("tool_name", tool_call["name"])
                                    tool_span.set_attribute("tool_arguments", json.dumps(tool_call["arguments"]))
                                    
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
                                    
                                    tool_span.set_attribute("tool_result", str(tool_results[-1]))
                else:
                    # Use HTTP client
                    async with streamable_http_client(environment_uri) as (read, write, _):
                        async with ClientSession(read, write) as mcp_session:
                            await mcp_session.initialize()
                            
                            tool_results = []
                            for tool_call in mcp_tool_calls:
                                with self.tracer.start_as_current_span("execute_tool") as tool_span:
                                    tool_span.set_attribute("tool_name", tool_call["name"])
                                    tool_span.set_attribute("tool_arguments", json.dumps(tool_call["arguments"]))
                                    
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
                                    
                                    tool_span.set_attribute("tool_result", str(tool_results[-1]))
            
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
