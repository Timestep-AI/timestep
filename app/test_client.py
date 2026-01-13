# /// script
# dependencies = [
#   "a2a-sdk",
#   "mcp",
#   "httpx",
#   "fastapi",
#   "uvicorn",
# ]
# ///

"""
Test client that orchestrates A2A and MCP servers.
Handles the main loop: send message to A2A, forward tool calls to MCP, handle handoff with sampling.
"""

import os
import sys
import asyncio
import httpx
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from a2a.client import ClientFactory, ClientConfig
from a2a.client.helpers import create_text_message_object
from a2a.types import TransportProtocol, Role
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
import uvicorn

# Server URLs
A2A_BASE_URL = os.getenv("A2A_URL", "http://localhost:8000")
MCP_URL = os.getenv("MCP_URL", "http://localhost:8080")

# Agent IDs
PERSONAL_ASSISTANT_ID = "00000000-0000-0000-0000-000000000000"
WEATHER_ASSISTANT_ID = "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"


def write_task(task: Any, agent_id: str) -> None:
    """Write task to tasks/ folder in proper A2A Task format."""
    try:
        if not task:
            print(f"\n[Warning: write_task called with None task]", file=sys.stderr)
            return
        if not hasattr(task, 'id'):
            print(f"\n[Warning: task object has no 'id' attribute]", file=sys.stderr)
            return
        if not task.id:
            print(f"\n[Warning: task.id is empty]", file=sys.stderr)
            return
        
        # Use relative path for local execution, /workspace for Docker
        tasks_dir = Path("tasks")
        tasks_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().isoformat().replace(":", "-")
        # Use short task_id for filename (first 8 chars)
        task_id_short = task.id[:8] if task.id else "unknown"
        agent_id_short = agent_id[:8] if agent_id else "unknown"
        task_file = tasks_dir / f"{timestamp}_{task_id_short}_{agent_id_short}.json"
        
        # Serialize Task to JSON using the A2A Task object directly
        with open(task_file, "w") as f:
            json.dump(task.model_dump(mode="json", exclude_none=True), f, indent=2)
        print(f"\n[Saved task to {task_file}]", file=sys.stderr)
    except Exception as e:
        # Log error but don't crash the client
        import traceback
        print(f"\n[Error: Failed to save task: {e}]", file=sys.stderr)
        print(f"[Traceback: {traceback.format_exc()}]", file=sys.stderr)


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call MCP tool using MCP Python SDK client."""
    try:
        async with streamable_http_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result: CallToolResult = await session.call_tool(tool_name, arguments)
                
                # Convert CallToolResult to dict format
                # CallToolResult has a 'content' field which is a list of content items
                if result.content:
                    # Extract text from content items
                    text_parts = []
                    for item in result.content:
                        if isinstance(item, TextContent):
                            text_parts.append(item.text)
                        elif isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                    
                    if text_parts:
                        return {"result": " ".join(text_parts)}
                    else:
                        # Return structured content
                        return {"result": [item.model_dump() if hasattr(item, 'model_dump') else item for item in result.content]}
                else:
                    return {"result": None}
    except Exception as e:
        return {"error": str(e)}


# Create FastAPI app for handling sampling requests
sampling_app = FastAPI()

async def sampling_handler(
    messages: list[Dict[str, Any]],
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    FastMCP-style sampling handler that processes LLM sampling requests.
    Matches the pattern from https://fastmcp.wiki/en/clients/sampling
    
    Args:
        messages: List of sampling messages (each with 'role' and 'content')
        params: Sampling parameters (systemPrompt, temperature, maxTokens, etc.)
        context: Request context (optional)
    
    Returns:
        Generated response text
    """
    print(f"\n[DEBUG: Sampling handler called]", file=sys.stderr)
    print(f"[DEBUG: messages={messages}]", file=sys.stderr)
    print(f"[DEBUG: params={params}]", file=sys.stderr)
    print(f"[DEBUG: context={context}]", file=sys.stderr)
    
    # Extract agent_id from prompt (handoff tool includes it in the prompt)
    # Also check query parameter as fallback
    agent_id = None
    import re
    
    # Extract agent_uri from context (passed by handoff tool)
    agent_uri = None
    if context:
        agent_uri = context.get("agent_uri")
        agent_id = context.get("agent_id")  # Also check for direct agent_id
    
    # Extract agent_id from agent_uri if we have it
    if agent_uri and not agent_id:
        import re
        agent_id_match = re.search(r'/agents/([^/\s]+)', agent_uri)
        if agent_id_match:
            agent_id = agent_id_match.group(1)
    
    # Extract message content from messages
    # FastMCP format: messages have 'role' and 'content' (with .text for text content)
    message_text = None
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, dict) and "text" in content:
                message_text = content["text"]
            elif isinstance(content, str):
                message_text = content
        elif isinstance(msg, str):
            message_text = msg
    
    if not message_text:
        message_text = "Please help with this task."
    
    # Extract agent_id from prompt if not already found
    # Handoff tool includes "Agent ID: ..." in the prompt
    if not agent_id and message_text:
        agent_id_match = re.search(r"Agent ID: ([^\s\n]+)", message_text)
        if agent_id_match:
            agent_id = agent_id_match.group(1)
    
    # Extract agent_uri from prompt to get agent_id as fallback
    if not agent_id and message_text:
        agent_uri_match = re.search(r"Agent URI: (http[^\s\n]+)", message_text)
        if agent_uri_match:
            agent_uri = agent_uri_match.group(1)
            # Extract agent_id from agent_uri
            agent_id_match = re.search(r'/agents/([^/\s]+)', agent_uri)
            if agent_id_match:
                agent_id = agent_id_match.group(1)
    
    # Extract context_id from prompt
    context_id = None
    if message_text:
        context_match = re.search(r"Context ID: ([^\s\n]+)", message_text)
        if context_match:
            context_id = context_match.group(1) if context_match.group(1) != "none" else None
    
    # Extract the actual message to send to the agent (after "Message to send to the agent:")
    actual_message = message_text
    if message_text and "Message to send to the agent:" in message_text:
        message_match = re.search(r"Message to send to the agent:\s*\n(.+)", message_text, re.DOTALL)
        if message_match:
            actual_message = message_match.group(1).strip()
    
    print(f"[DEBUG: Extracted agent_uri={agent_uri}, agent_id={agent_id}, context_id={context_id}, actual_message={actual_message}]", file=sys.stderr)
    
    # If we have agent_id, handle the sampling by calling A2A agent
    if agent_id:
        print(f"[DEBUG: Calling handle_mcp_sampling_internal with agent_id={agent_id}]", file=sys.stderr)
        return await handle_mcp_sampling_internal(
            agent_id=agent_id,
            context_id=context_id,
            message=actual_message,
            task_ids_tracker=None,  # Can't track here easily, will be tracked in main loop
        )
    else:
        error_msg = "Error: agent_id is required for sampling. Could not extract agent_id from prompt."
        print(f"[DEBUG: {error_msg}]", file=sys.stderr)
        return error_msg


@sampling_app.post("/sampling/complete")
async def handle_sampling_complete(request: Request):
    """Handle MCP sampling/complete requests (standard MCP endpoint)."""
    try:
        body = await request.json()
        
        # Extract agent_id from query parameter (passed via agent_uri)
        agent_id = request.query_params.get("agent_id")
        
        # Parse FastMCP-style sampling request
        # Format: { "messages": [...], "params": {...}, "context": {...} }
        messages = body.get("messages", [])
        params = body.get("params", {})
        context = body.get("context", {})
        
        # If agent_id not in context, add it
        if agent_id and "agent_id" not in context:
            context["agent_id"] = agent_id
        
        # If messages is a simple list of strings, convert to FastMCP format
        if messages and isinstance(messages[0], str):
            messages = [{"role": "user", "content": {"text": msg}} for msg in messages]
        
        # If params is empty but body has direct fields, extract them
        if not params:
            params = {
                "systemPrompt": body.get("system_prompt") or body.get("systemPrompt"),
                "temperature": body.get("temperature"),
                "maxTokens": body.get("maxTokens") or body.get("max_tokens", 1000),
            }
        
        # Call the sampling handler
        result = await sampling_handler(messages, params, context)
        
        return JSONResponse({
            "text": result,
        })
    
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)


async def handle_mcp_sampling_internal(
    agent_id: str,
    context_id: Optional[str],
    message: Optional[str],
    task_ids_tracker: Optional[List[str]] = None,
) -> str:
    """
    Handle MCP sampling request by calling the appropriate A2A agent.
    This is called when MCP handoff tool uses ctx.sample().
    Returns the final assistant message text.
    """

    print(f"Handle MCP sampling internal agent_id: {agent_id}")
    print(f"Handle MCP sampling internal context_id: {context_id}")
    print(f"Handle MCP sampling internal message: {message}")
    print(f"Handle MCP sampling internal task_ids_tracker: {task_ids_tracker}")

    # Construct A2A URL with agent path
    agent_url = f"{A2A_BASE_URL}/agents/{agent_id}"
    
    # Create A2A client for the target agent
    httpx_client = httpx.AsyncClient(timeout=60.0)
    config = ClientConfig(
        streaming=True,
        polling=False,
        httpx_client=httpx_client,
        supported_transports=[TransportProtocol.http_json],  # Support HTTP+JSON transport
    )
    
    try:
        a2a_client = await ClientFactory.connect(agent_url, client_config=config)
    except Exception as e:
        return f"Error connecting to agent {agent_id}: {e}"
    
    try:
        # Create message from handoff
        handoff_message = message or "Please help with this task."
        message_obj = create_text_message_object(
            role="user",
            content=handoff_message,
        )
        
        # Run full agent loop for the target agent
        final_message = ""
        task_id = None
        context_id = None
        
        async for event in a2a_client.send_message(message_obj):
            # Task-generating agents ALWAYS return Task objects - no exceptions
            if isinstance(event, tuple):
                task, update = event
            elif hasattr(event, 'id') and hasattr(event, 'status'):
                task = event
                update = None
            else:
                # This should NEVER happen - Task-generating agents only return Tasks
                raise ValueError(f"Received non-Task event from Task-generating agent: {type(event)}")
            
            # We MUST have a Task object
            if not task or not hasattr(task, 'id') or not hasattr(task, 'status'):
                raise ValueError(f"Invalid Task object received: {task}")
            
            # Save task to tasks folder
            write_task(task, agent_id)
            
            # Track task ID from task object
            if hasattr(task, 'id') and task.id:
                if task_ids_tracker is not None and task.id not in task_ids_tracker:
                    task_ids_tracker.append(task.id)
            task_id = task.id if hasattr(task, 'id') else None
            context_id = task.context_id if hasattr(task, 'context_id') else None
            
            # Collect agent messages (Task has 'history' not 'messages')
            if task.history:
                for msg in task.history:
                    if msg.role == Role.agent or (hasattr(msg, 'role') and str(msg.role) == "agent"):
                        if hasattr(msg, 'parts'):
                            for part in msg.parts:
                                if hasattr(part, 'kind') and part.kind == 'text':
                                    final_message += part.text
                        elif hasattr(msg, 'content'):
                            final_message += msg.content
            
            # Check if task is completed
            if task.status.state.value == "completed":
                break
            
            # Check if input is required (tool calls)
            if task.status.state.value == "input-required":
                # Check for tool calls in task.status.message parts (DataPart with tool_calls)
                tool_calls = None
                if task.status.message and task.status.message.parts:
                    for part in task.status.message.parts:
                        # Check if this is a DataPart with tool_calls
                        if hasattr(part, 'root'):
                            part_data = part.root
                            if hasattr(part_data, 'kind') and part_data.kind == 'data':
                                if hasattr(part_data, 'data') and isinstance(part_data.data, dict):
                                    tool_calls = part_data.data.get("tool_calls")
                                    if tool_calls:
                                        break
                
                # Fallback: check last agent message in history
                if not tool_calls and task.history:
                    for msg in reversed(task.history):
                        if msg.role == Role.agent or (hasattr(msg, 'role') and str(msg.role) == "agent"):
                            if msg.parts:
                                for part in msg.parts:
                                    if hasattr(part, 'root'):
                                        part_data = part.root
                                        if hasattr(part_data, 'kind') and part_data.kind == 'data':
                                            if hasattr(part_data, 'data') and isinstance(part_data.data, dict):
                                                tool_calls = part_data.data.get("tool_calls")
                                                if tool_calls:
                                                    break
                            if tool_calls:
                                break
                
                if tool_calls:
                        # Execute tool calls via MCP
                        for tool_call in tool_calls:
                            # Tool calls from DataPart are dicts, not objects
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get("function", {}).get("name")
                                tool_args_str = tool_call.get("function", {}).get("arguments")
                            else:
                                # Fallback for object-style access
                                tool_name = tool_call.function.name if hasattr(tool_call, 'function') else None
                                tool_args_str = tool_call.function.arguments if hasattr(tool_call, 'function') else None
                            
                            # Parse arguments
                            try:
                                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                            except:
                                tool_args = {}
                            
                            # Call MCP tool
                            result = await call_mcp_tool(tool_name, tool_args)
                            
                            # Send tool result back to A2A as continuation message
                            # Per A2A spec 6.3: use same task_id and context_id for continuation
                            tool_result_msg = create_text_message_object(
                                role="user",
                                content=json.dumps(result),
                            )
                            # Set task_id and context_id for continuation (multi-turn interaction)
                            if task_id:
                                tool_result_msg.task_id = task_id
                            if context_id:
                                tool_result_msg.context_id = context_id
                            
                            # Continue with tool result
                            async for event2 in a2a_client.send_message(tool_result_msg):
                                # Task-generating agents ALWAYS return Task objects - no exceptions
                                if isinstance(event2, tuple):
                                    task2, update2 = event2
                                elif hasattr(event2, 'id') and hasattr(event2, 'status'):
                                    task2 = event2
                                    update2 = None
                                else:
                                    # This should NEVER happen - Task-generating agents only return Tasks
                                    raise ValueError(f"Received non-Task event from Task-generating agent: {type(event2)}")
                                
                                # We MUST have a Task object
                                if not task2 or not hasattr(task2, 'id') or not hasattr(task2, 'status'):
                                    raise ValueError(f"Invalid Task object received: {task2}")
                                
                                # Save task to tasks folder
                                write_task(task2, agent_id)
                                
                                # Track task ID from task object
                                if hasattr(task2, 'id') and task2.id:
                                    if task_ids_tracker is not None and task2.id not in task_ids_tracker:
                                        task_ids_tracker.append(task2.id)
                                
                                # Update final message
                                if task2.history:
                                    for msg2 in task2.history:
                                        if msg2.role == Role.agent or (hasattr(msg2, 'role') and str(msg2.role) == "agent"):
                                            if hasattr(msg2, 'parts'):
                                                for part in msg2.parts:
                                                    if hasattr(part, 'kind') and part.kind == 'text':
                                                        final_message += part.text
                                            elif hasattr(msg2, 'content'):
                                                final_message += msg2.content
                                
                                if task2.status.state.value == "completed":
                                    break
        
        return final_message.strip() or "Task completed."
    
    finally:
        await httpx_client.aclose()


async def run_client_loop(
    initial_message: str,
    agent_id: str = PERSONAL_ASSISTANT_ID,
) -> None:
    """Main client loop that orchestrates A2A and MCP (fully async)."""
    
    # Construct A2A URL with agent path - ClientFactory will fetch agent card from /.well-known/agent-card.json
    agent_url = f"{A2A_BASE_URL}/agents/{agent_id}"
    
    # Track all task IDs encountered
    task_ids: List[str] = []
    
    # Create A2A client using ClientFactory
    # The client will fetch the agent card from {agent_url}/.well-known/agent-card.json
    httpx_client = httpx.AsyncClient(timeout=60.0)
    config = ClientConfig(
        streaming=True,
        polling=False,
        httpx_client=httpx_client,
        supported_transports=[TransportProtocol.http_json],  # Support HTTP+JSON transport
    )
    
    a2a_client = None
    try:
        # ClientFactory.connect expects a URL that points to where the agent card can be fetched
        # It will append /.well-known/agent-card.json to the URL
        a2a_client = await ClientFactory.connect(agent_url, client_config=config)
    except Exception as e:
        print(f"Error connecting to A2A server: {e}")
        raise
    
    try:
        # Create initial message
        message = create_text_message_object(
            role="user",
            content=initial_message,
        )
        
        # Send message to A2A server and process events
        task_id = None
        context_id = None
        print(f"\n[DEBUG: Starting to send message to A2A server]", file=sys.stderr)
        try:
            async for event in a2a_client.send_message(message):
                # Task-generating agents ALWAYS return Task objects - no exceptions
                if isinstance(event, tuple):
                    task, update = event
                elif hasattr(event, 'id') and hasattr(event, 'status'):
                    task = event
                    update = None
                else:
                    # This should NEVER happen - Task-generating agents only return Tasks
                    raise ValueError(f"Received non-Task event from Task-generating agent: {type(event)}")
                
                # We MUST have a Task object
                if not task or not hasattr(task, 'id') or not hasattr(task, 'status'):
                    raise ValueError(f"Invalid Task object received: {task}")
                
                # Save task to tasks folder
                print(f"\n[DEBUG: Received task, id={getattr(task, 'id', 'NO_ID')}, type={type(task)}]", file=sys.stderr)
                write_task(task, agent_id)
                
                # Track task ID from task object
                if hasattr(task, 'id') and task.id:
                    if task.id not in task_ids:
                        task_ids.append(task.id)
                task_id = task.id if hasattr(task, 'id') else None
                context_id = task.context_id if hasattr(task, 'context_id') else None
                
                # Print agent messages (Task has 'history' not 'messages')
                if task.history:
                    for msg in task.history:
                        if msg.role == Role.agent or (hasattr(msg, 'role') and str(msg.role) == "agent"):
                            if hasattr(msg, 'parts'):
                                for part in msg.parts:
                                    if hasattr(part, 'kind') and part.kind == 'text':
                                        print(part.text, end="", flush=True)
                            elif hasattr(msg, 'content'):
                                print(msg.content, end="", flush=True)
                
                # Check if task is completed
                if task.status.state.value == "completed":
                    print("\n[Task completed]")
                    break
                
                # Check if input is required (tool calls)
                if task.status.state.value == "input-required":
                        # Check for tool calls in task.status.message parts (DataPart with tool_calls)
                        tool_calls = None
                        if task.status.message and task.status.message.parts:
                            for part in task.status.message.parts:
                                # Check if this is a DataPart with tool_calls
                                if hasattr(part, 'root'):
                                    part_data = part.root
                                    if hasattr(part_data, 'kind') and part_data.kind == 'data':
                                        if hasattr(part_data, 'data') and isinstance(part_data.data, dict):
                                            tool_calls = part_data.data.get("tool_calls")
                                            if tool_calls:
                                                break
                        
                        # Fallback: check last agent message in history
                        if not tool_calls and task.history:
                            for msg in reversed(task.history):
                                if msg.role == Role.agent or (hasattr(msg, 'role') and str(msg.role) == "agent"):
                                    if msg.parts:
                                        for part in msg.parts:
                                            if hasattr(part, 'root'):
                                                part_data = part.root
                                                if hasattr(part_data, 'kind') and part_data.kind == 'data':
                                                    if hasattr(part_data, 'data') and isinstance(part_data.data, dict):
                                                        tool_calls = part_data.data.get("tool_calls")
                                                        if tool_calls:
                                                            break
                                    if tool_calls:
                                        break
                        
                        if tool_calls:
                            # Execute tool calls via MCP
                            pending_events = []
                            
                            for tool_call in tool_calls:
                                # Tool calls from DataPart are dicts, not objects
                                if isinstance(tool_call, dict):
                                    tool_id = tool_call.get("id")
                                    tool_name = tool_call.get("function", {}).get("name")
                                    tool_args_str = tool_call.get("function", {}).get("arguments")
                                else:
                                    # Fallback for object-style access
                                    tool_id = tool_call.id if hasattr(tool_call, 'id') else None
                                    tool_name = tool_call.function.name if hasattr(tool_call, 'function') else None
                                    tool_args_str = tool_call.function.arguments if hasattr(tool_call, 'function') else None
                                
                                # Parse arguments
                                try:
                                    tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                                except:
                                    tool_args = {}
                                
                                print(f"\n[Calling tool: {tool_name}]")
                                
                                # Check if it's a handoff
                                if tool_name == "handoff":
                                    # Call the handoff tool - it will make HTTP request to agent_uri
                                    handoff_result = await call_mcp_tool("handoff", tool_args)
                                    
                                    # The handoff_result should contain the sampling response
                                    # Add tool result as event
                                    pending_events.append({
                                        "kind": "tool-result",
                                        "toolCallId": tool_id,
                                        "content": json.dumps(handoff_result),
                                    })
                                else:
                                    # Regular tool call (e.g., get_weather)
                                    result = await call_mcp_tool(tool_name, tool_args)
                                    print(f"[Tool result: {result}]")
                                    
                                    # Add tool result as event
                                    pending_events.append({
                                        "kind": "tool-result",
                                        "toolCallId": tool_id,
                                        "content": json.dumps(result),
                                    })
                            
                            # Send all tool results back to A2A as continuation messages
                            # Per A2A spec 6.3: use same task_id and context_id for continuation
                            if pending_events:
                                # We need to send tool results as messages
                                # The A2A protocol expects tool results as part of the message
                                for event in pending_events:
                                    tool_result_msg = create_text_message_object(
                                        role="user",
                                        content=event["content"],
                                    )
                                    # Set task_id and context_id for continuation (multi-turn interaction)
                                    if task_id:
                                        tool_result_msg.task_id = task_id
                                    if context_id:
                                        tool_result_msg.context_id = context_id
                                    
                                    # Continue the loop with tool result
                                    async for event2 in a2a_client.send_message(tool_result_msg):
                                        # Task-generating agents ALWAYS return Task objects - no exceptions
                                        if isinstance(event2, tuple):
                                            task2, update2 = event2
                                        elif hasattr(event2, 'id') and hasattr(event2, 'status'):
                                            task2 = event2
                                            update2 = None
                                        else:
                                            # This should NEVER happen - Task-generating agents only return Tasks
                                            raise ValueError(f"Received non-Task event from Task-generating agent: {type(event2)}")
                                        
                                        # We MUST have a Task object
                                        if not task2 or not hasattr(task2, 'id') or not hasattr(task2, 'status'):
                                            raise ValueError(f"Invalid Task object received: {task2}")
                                        
                                        # Save task to tasks folder
                                        print(f"\n[DEBUG: Received task2, id={getattr(task2, 'id', 'NO_ID')}, type={type(task2)}]", file=sys.stderr)
                                        write_task(task2, agent_id)
                                        
                                        # Track task ID from task object
                                        if hasattr(task2, 'id') and task2.id:
                                            if task2.id not in task_ids:
                                                task_ids.append(task2.id)
                                        
                                        # Print agent messages (Task has 'history' not 'messages')
                                        if task2.history:
                                            for msg in task2.history:
                                                if msg.role == Role.agent or (hasattr(msg, 'role') and str(msg.role) == "agent"):
                                                    if hasattr(msg, 'parts'):
                                                        for part in msg.parts:
                                                            if hasattr(part, 'kind') and part.kind == 'text':
                                                                print(part.text, end="", flush=True)
                                                    elif hasattr(msg, 'content'):
                                                        print(msg.content, end="", flush=True)
                                        
                                        if task2.status.state.value == "completed":
                                            print("\n[Task completed]")
                                            return
                                        break
        except Exception as e:
            print(f"\n[Error in client loop: {e}]")
            raise
    
    finally:
        # Cleanup
        if a2a_client:
            await a2a_client.close()
        await httpx_client.aclose()


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: uv run app/test_client.py <message>")
        sys.exit(1)
    
    message = " ".join(sys.argv[1:])
    
    # Start the sampling server in the background
    config = uvicorn.Config(
        app=sampling_app,
        host="0.0.0.0",
        port=3002,
        log_level="error",
    )
    server = uvicorn.Server(config)
    
    # Start server and run client loop
    async def run_with_sampling():
        # Start the sampling server
        server_task = asyncio.create_task(server.serve())
        
        try:
            # Wait a bit for server to start
            await asyncio.sleep(0.5)
            
            # Run the client loop
            await run_client_loop(message, agent_id=PERSONAL_ASSISTANT_ID)
        finally:
            # Cleanup - gracefully shutdown the server
            server.should_exit = True
            # Wait for server to shutdown gracefully
            try:
                await asyncio.wait_for(server_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # Server didn't shutdown in time, cancel it
                if not server_task.done():
                    server_task.cancel()
                    try:
                        await server_task
                    except asyncio.CancelledError:
                        pass
    
    await run_with_sampling()


if __name__ == "__main__":
    asyncio.run(main())
