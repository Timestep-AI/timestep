# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "a2a-sdk[http-server]",
#   "mcp",
#   "openai",
#   "fastapi",
#   "uvicorn",
#   "httpx",
# ]
# ///

"""Weather Assistant Agent - A2A Server.

This script runs a Weather Assistant Agent that connects to an Environment
via MCP to get system prompts and tools.
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from uuid import uuid4
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx

# Fix package import: lib/python/ contains the timestep package
# but Python needs to import it as 'timestep', not 'python'
script_dir = Path(__file__).parent
lib_dir = script_dir.parent / "lib"
lib_python_dir = lib_dir / "python"

# Add lib/python to path
if str(lib_python_dir) not in sys.path:
    sys.path.insert(0, str(lib_python_dir))

# Create a 'timestep' module that points to the python directory
# This allows imports like 'from timestep.core import Agent' to work
import types
timestep_module = types.ModuleType('timestep')
timestep_module.__path__ = [str(lib_python_dir)]
sys.modules['timestep'] = timestep_module

# Now import the core module which will set up timestep.core
import importlib.util
core_init_path = lib_python_dir / "core" / "__init__.py"
spec = importlib.util.spec_from_file_location("timestep.core", core_init_path)
core_module = importlib.util.module_from_spec(spec)
sys.modules['timestep.core'] = core_module
spec.loader.exec_module(core_module)

# Now we can import Agent and Environment
from timestep.core import Agent, Environment
from timestep.utils.message_helpers import (
    extract_user_text_and_tool_results,
    build_tool_result_message,
    TOOL_CALLS_KEY,
    TOOL_RESULTS_KEY,
)
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Message,
    Part,
    DataPart,
    Role,
)
from a2a.client.helpers import create_text_message_object
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
from mcp.types import TextContent


def main():
    """Run the Weather Assistant Agent."""
    # Get port from environment variable or use default
    port = int(os.getenv("WEATHER_AGENT_PORT", "9999"))
    host = "0.0.0.0"
    http_host = "localhost" if host == "0.0.0.0" else host
    
    # Create Environment instance
    environment = Environment(
        environment_id="weather-assistant-env",
        context_id="weather-context",
        agent_id="weather-assistant",
        human_in_loop=False,
    )
    
    # Add get_weather tool to the environment
    @environment.tool()
    async def get_weather(location: str) -> Dict[str, Any]:
        """Get the current weather for a specific location."""
        # Return hardcoded weather data
        return {
            "location": location,
            "temperature": "72°F",
            "condition": "Sunny",
            "humidity": "65%"
        }
    
    # Update system prompt for weather assistant
    @environment.prompt()
    def system_prompt(agent_name: str) -> str:
        """System prompt for the weather assistant agent."""
        return f"You are {agent_name}, a helpful weather assistant. You can get weather information for any location using the get_weather tool."
    
    # Get MCP app from environment
    mcp_app = environment.streamable_http_app()
    
    # Create agent
    agent = Agent(
        agent_id="weather-assistant",
        name="Weather Assistant",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        context_id_to_environment_uri={
            "weather-context": f"http://{http_host}:{port}/mcp"
        },
        human_in_loop=False,
    )
    
    # Get FastAPI app from agent
    fastapi_app = agent.fastapi_app
    
    # Manually manage MCP task group (required for streamable HTTP)
    # This is what run_streamable_http_async() does internally
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: Create and initialize task group
        import anyio
        tg = anyio.create_task_group()
        await tg.__aenter__()
        app._mcp_task_group = tg
        
        # Set task group on session manager
        session_manager = environment.session_manager
        if hasattr(session_manager, '_task_group') and session_manager._task_group is None:
            session_manager._task_group = tg
        
        yield
        
        # Shutdown: Clean up task group
        if hasattr(app, '_mcp_task_group') and app._mcp_task_group:
            try:
                await app._mcp_task_group.__aexit__(None, None, None)
            except Exception:
                pass
    
    # Create a new FastAPI app with lifespan
    combined_app = FastAPI(
        title="Weather Assistant Agent",
        lifespan=lifespan,
    )
    
    # Include all routes from the agent's FastAPI app
    for route in fastapi_app.routes:
        combined_app.routes.append(route)
    
    # Include all routes from the MCP app
    for route in mcp_app.routes:
        combined_app.routes.append(route)
    
    # Helper functions for /v1/responses endpoint
    def extract_tool_calls(task: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from task status message DataPart."""
        # Handle both dict and object access
        if isinstance(task, dict):
            status = task.get('status', {})
        else:
            status = getattr(task, 'status', None)
        
        if not status:
            return None
        
        # Get message from status
        if isinstance(status, dict):
            status_message = status.get('message')
        else:
            status_message = getattr(status, 'message', None)
        
        if not status_message:
            return None
        
        # Get parts from message
        if isinstance(status_message, dict):
            parts = status_message.get('parts', [])
        else:
            parts = getattr(status_message, 'parts', [])
        
        if not parts:
            return None
        
        # Look for tool calls in parts
        for part in parts:
            part_data = part.root if hasattr(part, 'root') else part
            if isinstance(part_data, dict):
                part_kind = part_data.get('kind')
                part_data_dict = part_data.get('data') if part_kind == 'data' else None
            else:
                part_kind = getattr(part_data, 'kind', None)
                part_data_dict = getattr(part_data, 'data', None) if part_kind == 'data' else None
            
            if part_kind == 'data' and part_data_dict:
                if isinstance(part_data_dict, dict):
                    tool_calls = part_data_dict.get(TOOL_CALLS_KEY)
                else:
                    tool_calls = getattr(part_data_dict, TOOL_CALLS_KEY, None) if hasattr(part_data_dict, TOOL_CALLS_KEY) else None
                
                if tool_calls:
                    return tool_calls
        
        return None
    
    async def execute_tool_via_mcp(tool_name: str, arguments: Dict[str, Any], environment_uri: str) -> Dict[str, Any]:
        """Execute a single tool via MCP."""
        try:
            async with streamable_http_client(environment_uri) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    
                    if result.content:
                        text_parts = [item.text for item in result.content if isinstance(item, TextContent)]
                        return {"result": " ".join(text_parts)} if text_parts else {"result": None}
                    return {"result": None}
        except Exception as e:
            return {"error": str(e)}
    
    def convert_responses_input_to_a2a(input_data: Any) -> Message:
        """Convert Responses API input to A2A message format.
        
        Input can be:
        - A string
        - A list of messages (for compatibility)
        - A list of items (from previous response output)
        """
        # Handle string input
        if isinstance(input_data, str):
            return create_text_message_object(role=Role.user, content=input_data)
        
        # Handle list input (messages or items)
        if isinstance(input_data, list):
            # Find the last user message or text item
            user_content = ""
            tool_results = []
            
            for item in input_data:
                # Handle message format (for compatibility)
                if isinstance(item, dict):
                    role = item.get("role", "")
                    if role == "user":
                        content = item.get("content", "")
                        if isinstance(content, str):
                            user_content = content
                        elif isinstance(content, list):
                            text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                            user_content = " ".join(text_parts)
                    elif role == "tool":
                        tool_call_id = item.get("tool_call_id") or item.get("call_id")
                        tool_content = item.get("content", "")
                        if tool_call_id:
                            tool_results.append({
                                "call_id": tool_call_id,
                                "output": tool_content,
                            })
                    # Handle item format (from Responses API output)
                    elif item.get("type") == "message" and item.get("role") == "user":
                        content = item.get("content", [])
                        if isinstance(content, list):
                            text_parts = [part.get("text", "") for part in content if part.get("type") == "output_text"]
                            user_content = " ".join(text_parts)
                    elif item.get("type") == "function_call_output":
                        call_id = item.get("call_id")
                        output = item.get("output", "")
                        if call_id:
                            tool_results.append({
                                "call_id": call_id,
                                "output": output,
                            })
            
            if not user_content:
                raise ValueError("No user input found in input data")
            
            a2a_message = create_text_message_object(role=Role.user, content=user_content)
            if tool_results:
                a2a_message.parts.append(Part(DataPart(data={TOOL_RESULTS_KEY: tool_results})))
            return a2a_message
        
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def convert_a2a_response_to_responses(task: Any, content: str = "", status: Any = None) -> Dict[str, Any]:
        """Convert A2A task response to Responses API format."""
        # Use provided status or get from task
        if status is None:
            status = getattr(task, 'status', None)
        
        # Extract content from task status message
        if not content and status:
            status_message = getattr(status, 'message', None) if hasattr(status, 'message') else status.get('message') if isinstance(status, dict) else None
            if status_message:
                if isinstance(status_message, dict):
                    parts = status_message.get('parts', [])
                else:
                    parts = getattr(status_message, 'parts', [])
                for part in parts:
                    part_data = part.root if hasattr(part, 'root') else part
                    if hasattr(part_data, 'kind') and part_data.kind == 'text':
                        if hasattr(part_data, 'text'):
                            content += part_data.text
        
        # Build output items array
        output_items = []
        
        # Add tool calls if present - create a task-like object for extract_tool_calls
        task_for_extraction = task
        if status and not hasattr(task, 'status'):
            # Create a wrapper object with status attribute
            class TaskWrapper:
                def __init__(self, task_obj, status_obj):
                    self._task = task_obj
                    self.status = status_obj
            task_for_extraction = TaskWrapper(task, status)
        
        tool_calls = extract_tool_calls(task_for_extraction)
        if tool_calls:
            for tc in tool_calls:
                output_items.append({
                    "id": f"fc_{uuid4().hex}",
                    "type": "function_call",
                    "call_id": tc.get("call_id", ""),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", {})
                })
        
        # Add message item
        if content or not tool_calls:
            output_items.append({
                "id": f"msg_{uuid4().hex}",
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": content
                }] if content else [],
                "role": "assistant"
            })
        
        response = {
            "id": f"resp_{uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "model": agent.model,
            "output": output_items
        }
        
        return response
    
    # Get agent card for A2A client
    agent_base_url = f"http://{http_host}:{port}"
    
    async def get_a2a_client() -> A2AClient:
        """Get A2A client for communicating with the agent."""
        async with httpx.AsyncClient() as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=agent_base_url,
            )
            agent_card = await resolver.get_agent_card()
            return A2AClient(httpx_client=httpx_client, agent_card=agent_card)
    
    @combined_app.post("/v1/responses")
    async def handle_responses(request: Request):
        """Handle /v1/responses endpoint - automatically executes tool calls."""
        body = await request.json()
        stream = body.get("stream", False)
        
        if stream:
            return StreamingResponse(
                handle_responses_streaming(body, agent_base_url, agent.context_id_to_environment_uri),
                media_type="text/event-stream"
            )
        else:
            return await handle_responses_non_streaming(body, agent_base_url, agent.context_id_to_environment_uri)
    
    async def handle_responses_non_streaming(
        body: Dict[str, Any],
        agent_base_url: str,
        context_id_to_environment_uri: Dict[str, str]
    ) -> Dict[str, Any]:
        """Handle non-streaming /v1/responses request."""
        # Responses API uses 'input' instead of 'messages'
        input_data = body.get("input")
        if input_data is None:
            raise HTTPException(status_code=400, detail="input is required")
        
        # Get context_id (use first available or generate one)
        context_id = None
        for cid in context_id_to_environment_uri.keys():
            context_id = cid
            break
        if not context_id:
            context_id = str(uuid4())
        
        environment_uri = context_id_to_environment_uri.get(context_id)
        if not environment_uri:
            raise HTTPException(status_code=400, detail=f"No environment found for context_id: {context_id}")
        
        async with httpx.AsyncClient() as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=agent_base_url)
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Convert Responses API input to A2A
            a2a_message = convert_responses_input_to_a2a(input_data)
            if context_id:
                a2a_message.context_id = context_id
            
            task_id = None
            final_content = ""
            
            # Process conversation loop until completion
            while True:
                # Send message
                send_request = SendMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(message=a2a_message)
                )
                response = await a2a_client.send_message(send_request)
                
                # Extract task from response - response is a Pydantic model
                # Try to get result from dict representation first (most reliable)
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else {}
                task_data = response_dict.get('result')
                
                if not task_data:
                    # Try direct attribute access as fallback
                    task_data = getattr(response, 'result', None)
                
                if not task_data:
                    raise HTTPException(status_code=500, detail="Failed to get result from response")
                
                # Convert to object-like access if it's a dict
                if isinstance(task_data, dict):
                    class TaskObj:
                        def __init__(self, d):
                            for k, v in d.items():
                                if isinstance(v, dict):
                                    setattr(self, k, TaskObj(v))
                                else:
                                    setattr(self, k, v)
                    task = TaskObj(task_data)
                else:
                    task = task_data
                
                if not task:
                    raise HTTPException(status_code=500, detail="Failed to get task from response")
                
                task_id = getattr(task, 'id', None) or task_id
                context_id = getattr(task, 'context_id', None) or context_id
                
                # Get status - handle both object and dict access
                status = getattr(task, 'status', None)
                if status is None:
                    status = {}
                if isinstance(status, dict):
                    state_value = status.get('state', {}).get('value') if isinstance(status.get('state'), dict) else status.get('state')
                else:
                    state_value = getattr(status, 'state', None)
                    if state_value:
                        state_value = getattr(state_value, 'value', None) or str(state_value)
                
                # Check if task is completed
                if state_value == "completed":
                    # Extract final content from status.message
                    status_message = getattr(status, 'message', None) if hasattr(status, 'message') else status.get('message') if isinstance(status, dict) else None
                    if status_message:
                        # Handle both object and dict access for message
                        if isinstance(status_message, dict):
                            parts = status_message.get('parts', [])
                        else:
                            parts = getattr(status_message, 'parts', [])
                        for part in parts:
                            part_data = part.root if hasattr(part, 'root') else part
                            if hasattr(part_data, 'kind') and part_data.kind == 'text':
                                if hasattr(part_data, 'text'):
                                    final_content += part_data.text
                    
                    # Also check task history for agent messages (important after tool execution)
                    task_history = getattr(task, 'history', None) if hasattr(task, 'history') else (task.get('history') if isinstance(task, dict) else None)
                    if task_history:
                        for msg in task_history:
                            # Handle both dict and object access
                            if isinstance(msg, dict):
                                msg_role = msg.get('role')
                                msg_parts = msg.get('parts', [])
                            else:
                                msg_role = getattr(msg, 'role', None)
                                msg_parts = getattr(msg, 'parts', [])
                            
                            if msg_role == 'agent' or str(msg_role) == 'agent':
                                for part in msg_parts:
                                    part_data = part.root if hasattr(part, 'root') else part
                                    if isinstance(part_data, dict):
                                        if part_data.get('kind') == 'text' and part_data.get('text'):
                                            final_content += part_data.get('text', '')
                                    elif hasattr(part_data, 'kind') and part_data.kind == 'text':
                                        if hasattr(part_data, 'text'):
                                            final_content += part_data.text
                    
                    break
                
                # Check if tool calls are needed
                if state_value == "input-required":
                    tool_calls = extract_tool_calls(task)
                    if tool_calls:
                        # Execute tools
                        tool_results = []
                        for tc in tool_calls:
                            tool_name = tc.get("name", "")
                            tool_args = tc.get("arguments", {})
                            call_id = tc.get("call_id", "")
                            
                            result = await execute_tool_via_mcp(tool_name, tool_args, environment_uri)
                            tool_results.append({
                                "call_id": call_id,
                                "name": tool_name,
                                "output": result,
                            })
                        
                        # Build tool result message and continue loop to get next response
                        a2a_message = build_tool_result_message(tool_results, task_id, context_id)
                        continue  # Continue outer loop to send tool results and get next response
                    else:
                        # No tool calls but input required - break
                        break
                
                # If working state, accumulate content (shouldn't happen in non-streaming, but handle it)
                if state_value == "working":
                    status_message = getattr(status, 'message', None) if hasattr(status, 'message') else status.get('message') if isinstance(status, dict) else None
                    if status_message:
                        # Handle both object and dict access for message
                        if isinstance(status_message, dict):
                            parts = status_message.get('parts', [])
                        else:
                            parts = getattr(status_message, 'parts', [])
                        for part in parts:
                            part_data = part.root if hasattr(part, 'root') else part
                            if hasattr(part_data, 'kind') and part_data.kind == 'text':
                                if hasattr(part_data, 'text'):
                                    final_content += part_data.text
                    # In non-streaming, working state means we should continue to get final response
                    continue
            
            # Convert to Responses API format - get final status from task
            final_status = getattr(task, 'status', None) if task else None
            return convert_a2a_response_to_responses(task, final_content, final_status)
    
    async def handle_responses_streaming(
        body: Dict[str, Any],
        agent_base_url: str,
        context_id_to_environment_uri: Dict[str, str]
    ):
        """Handle streaming /v1/responses request."""
        # Responses API uses 'input' instead of 'messages'
        input_data = body.get("input")
        if input_data is None:
            yield f"data: {json.dumps({'error': {'message': 'input is required'}})}\n\n"
            return
        
        # Get context_id
        context_id = None
        for cid in context_id_to_environment_uri.keys():
            context_id = cid
            break
        if not context_id:
            context_id = str(uuid4())
        
        environment_uri = context_id_to_environment_uri.get(context_id)
        if not environment_uri:
            yield f"data: {json.dumps({'error': {'message': f'No environment found for context_id: {context_id}'}})}\n\n"
            return
        
        async with httpx.AsyncClient() as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=agent_base_url)
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Convert Responses API input to A2A
            a2a_message = convert_responses_input_to_a2a(input_data)
            if context_id:
                a2a_message.context_id = context_id
            
            task_id = None
            accumulated_content = ""
            
            # Process conversation loop
            while True:
                # Send streaming message
                streaming_request = SendStreamingMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(message=a2a_message)
                )
                
                stream_response = a2a_client.send_message_streaming(streaming_request)
                
                tool_calls_found = False
                async for chunk in stream_response:
                    # Handle chunk - can be dict or object
                    chunk_dict = chunk.model_dump() if hasattr(chunk, 'model_dump') else (dict(chunk) if hasattr(chunk, '__dict__') else chunk)
                    event = chunk_dict.get('result') if isinstance(chunk_dict, dict) else (getattr(chunk, 'result', None) or chunk)
                    
                    # Handle both dict and object access
                    if isinstance(event, dict):
                        event_kind = event.get('kind')
                        event_status = event.get('status', {})
                        if isinstance(event_status, dict):
                            state_value = event_status.get('state', {}).get('value') if isinstance(event_status.get('state'), dict) else event_status.get('state')
                        else:
                            state_value = getattr(event_status, 'state', {}).get('value') if hasattr(event_status, 'state') else None
                    else:
                        event_kind = getattr(event, 'kind', None)
                        event_status = getattr(event, 'status', None)
                        if event_status:
                            state_obj = getattr(event_status, 'state', None)
                            state_value = getattr(state_obj, 'value', None) if state_obj else None
                        else:
                            state_value = None
                    
                    if event_kind == 'status-update' or (isinstance(event, dict) and event.get('kind') == 'status-update'):
                        task = event
                        
                        # Extract task_id and context_id
                        if isinstance(task, dict):
                            task_id = task.get('taskId') or task.get('task_id') or task_id
                            context_id = task.get('contextId') or task.get('context_id') or context_id
                        else:
                            task_id = getattr(task, 'taskId', None) or getattr(task, 'task_id', None) or task_id
                            context_id = getattr(task, 'contextId', None) or getattr(task, 'context_id', None) or context_id
                        
                        # Stream content updates
                        if state_value == "working":
                            # Get message from status
                            if isinstance(event_status, dict):
                                status_message = event_status.get('message')
                            else:
                                status_message = getattr(event_status, 'message', None)
                            
                            if status_message:
                                # Handle both dict and object access
                                if isinstance(status_message, dict):
                                    parts = status_message.get('parts', [])
                                else:
                                    parts = getattr(status_message, 'parts', [])
                                
                                for part in parts:
                                    part_data = part.root if hasattr(part, 'root') else part
                                    if hasattr(part_data, 'kind') and part_data.kind == 'text':
                                        if hasattr(part_data, 'text'):
                                            delta_content = part_data.text
                                            accumulated_content += delta_content
                                            chunk_data = {
                                                'id': f'resp_{uuid4().hex}',
                                                'object': 'response.delta',
                                                'created_at': int(time.time()),
                                                'model': agent.model,
                                                'output': [{
                                                    'type': 'message',
                                                    'delta': {
                                                        'content': [{
                                                            'type': 'output_text',
                                                            'text': delta_content
                                                        }]
                                                    }
                                                }]
                                            }
                                            yield f"data: {json.dumps(chunk_data)}\n\n"
                        
                        # Check if completed
                        if state_value == "completed":
                            chunk_data = {
                                'id': f'resp_{uuid4().hex}',
                                'object': 'response.delta',
                                'created_at': int(time.time()),
                                'model': agent.model,
                                'output': [{
                                    'type': 'message',
                                    'delta': {}
                                }]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        
                        # Check if tool calls needed
                        if state_value == "input-required":
                            tool_calls = extract_tool_calls(task)
                            if tool_calls:
                                tool_calls_found = True
                                # Execute tools
                                tool_results = []
                                for tc in tool_calls:
                                    tool_name = tc.get("name", "")
                                    tool_args = tc.get("arguments", {})
                                    call_id = tc.get("call_id", "")
                                    
                                    result = await execute_tool_via_mcp(tool_name, tool_args, environment_uri)
                                    tool_results.append({
                                        "call_id": call_id,
                                        "name": tool_name,
                                        "output": result,
                                    })
                                
                                # Build tool result message and continue outer loop
                                a2a_message = build_tool_result_message(tool_results, task_id, context_id)
                                break  # Break from inner loop, continue outer loop
                            else:
                                # No tool calls - finish
                                chunk_data = {
                                    'id': f'resp_{uuid4().hex}',
                                    'object': 'response.delta',
                                    'created_at': int(time.time()),
                                    'model': agent.model,
                                    'output': [{
                                        'type': 'message',
                                        'delta': {}
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                yield "data: [DONE]\n\n"
                                return
                
                # If we broke from inner loop due to tool calls, continue outer loop
                if tool_calls_found:
                    continue
                else:
                    # No more chunks and no tool calls - break
                    break
    
    # Run combined app (blocking)
    print(f"Starting Weather Assistant Agent on port {port}...")
    print(f"Environment mounted at: http://{http_host}:{port}/mcp")
    uvicorn.run(combined_app, host=host, port=port)


if __name__ == "__main__":
    main()
