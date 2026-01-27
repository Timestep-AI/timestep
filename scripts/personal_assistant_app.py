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

"""Personal Assistant Agent - A2A Server.

This script runs a Personal Assistant Agent that connects to an Environment
via MCP to get system prompts and tools. It can hand off tasks to other agents.
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

# Now we can import Agent, Environment, and ResponsesAPI
from timestep.core import Agent, Environment, ResponsesAPI
from timestep.utils.message_helpers import (
    extract_user_text_and_tool_results,
    build_tool_result_message,
    TOOL_CALLS_KEY,
    TOOL_RESULTS_KEY,
)
from timestep.utils.event_helpers import extract_event_data, extract_task_from_tuple
from a2a.client import ClientFactory
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
from mcp.server.fastmcp.server import Context
from mcp.shared.context import RequestContext
import mcp.types as mcp_types


def main():
    """Run the Personal Assistant Agent."""
    # Get port from environment variable or use default
    port = int(os.getenv("PERSONAL_AGENT_PORT", "9999"))
    host = "0.0.0.0"
    http_host = "localhost" if host == "0.0.0.0" else host
    
    # Set A2A_BASE_URL environment variable for agent card generation
    # This ensures the agent card has the correct URL
    # Always set it to match the port we're actually running on
    os.environ["A2A_BASE_URL"] = f"http://{http_host}:{port}"
    
    # Create Environment instance
    environment = Environment(
        environment_id="personal-assistant-env",
        context_id="personal-context",
        agent_id="personal-assistant",
        human_in_loop=False,
    )
    
    # Add handoff tool to the environment
    @environment.tool()
    async def handoff(
        agent_uri: str,
        context_id: Optional[str] = None,
        message: Optional[str] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """Handoff tool that uses MCP sampling to call another agent via the client.
        
        Args:
            agent_uri: The full base URL of the target agent (e.g., "http://localhost:10000").
                      This must be the complete URL where the agent's A2A endpoint is available.
            context_id: Optional context ID for the handoff.
            message: The message/question to send to the target agent.
        
        The client's sampling handler will invoke the A2A server for the target agent.
        For the weather assistant, use agent_uri="http://localhost:10000"
        """
        if not message:
            raise ValueError("Message is required for handoff")
        
        if not ctx:
            raise ValueError("Context not available for sampling")
        
        sampling_message = mcp_types.SamplingMessage(
            role="user",
            content=mcp_types.TextContent(type="text", text=message)
        )
        
        result = await ctx.session.create_message(
            messages=[sampling_message],
            max_tokens=1000,
            metadata={"agent_uri": agent_uri}
        )
        
        # result.content is a TextContent object
        return {"response": result.content.text.strip()}
    
    # Update system prompt for personal assistant
    @environment.prompt()
    def system_prompt(agent_name: str) -> str:
        """System prompt for the personal assistant agent."""
        return f"""You are {agent_name}, a helpful personal assistant. 
You can help with various tasks. For weather-related questions, use the handoff tool 
to delegate to the weather assistant agent.

IMPORTANT: When using the handoff tool, you MUST use the exact agent_uri: http://localhost:10000
Do NOT use placeholder values like weather_agent or weather_service. 
The agent_uri must be the full URL: http://localhost:10000

Example handoff tool call:
- agent_uri: http://localhost:10000
- message: What is the current weather in Oakland?"""
    
    # Get MCP app from environment
    mcp_app = environment.streamable_http_app()
    
    # Create agent
    agent = Agent(
        agent_id="personal-assistant",
        name="Personal Assistant",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        context_id_to_environment_uri={
            "personal-context": f"http://{http_host}:{port}/mcp"
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
        title="Personal Assistant Agent",
        lifespan=lifespan,
    )
    
    # Include all routes from the agent's FastAPI app
    for route in fastapi_app.routes:
        combined_app.routes.append(route)
    
    # Include all routes from the MCP app
    for route in mcp_app.routes:
        combined_app.routes.append(route)
    
    # Get agent base URL for ResponsesAPI
    agent_base_url = f"http://{http_host}:{port}"
    
    # Helper functions for handoff (specific to personal assistant)
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
    
    def extract_final_message(task: Any) -> str:
        """Extract final message text from a completed task.
        
        Only extracts the final completed message, not incremental updates.
        """
        message_text = ""
        
        # First, try to extract from task.status.message (this is the final completed message)
        status = getattr(task, 'status', None) if hasattr(task, 'status') else (task.get('status') if isinstance(task, dict) else None)
        if status:
            # Get message from status - handle both dict and object access
            if isinstance(status, dict):
                status_message = status.get('message')
            else:
                status_message = getattr(status, 'message', None)
            
            if status_message:
                # Get parts from message - handle both dict and object access
                if isinstance(status_message, dict):
                    parts = status_message.get('parts', [])
                else:
                    parts = getattr(status_message, 'parts', [])
                
                for part in parts:
                    part_data = part.root if hasattr(part, 'root') else part
                    if isinstance(part_data, dict):
                        if part_data.get('kind') == 'text' and part_data.get('text'):
                            message_text += part_data.get('text', '')
                    elif hasattr(part_data, 'kind') and part_data.kind == 'text':
                        if hasattr(part_data, 'text'):
                            message_text += part_data.text
        
        # If we got text from status.message, return it (this is the final message)
        if message_text.strip():
            return message_text.strip()
        
        # Otherwise, check task history for the LAST agent message (not all of them)
        # This handles cases where status.message might not be set
        if isinstance(task, dict):
            task_history = task.get('history', [])
        else:
            task_history = getattr(task, 'history', []) if hasattr(task, 'history') else []
        
        if task_history:
            # Iterate in reverse to find the last agent message
            for msg in reversed(task_history):
                # Handle both dict and object access for message
                if isinstance(msg, dict):
                    msg_role = msg.get('role')
                    msg_parts = msg.get('parts', [])
                    msg_content = msg.get('content', '')
                else:
                    msg_role = getattr(msg, 'role', None)
                    msg_parts = getattr(msg, 'parts', []) if hasattr(msg, 'parts') else []
                    msg_content = getattr(msg, 'content', '') if hasattr(msg, 'content') else ''
                
                if msg_role == Role.agent or str(msg_role) == "agent":
                    # Extract text from parts - only from this last agent message
                    for part in msg_parts:
                        part_data = part.root if hasattr(part, 'root') else part
                        if isinstance(part_data, dict):
                            if part_data.get('kind') == 'text' and part_data.get('text'):
                                message_text += part_data.get('text', '')
                        elif hasattr(part_data, 'kind') and part_data.kind == 'text':
                            if hasattr(part_data, 'text'):
                                message_text += part_data.text
                    # Also check for direct content if no parts
                    if not msg_parts and msg_content:
                        message_text += str(msg_content)
                    
                    # Found the last agent message, return it (don't accumulate from earlier messages)
                    break
        
        return message_text.strip()
    
    def extract_tool_output(result: Dict[str, Any]) -> str:
        """Extract text output from tool result dict.
        
        Handles:
        - {"result": "text"} -> "text"
        - {"error": "error message"} -> "error message"
        - Other dicts -> JSON stringified
        - Strings -> as-is
        """
        if isinstance(result, dict):
            if "result" in result:
                return str(result["result"])
            elif "error" in result:
                return str(result["error"])
            else:
                return json.dumps(result)
        return str(result)
    
    def build_tool_result_message(
        tool_results: List[Dict[str, Any]],
        task_id: Optional[str],
        context_id: Optional[str],
    ) -> Any:
        """Build a user message carrying tool results via DataPart."""
        tool_result_msg = create_text_message_object(role=Role.user, content="")
        if task_id:
            tool_result_msg.task_id = task_id
        if context_id:
            tool_result_msg.context_id = context_id
        # DataPart tool_results maps to OpenAI tool messages in the A2A server.
        tool_result_msg.parts.append(Part(DataPart(data={TOOL_RESULTS_KEY: tool_results})))
        return tool_result_msg
    
    async def handle_agent_handoff(agent_uri: str, message: str) -> str:
        """Handle agent handoff by calling the A2A agent at agent_uri."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Use ClientFactory (supports JSON-RPC by default)
        a2a_client = await ClientFactory.connect(agent_uri)
        
        try:
            message_obj = create_text_message_object(role="user", content=message)
            
            # Process message using ClientFactory (non-streaming, polling mode)
            final_message = ""
            task_id = None
            context_id = None
            
            # Process message stream until completion
            while True:
                # Send message - Client.send_message returns an async generator
                # Client.send_message expects just the message object, not SendMessageRequest
                
                # Process message stream
                task = None
                state_value = None
                tool_calls = None
                async for event in a2a_client.send_message(message_obj):
                    # Extract Task from tuple (first element contains full Task object)
                    task = extract_task_from_tuple(event)
                    
                    # Extract TaskStatusUpdateEvent for state information (second element)
                    event_data = extract_event_data(event)
                    
                    if task:
                        task_id = getattr(task, 'id', None) or task_id
                        context_id = getattr(task, 'context_id', None) or context_id
                    
                    if not task:
                        logger.warning("Handoff: No task extracted from event, continuing...")
                        continue
                    
                    # Get status from Task object
                    status = getattr(task, 'status', None)
                    if not status:
                        logger.warning("Handoff: Task has no status, continuing...")
                        continue
                    
                    # Extract state from TaskStatusUpdateEvent if available, otherwise from Task
                    if event_data and hasattr(event_data, 'status'):
                        event_status = getattr(event_data, 'status', None)
                        if event_status:
                            if hasattr(event_status, 'state'):
                                state_obj = getattr(event_status, 'state', None)
                                if state_obj:
                                    state_value = getattr(state_obj, 'value', None)
                    else:
                        # Fallback to Task status
                        if isinstance(status, dict):
                            state_value = status.get('state', {}).get('value') if isinstance(status.get('state'), dict) else status.get('state')
                        else:
                            state_obj = getattr(status, 'state', None)
                            state_value = getattr(state_obj, 'value', None) if state_obj else None
                    
                    if state_value == "completed":
                        final_message = extract_final_message(task)
                        break
                    
                    if state_value == "input-required":
                        tool_calls = extract_tool_calls(task)
                        if tool_calls:
                            # Execute tools via MCP
                            tool_results = []
                            for tc in tool_calls:
                                tool_name = tc.get("name", "")
                                tool_args = tc.get("arguments", {})
                                call_id = tc.get("call_id", "")
                                
                                # Construct MCP endpoint URI from agent_uri
                                weather_env_uri = f"{agent_uri}/mcp"
                                
                                result = await execute_tool_via_mcp(tool_name, tool_args, weather_env_uri)
                                # Extract text from result dict before passing to build_tool_result_message
                                output_text = extract_tool_output(result)
                                
                                tool_results.append({
                                    "call_id": call_id,
                                    "name": tool_name,
                                    "output": output_text,
                                })
                            
                            # Build tool result message and send it
                            tool_result_msg = build_tool_result_message(tool_results, task_id, context_id)
                            message_obj = tool_result_msg
                            # Break from inner loop to continue outer loop with new message
                            break
                        else:
                            logger.warning("Handoff: input-required but no tool calls found")
                            break
                
                # If we completed, exit outer loop
                if state_value == "completed":
                    break
                
                # If we broke from the loop due to tool calls, continue outer loop
                if state_value == "input-required" and tool_calls:
                    continue
                
                # Otherwise, exit
                break
            
            return final_message.strip() or "Task completed."
        except Exception as e:
            logger.error(f"Error during handoff: {str(e)}", exc_info=True)
            return f"Error during handoff: {str(e)}"
    
    async def mcp_sampling_callback(
        context: RequestContext["ClientSession", Any],
        params: mcp_types.CreateMessageRequestParams,
    ) -> mcp_types.CreateMessageResult | mcp_types.CreateMessageResultWithTools | mcp_types.ErrorData:
        """MCP sampling callback that handles sampling requests from the MCP server."""
        try:
            # Extract agent_uri from metadata (passed by handoff tool)
            agent_uri = (params.metadata or {}).get("agent_uri")
            if not agent_uri:
                return mcp_types.ErrorData(
                    code=mcp_types.INVALID_PARAMS,
                    message="agent_uri is required for sampling",
                )
            
            # Extract message text from params.messages (SamplingMessage objects with TextContent)
            message_text = params.messages[0].content.text if params.messages else "Please help with this task."
            
            result_text = await handle_agent_handoff(agent_uri=agent_uri, message=message_text)
            
            return mcp_types.CreateMessageResult(
                role="assistant",
                content=mcp_types.TextContent(type="text", text=result_text),
                model="a2a-agent"
            )
        except Exception as e:
            return mcp_types.ErrorData(
                code=mcp_types.INTERNAL_ERROR,
                message=f"Sampling error: {str(e)}",
            )
    
    async def execute_tool_via_mcp(tool_name: str, arguments: Dict[str, Any], environment_uri: str) -> Dict[str, Any]:
        """Execute a single tool via MCP (for handoff)."""
        try:
            async with streamable_http_client(environment_uri) as (read, write, _):
                async with ClientSession(read, write, sampling_callback=mcp_sampling_callback) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    
                    if result.content:
                        text_parts = [item.text for item in result.content if isinstance(item, TextContent)]
                        if text_parts:
                            result_text = " ".join(text_parts)
                            
                            # Special handling for handoff tool: extract response from nested structure
                            if tool_name == "handoff":
                                try:
                                    # Try to parse as JSON to extract "response" field
                                    parsed = json.loads(result_text)
                                    if isinstance(parsed, dict) and "response" in parsed:
                                        result_text = parsed["response"]
                                except (json.JSONDecodeError, KeyError, TypeError):
                                    # If parsing fails, use the text as-is
                                    pass
                            
                            return {"result": result_text}
                        return {"result": None}
                    return {"result": None}
        except Exception as e:
            return {"error": str(e)}
    
    # Create ResponsesAPI instance
    responses_api = ResponsesAPI(
        agent=agent,
        agent_base_url=agent_base_url,
        context_id_to_environment_uri=agent.context_id_to_environment_uri,
        sampling_callback=mcp_sampling_callback,
    )
    
    # Mount ResponsesAPI routes
    for route in responses_api.fastapi_app.routes:
        combined_app.routes.append(route)
    
    # All /v1/responses endpoint code has been moved to ResponsesAPI
    # The endpoint is now registered via responses_api.fastapi_app above
    
    # Run combined app (blocking)
    print(f"Starting Personal Assistant Agent on port {port}...")
    print(f"Environment mounted at: http://{http_host}:{port}/mcp")
    uvicorn.run(combined_app, host=host, port=port)


if __name__ == "__main__":
    main()
