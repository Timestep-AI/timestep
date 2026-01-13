# /// script
# dependencies = [
#   "a2a-sdk",
#   "mcp",
#   "httpx",
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
from a2a.client import ClientFactory, ClientConfig
from a2a.client.helpers import create_text_message_object
from a2a.types import TransportProtocol, Role
from mcp import ClientSession, types as mcp_types
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
from mcp.shared.context import RequestContext

# Server URLs
A2A_BASE_URL = os.getenv("A2A_URL", "http://localhost:8000")
MCP_URL = os.getenv("MCP_URL", "http://localhost:8080/mcp")

# Agent IDs
PERSONAL_ASSISTANT_ID = "00000000-0000-0000-0000-000000000000"
WEATHER_ASSISTANT_ID = "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"


def write_task(task: Any, agent_id: str) -> None:
    """Write task to tasks/ folder in proper A2A Task format."""
    tasks_dir = Path("tasks")
    tasks_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().isoformat().replace(":", "-")
    task_id_short = task.id[:8] if task.id else "unknown"
    agent_id_short = agent_id[:8] if agent_id else "unknown"
    task_file = tasks_dir / f"{timestamp}_{task_id_short}_{agent_id_short}.json"
    
    with open(task_file, "w") as f:
        json.dump(task.model_dump(mode="json", exclude_none=True), f, indent=2)
    print(f"\n[Saved task to {task_file}]", file=sys.stderr)


def extract_task_from_event(event: Any) -> Any:
    """Extract Task object from A2A event."""
    if isinstance(event, tuple):
        task, _ = event
        return task
    elif hasattr(event, 'id') and hasattr(event, 'status'):
        return event
    else:
        raise ValueError(f"Received non-Task event from Task-generating agent: {type(event)}")


def extract_final_message(task: Any) -> str:
    """Extract final message text from a completed task."""
    message_text = ""
    
    # Extract from task.status.message if available
    if task.status.message and task.status.message.parts:
        for part in task.status.message.parts:
            if hasattr(part, 'kind') and part.kind == 'text':
                if hasattr(part, 'text'):
                    message_text += part.text
            elif hasattr(part, 'root'):
                part_data = part.root
                if hasattr(part_data, 'kind') and part_data.kind == 'text':
                    if hasattr(part_data, 'text'):
                        message_text += part_data.text
    
    # Also check task history for agent messages
    if task.history:
        for msg in task.history:
            if msg.role == Role.agent or (hasattr(msg, 'role') and str(msg.role) == "agent"):
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'kind') and part.kind == 'text':
                            message_text += part.text
                elif hasattr(msg, 'content'):
                    message_text += msg.content
    
    return message_text.strip()


def extract_tool_calls(task: Any) -> Optional[List[Dict[str, Any]]]:
    """Extract tool calls from task status message or history."""
    # Check task.status.message parts first
    if task.status.message and task.status.message.parts:
        for part in task.status.message.parts:
            if hasattr(part, 'root'):
                part_data = part.root
                if hasattr(part_data, 'kind') and part_data.kind == 'data':
                    if hasattr(part_data, 'data') and isinstance(part_data.data, dict):
                        tool_calls = part_data.data.get("tool_calls")
                        if tool_calls:
                            return tool_calls
    
    # Fallback: check last agent message in history
    if task.history:
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
                                        return tool_calls
    
    return None


def parse_tool_call(tool_call: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
    """Parse tool call dict to extract tool name and arguments."""
    tool_name = tool_call.get("function", {}).get("name")
    tool_args_str = tool_call.get("function", {}).get("arguments")
    
    try:
        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str or {}
    except (json.JSONDecodeError, ValueError):
        tool_args = {}
    
    return tool_name, tool_args


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
        print(f"[ERROR: mcp_sampling_callback error: {e}]", file=sys.stderr)
        return mcp_types.ErrorData(
            code=mcp_types.INTERNAL_ERROR,
            message=f"Sampling error: {str(e)}",
        )


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call MCP tool using MCP Python SDK client."""
    try:
        async with streamable_http_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write, sampling_callback=mcp_sampling_callback) as session:
                await session.initialize()
                result: CallToolResult = await session.call_tool(tool_name, arguments)
                
                # CallToolResult.content is a list of TextContent objects
                if result.content:
                    text_parts = [item.text for item in result.content if isinstance(item, TextContent)]
                    return {"result": " ".join(text_parts)} if text_parts else {"result": None}
                return {"result": None}
    except Exception as e:
        return {"error": str(e)}

def extract_agent_id_from_uri(agent_uri: str) -> str:
    """Extract agent_id from agent_uri for use in write_task."""
    import re
    match = re.search(r'/agents/([^/\s]+)', agent_uri)
    return match.group(1) if match else "unknown"


async def process_message_stream(
    a2a_client: Any,
    message_obj: Any,
    agent_id: str,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> str:
    """Process a message stream, handling tool calls recursively."""
    final_message = ""
    
    async for event in a2a_client.send_message(message_obj):
        task = extract_task_from_event(event)
        write_task(task, agent_id)
        
        current_task_id = task.id or task_id
        current_context_id = task.context_id or context_id
        
        if task.status.state.value == "completed":
            final_message = extract_final_message(task)
            break
        
        if task.status.state.value == "input-required":
            tool_calls = extract_tool_calls(task)
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name, tool_args = parse_tool_call(tool_call)
                    result = await call_mcp_tool(tool_name, tool_args)
                    
                    tool_result_msg = create_text_message_object(
                        role="user",
                        content=json.dumps(result),
                    )
                    if current_task_id:
                        tool_result_msg.task_id = current_task_id
                    if current_context_id:
                        tool_result_msg.context_id = current_context_id
                    
                    # Recursively process tool result stream
                    result_message = await process_message_stream(
                        a2a_client, tool_result_msg, agent_id, current_task_id, current_context_id
                    )
                    if result_message:
                        final_message = result_message
                        break
    
    return final_message.strip() or "Task completed."


async def handle_agent_handoff(agent_uri: str, message: str) -> str:
    """Handle agent handoff by calling the A2A agent at agent_uri."""
    httpx_client = httpx.AsyncClient(timeout=60.0)
    config = ClientConfig(
        streaming=True,
        polling=False,
        httpx_client=httpx_client,
        supported_transports=[TransportProtocol.http_json],
    )
    
    a2a_client = None
    try:
        a2a_client = await ClientFactory.connect(agent_uri, client_config=config)
    except Exception as e:
        await httpx_client.aclose()
        return f"Error connecting to agent: {e}"
    
    try:
        message_obj = create_text_message_object(role="user", content=message)
        agent_id = extract_agent_id_from_uri(agent_uri)
        return await process_message_stream(a2a_client, message_obj, agent_id)
    finally:
        if a2a_client:
            try:
                await a2a_client.close()
            except Exception:
                pass
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
        message = create_text_message_object(role="user", content=initial_message)
        print(f"\n[DEBUG: Starting to send message to A2A server]", file=sys.stderr)
        
        async def process_with_output(a2a_client: Any, message_obj: Any, agent_id: str) -> None:
            """Process message stream and print output."""
            async for event in a2a_client.send_message(message_obj):
                task = extract_task_from_event(event)
                print(f"\n[DEBUG: Received task, id={getattr(task, 'id', 'NO_ID')}, type={type(task)}]", file=sys.stderr)
                write_task(task, agent_id)
                
                if task.id and task.id not in task_ids:
                    task_ids.append(task.id)
                
                # Print agent messages
                if task.history:
                    for msg in task.history:
                        if msg.role == Role.agent or (hasattr(msg, 'role') and str(msg.role) == "agent"):
                            if hasattr(msg, 'parts'):
                                for part in msg.parts:
                                    if hasattr(part, 'kind') and part.kind == 'text':
                                        print(part.text, end="", flush=True)
                            elif hasattr(msg, 'content'):
                                print(msg.content, end="", flush=True)
                
                if task.status.state.value == "completed":
                    print("\n[Task completed]")
                    break
                
                if task.status.state.value == "input-required":
                    tool_calls = extract_tool_calls(task)
                    if tool_calls:
                        for tool_call in tool_calls:
                            tool_name, tool_args = parse_tool_call(tool_call)
                            
                            print(f"\n[Calling tool: {tool_name}]")
                            result = await call_mcp_tool(tool_name, tool_args)
                            if tool_name != "handoff":
                                print(f"[Tool result: {result}]")
                            
                            tool_result_msg = create_text_message_object(
                                role="user",
                                content=json.dumps(result),
                            )
                            if task.id:
                                tool_result_msg.task_id = task.id
                            if task.context_id:
                                tool_result_msg.context_id = task.context_id
                            
                            # Recursively process tool result
                            await process_with_output(a2a_client, tool_result_msg, agent_id)
                            break
        
        await process_with_output(a2a_client, message, agent_id)
    except Exception as e:
        print(f"\n[Error in client loop: {e}]")
        raise
    
    finally:
        # Cleanup - close a2a_client before httpx_client
        if a2a_client:
            try:
                await a2a_client.close()
            except Exception:
                pass  # Ignore errors during cleanup
        await httpx_client.aclose()


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: uv run app/test_client.py <message>")
        sys.exit(1)
    
    message = " ".join(sys.argv[1:])
    await run_client_loop(message, agent_id=PERSONAL_ASSISTANT_ID)


if __name__ == "__main__":
    asyncio.run(main())
