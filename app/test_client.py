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
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from a2a.client import ClientFactory, ClientConfig
from a2a.client.helpers import create_text_message_object
from a2a.types import TaskQueryParams
import uvicorn

# Server URLs
A2A_BASE_URL = os.getenv("A2A_URL", "http://localhost:8000")
MCP_URL = os.getenv("MCP_URL", "http://localhost:3001")
CLIENT_SAMPLING_PORT = int(os.getenv("CLIENT_SAMPLING_PORT", "3002"))
CLIENT_SAMPLING_URL = f"http://localhost:{CLIENT_SAMPLING_PORT}"

# Agent IDs
PERSONAL_ASSISTANT_ID = "00000000-0000-0000-0000-000000000000"
WEATHER_ASSISTANT_ID = "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"

# Global sampling request handler (no longer needed - handled synchronously)


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call MCP tool via HTTP (standard MCP endpoint)."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{MCP_URL}/tools/call",
                json={
                    "name": tool_name,
                    "arguments": arguments,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
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
    # Extract agent_id from query parameter (passed via agent_uri)
    # The agent_uri includes ?agent_id=... which we'll extract from context
    agent_id = None
    if context:
        agent_id = context.get("agent_id")
    
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
    
    # Extract system prompt from params
    system_prompt = params.get("systemPrompt") or params.get("system_prompt", "")
    
    # Extract context_id from system_prompt
    context_id = None
    if system_prompt:
        import re
        context_match = re.search(r"Context ID: ([^\s.]+)", system_prompt)
        if context_match:
            context_id = context_match.group(1) if context_match.group(1) != "none" else None
    
    # If we have agent_id, handle the sampling by calling A2A agent
    if agent_id:
        return await handle_mcp_sampling_internal(
            agent_id=agent_id,
            context_id=context_id,
            message=message_text,
        )
    else:
        return "Error: agent_id is required for sampling"


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


# Removed process_sampling_requests - we handle sampling synchronously now


async def handle_mcp_sampling_internal(
    agent_id: str,
    context_id: Optional[str],
    message: Optional[str],
) -> str:
    """
    Handle MCP sampling request by calling the appropriate A2A agent.
    This is called when MCP handoff tool uses ctx.sample().
    Returns the final assistant message text.
    """
    # Construct A2A URL with agent path
    agent_url = f"{A2A_BASE_URL}/agents/{agent_id}"
    
    # Create A2A client for the target agent
    httpx_client = httpx.AsyncClient(timeout=60.0)
    config = ClientConfig(streaming=False, polling=True, httpx_client=httpx_client)
    
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
        
        async for event in a2a_client.send_message(message_obj):
            if isinstance(event, tuple):
                task, update = event
                task_id = task.id
                
                # Collect assistant messages
                if task.messages:
                    for msg in task.messages:
                        if msg.role == "assistant":
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
                    # Check for tool calls
                    last_assistant = None
                    for msg in reversed(task.messages):
                        if msg.role == "assistant":
                            last_assistant = msg
                            break
                    
                    if last_assistant and hasattr(last_assistant, 'tool_calls') and last_assistant.tool_calls:
                        # Execute tool calls via MCP
                        for tool_call in last_assistant.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args_str = tool_call.function.arguments
                            
                            # Parse arguments
                            try:
                                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                            except:
                                tool_args = {}
                            
                            # Call MCP tool
                            result = await call_mcp_tool(tool_name, tool_args)
                            
                            # Send tool result back to A2A
                            tool_result_msg = create_text_message_object(
                                role="user",
                                content=json.dumps(result),
                            )
                            
                            # Continue with tool result
                            async for event2 in a2a_client.send_message(tool_result_msg):
                                if isinstance(event2, tuple):
                                    task2, update2 = event2
                                    # Update final message
                                    if task2.messages:
                                        for msg2 in task2.messages:
                                            if msg2.role == "assistant":
                                                if hasattr(msg2, 'parts'):
                                                    for part in msg2.parts:
                                                        if hasattr(part, 'kind') and part.kind == 'text':
                                                            final_message += part.text
                                                elif hasattr(msg2, 'content'):
                                                    final_message += msg2.content
                                    
                                    if task2.status.state.value == "completed":
                                        break
        
        # Poll for final updates if needed
        if task_id:
            while True:
                task = await a2a_client.get_task(TaskQueryParams(task_id=task_id))
                
                if task.status.state.value == "completed":
                    # Get final message
                    if task.messages:
                        for msg in task.messages:
                            if msg.role == "assistant":
                                if hasattr(msg, 'parts'):
                                    for part in msg.parts:
                                        if hasattr(part, 'kind') and part.kind == 'text':
                                            final_message += part.text
                                elif hasattr(msg, 'content'):
                                    final_message += msg.content
                    break
                
                await asyncio.sleep(0.5)
        
        return final_message.strip() or "Task completed."
    
    finally:
        await httpx_client.aclose()


async def run_client_loop(
    initial_message: str,
    agent_id: str = PERSONAL_ASSISTANT_ID,
) -> None:
    """Main client loop that orchestrates A2A and MCP (fully async)."""
    
    # Construct A2A URL with agent path
    agent_url = f"{A2A_BASE_URL}/agents/{agent_id}"
    
    # Create A2A client using ClientFactory
    httpx_client = httpx.AsyncClient(timeout=60.0)
    config = ClientConfig(streaming=False, polling=True, httpx_client=httpx_client)
    
    try:
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
        async for event in a2a_client.send_message(message):
            if isinstance(event, tuple):
                task, update = event
                task_id = task.id
                
                # Print assistant messages
                if task.messages:
                    for msg in task.messages:
                        if msg.role == "assistant":
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
                    # Check for tool calls in the last assistant message
                    last_assistant = None
                    for msg in reversed(task.messages):
                        if msg.role == "assistant":
                            last_assistant = msg
                            break
                    
                    if last_assistant and hasattr(last_assistant, 'tool_calls') and last_assistant.tool_calls:
                        # Execute tool calls via MCP
                        pending_events = []
                        
                        for tool_call in last_assistant.tool_calls:
                            tool_id = tool_call.id
                            tool_name = tool_call.function.name
                            tool_args_str = tool_call.function.arguments
                            
                            # Parse arguments
                            try:
                                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                            except:
                                tool_args = {}
                            
                            print(f"\n[Calling tool: {tool_name}]")
                            
                            # Check if it's a handoff
                            if tool_name == "handoff":
                                # Inject agent_uri (client sampling endpoint with agent_id) if not present
                                if "agent_uri" not in tool_args:
                                    # Extract agent_id from tool_args if present, otherwise use default
                                    agent_id = tool_args.get("agent_id", WEATHER_ASSISTANT_ID)
                                    tool_args["agent_uri"] = f"{CLIENT_SAMPLING_URL}/sampling/complete?agent_id={agent_id}"
                                    # Remove agent_id from tool_args since we're using agent_uri now
                                    if "agent_id" in tool_args:
                                        del tool_args["agent_id"]
                                
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
                        
                        # Send all tool results back to A2A
                        if pending_events:
                            # We need to send tool results as messages
                            # The A2A protocol expects tool results as part of the message
                            for event in pending_events:
                                tool_result_msg = create_text_message_object(
                                    role="user",
                                    content=event["content"],
                                )
                                
                                # Continue the loop with tool result
                                async for event2 in a2a_client.send_message(tool_result_msg):
                                    if isinstance(event2, tuple):
                                        task2, update2 = event2
                                        
                                        # Print assistant messages
                                        if task2.messages:
                                            for msg in task2.messages:
                                                if msg.role == "assistant":
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
        
        # Poll for final updates if needed
        if task_id:
            while True:
                task = await a2a_client.get_task(TaskQueryParams(task_id=task_id))
                
                if task.status.state.value == "completed":
                    print("\n[Task completed]")
                    break
                
                await asyncio.sleep(0.5)
    
    finally:
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
        port=CLIENT_SAMPLING_PORT,
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
            # Cleanup
            server.should_exit = True
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
    
    await run_with_sampling()


if __name__ == "__main__":
    asyncio.run(main())
