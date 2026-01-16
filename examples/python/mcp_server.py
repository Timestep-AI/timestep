# /// script
# dependencies = [
#   "mcp",
#   "uvicorn",
# ]
# ///

"""
MCP Server using FastMCP from mcp python-sdk.
Handles tool execution and handoff sampling.
"""

import datetime
import uuid
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context

# Create FastMCP server (host/port will be configured in main if needed)
mcp = FastMCP("MCP Server")


# Tool implementations
@mcp.tool()
async def handoff(
    agent_uri: str,
    context_id: Optional[str] = None,
    message: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Handoff tool that uses MCP sampling to call another agent via the client.
    The client's sampling handler will invoke the A2A server for the target agent.
    """
    if not message:
        raise ValueError("Message is required for handoff")

    if not ctx:
        raise ValueError("Context not available for sampling")

    from mcp import types as mcp_types
    
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


@mcp.tool()
async def get_weather(location: str) -> Dict[str, Any]:
    """Get the current weather for a specific location."""
    # Return hardcoded weather data
    return {
        "location": location,
        "temperature": "72°F",
        "condition": "Sunny",
        "humidity": "65%"
    }


@mcp.tool()
async def plan_tasks(
    goal: str,
    tasks: List[Dict[str, Any]],
    output_format: str = "json",
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a structured execution plan with parallelizable tasks."""
    plan_id = str(uuid.uuid4())
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    normalized_tasks: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks or []):
        task_id = task.get("id") or f"task_{index + 1}"
        task_title = task.get("title") or task.get("name") or f"Task {index + 1}"
        task_instructions = task.get("instructions") or task.get("description") or ""
        depends_on = task.get("depends_on") or []
        if not isinstance(depends_on, list):
            depends_on = [str(depends_on)]
        execution = task.get("execution") if isinstance(task.get("execution"), dict) else {}
        if "mode" not in execution:
            execution["mode"] = "json"

        normalized_tasks.append(
            {
                "id": task_id,
                "title": task_title,
                "instructions": task_instructions,
                "depends_on": depends_on,
                "parallel_group": task.get("parallel_group"),
                "execution": execution,
                "metadata": task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            }
        )

    return {
        "plan": {
            "plan_id": plan_id,
            "title": title or goal,
            "goal": goal,
            "output_format": output_format,
            "created_at": created_at,
            "version": "1.0",
            "tasks": normalized_tasks,
            "metadata": metadata or {},
        }
    }


if __name__ == "__main__":
    import os
    import asyncio
    
    # FastMCP handles running the server with HTTP transport
    port = int(os.getenv("MCP_PORT", "8080"))
    
    # Update settings for host, port, and json_response
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = port
    mcp.settings.json_response = True
    
    async def main():
        # Use run_streamable_http_async for HTTP transport
        await mcp.run_streamable_http_async()
    
    asyncio.run(main())
