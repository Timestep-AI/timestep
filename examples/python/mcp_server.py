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

from typing import Any, Dict, Optional

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
