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
    
    print(f"Handoff agent_uri: {agent_uri}")
    print(f"Handoff context_id: {context_id}")
    print(f"Handoff message: {message}")
    print(f"Handoff ctx: {ctx}")

    # Use session.create_message() to request LLM sampling
    # The client's sampling handler will invoke the A2A server
    from mcp import types as mcp_types
    
    # Create a sampling message from the text
    sampling_message = mcp_types.SamplingMessage(
        role="user",
        content=mcp_types.TextContent(type="text", text=message)
    )
    
    # Request sampling from the client
    # Pass agent_uri in metadata so the client's sampling handler can extract it
    result = await ctx.session.create_message(
        messages=[sampling_message],
        max_tokens=1000,
        metadata={"agent_uri": agent_uri}
    )
    
    # Extract the response text from the result
    # result.content is a single TextContent object (not a list)
    response_text = ""
    if hasattr(result, 'content') and result.content:
        if isinstance(result.content, mcp_types.TextContent):
            response_text = result.content.text
        elif hasattr(result.content, 'text'):
            response_text = result.content.text
        else:
            # Fallback: try to convert to string
            response_text = str(result.content)
    
    # Return dict with response
    return {"response": response_text.strip()}


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
