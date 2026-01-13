# /// script
# dependencies = [
#   "fastmcp",
#   "mcp",
#   "httpx",
#   "uvicorn",
# ]
# ///

"""
MCP Server using FastMCP from mcp python-sdk.
Handles tool execution and handoff sampling.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp.server.context import Context
else:
    # Context will be available at runtime from FastMCP
    Context = Any

from fastmcp import FastMCP

# Create FastMCP server
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
    if not ctx:
        raise ValueError("Context not available for sampling")
    
    if not message:
        raise ValueError("Message is required for handoff")

    print(f"Handoff agent_uri: {agent_uri}")
    print(f"Handoff context_id: {context_id}")
    print(f"Handoff message: {message}")
    print(f"Handoff ctx: {ctx}")

    # Use FastMCP's ctx.sample() to request LLM sampling
    # Pass agent_uri in context so the client's sampling handler can extract agent_id
    response = await ctx.sample(
        messages=[message],
        context={"agent_uri": agent_uri}
    )
    
    # Extract the response text
    response_text = response.text.strip() if hasattr(response, 'text') else str(response)
    
    # Return dict with response
    return {"response": response_text}


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
    
    async def main():
        # FastMCP's run_async doesn't support host parameter directly
        # But we can monkey-patch or use environment variable
        # Actually, let's just use run_async and see if it respects host binding
        # If not, we'll need to check FastMCP source or use a workaround
        # For now, let's try using the original method but check FastMCP docs
        # Actually, FastMCP might bind to 0.0.0.0 by default in some versions
        # Let's check if we can pass host through run_async
        try:
            # Try with host parameter (might not be supported)
            await mcp.run_async(transport="http", host="0.0.0.0", port=port, json_response=True)
        except TypeError:
            # If host not supported, use run_async without it
            # FastMCP might bind to 0.0.0.0 by default, or we need another approach
            await mcp.run_async(transport="http", port=port, json_response=True)
    
    asyncio.run(main())
