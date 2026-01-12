# /// script
# dependencies = [
#   "fastmcp",
#   "mcp",
#   "httpx",
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
    Handoff tool that triggers sampling/complete.
    This uses MCP sampling to call the A2A server with a new agent.
    """
    # Use agent_uri directly to make HTTP request to client's sampling endpoint
    try:
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            # Format request according to FastMCP sampling handler pattern
            # See https://fastmcp.wiki/en/clients/sampling
            sampling_response = await http_client.post(
                agent_uri,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": {"text": message or "Please help with this task."}
                        }
                    ],
                    "params": {
                        "systemPrompt": f"Agent handoff request. Context ID: {context_id or 'none'}. Message: {message or 'none'}",
                        "maxTokens": 1000,
                    },
                    "context": {},
                },
            )
            sampling_response.raise_for_status()
            result = sampling_response.json()
            sampling_text = result.get("text", str(result))
        
        return {
            "handoff": True,
            "agent_uri": agent_uri,
            "context_id": context_id,
            "message": message,
            "sampling_response": sampling_text,
        }
    except Exception as e:
        return {
            "handoff": True,
            "agent_uri": agent_uri,
            "context_id": context_id,
            "message": message,
            "error": str(e),
        }


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
    port = int(os.getenv("MCP_PORT", "3001"))
    
    async def main():
        await mcp.run_async(transport="http", port=port, json_response=True)
    
    asyncio.run(main())
