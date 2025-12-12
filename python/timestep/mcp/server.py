"""MCP server implementation for Timestep tools."""

import contextlib
import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import mcp.types as types
from a2a.client import A2AClient
from a2a.types import Message
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

# Configure logging
logger = logging.getLogger(__name__)


def get_server() -> Server:
    """Create and configure the MCP server with tools."""
    app = Server("timestep-mcp-server")

    @app.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.ContentBlock]:
        """Handle tool calls."""
        if name == "get_weather":
            city = arguments.get("city", "Unknown")
            # Simple mock implementation - in production this would call a real weather API
            return [
                types.TextContent(
                    type="text",
                    text=f"The weather in {city} is sunny",
                )
            ]
        elif name == "handoff":
            message = arguments.get("message", "")
            agent_id = arguments.get("agentId", "")
            
            if not agent_id:
                return [
                    types.TextContent(
                        type="text",
                        text="Error: agentId parameter is required for handoff tool.",
                    )
                ]
            
            try:
                # Get A2A server URL from environment, default to localhost:8080
                a2a_server_url = os.getenv("A2A_SERVER_URL", "http://localhost:8080")
                agent_card_url = f"{a2a_server_url}/agents/{agent_id}/.well-known/agent-card.json"
                
                # Create A2A client from agent card URL
                client = await A2AClient.from_card_url(agent_card_url)
                
                # Create A2A message
                a2a_message: Message = Message(
                    kind="message",
                    role="user",
                    message_id=str(uuid.uuid4()),
                    parts=[
                        {
                            "kind": "text",
                            "text": message,
                        }
                    ],
                    context_id=str(uuid.uuid4()),  # Target agent gets its own isolated context
                )
                
                # Send message and wait for response
                response = await client.send_message(
                    message=a2a_message,
                    configuration={
                        "waitForCompletion": True,  # Wait for task to complete
                    },
                )
                
                # Handle error response
                if "error" in response:
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Error from A2A agent: {response.get('error', {}).get('message', 'Unknown error')}",
                        )
                    ]
                
                # Extract text from response
                response_text = ""
                if "result" in response:
                    result = response["result"]
                    if isinstance(result, dict):
                        if result.get("kind") == "task":
                            # Extract text from task status message
                            status = result.get("status", {})
                            status_message = status.get("message", {})
                            if status_message and "parts" in status_message:
                                for part in status_message["parts"]:
                                    if part.get("kind") == "text" and part.get("text"):
                                        response_text += part["text"]
                            
                            # If no status message, check history for agent messages
                            if not response_text and "history" in result:
                                for msg in result["history"]:
                                    if msg.get("role") == "agent" and "parts" in msg:
                                        for part in msg["parts"]:
                                            if part.get("kind") == "text" and part.get("text"):
                                                response_text += part["text"]
                        elif result.get("kind") == "message":
                            # Extract text from message parts
                            if "parts" in result:
                                for part in result["parts"]:
                                    if part.get("kind") == "text" and part.get("text"):
                                        response_text += part["text"]
                
                return [
                    types.TextContent(
                        type="text",
                        text=response_text or "Handoff completed but no response text available.",
                    )
                ]
            except Exception as error:
                error_message = str(error) if error else "Unknown error"
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error during handoff: {error_message}",
                    )
                ]
        
        raise ValueError(f"Unknown tool: {name}")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """List available tools."""
        return [
            types.Tool(
                name="get_weather",
                description="Returns weather info for the specified city.",
                inputSchema={
                    "type": "object",
                    "required": ["city"],
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city name.",
                        }
                    },
                },
            ),
            types.Tool(
                name="handoff",
                description="Hand off a message to another agent via A2A protocol. The target agent will have its own isolated context and task id. Requires agentId parameter to specify which agent to hand off to.",
                inputSchema={
                    "type": "object",
                    "required": ["message", "agentId"],
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send to the target agent.",
                        },
                        "agentId": {
                            "type": "string",
                            "description": "The ID of the agent to hand off to (e.g., 'weather-assistant', 'personal-assistant').",
                        },
                    },
                },
            ),
        ]

    return app


def create_server(
    host: str = "0.0.0.0",
    port: int = 3000,
) -> Starlette:
    """Create and configure the Starlette app with MCP endpoints.

    Args:
        host: Host address to bind to.
        port: Port number to listen on.

    Returns:
        Configured Starlette application.
    """
    app_instance = get_server()

    # Create session manager for Streamable HTTP
    session_manager = StreamableHTTPSessionManager(
        app=app_instance,
        event_store=None,  # No event store for stateless server
        json_response=False,
    )

    # ASGI handler for streamable HTTP connections
    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for managing session manager lifecycle."""
        async with session_manager.run():
            logger.info("MCP server started with StreamableHTTP session manager!")
            try:
                yield
            finally:
                logger.info("MCP server shutting down...")

    # Create an ASGI application using the transport
    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    # Wrap ASGI application with CORS middleware
    starlette_app = CORSMiddleware(
        starlette_app,
        allow_origins=["*"],  # Allow all origins - adjust as needed for production
        allow_methods=["GET", "POST", "DELETE"],  # MCP streamable HTTP methods
        expose_headers=["Mcp-Session-Id"],
    )

    return starlette_app


def run_server(
    host: str = "0.0.0.0",
    port: int = 3000,
) -> None:
    """Run the MCP server.

    Args:
        host: Host address to bind to.
        port: Port number to listen on.
    """
    import uvicorn

    app = create_server(host, port)
    port_num = int(os.environ.get("MCP_PORT", port))
    host_str = os.environ.get("MCP_HOST", host)

    logger.info(f"Starting Timestep MCP server on {host_str}:{port_num}")
    logger.info(f"MCP endpoint: http://{host_str}:{port_num}/mcp")

    uvicorn.run(app, host=host_str, port=port_num)


