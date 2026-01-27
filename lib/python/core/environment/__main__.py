"""Environment class - MCP Server (extends FastMCP) that provides system prompt and tools."""

import os
import asyncio
from typing import Dict, Optional, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
import mcp.types as mcp_types


class Environment(FastMCP):
    """Environment is an MCP Server that provides system prompt and tools for an agent.
    
    The system prompt is stored as a FastMCP prompt, and tools are registered
    using FastMCP's @tool() decorator. Each agent has its own environment.
    """
    
    def __init__(
        self,
        environment_id: str,
        context_id: str,  # Maps to A2A context_id
        agent_id: str,    # Which agent this environment is for
        human_in_loop: bool = False,  # Enable MCP Elicitation/Sampling
        enable_handoff: bool = True,  # Enable handoff tool
    ):
        super().__init__(f"Environment-{environment_id}")
        self.environment_id = environment_id
        self.context_id = context_id
        self.agent_id = agent_id
        self.human_in_loop = human_in_loop
        self.enable_handoff = enable_handoff
        
        # Default system prompt (can be overridden with @environment.prompt() decorator)
        @self.prompt()
        def system_prompt(agent_name: str) -> str:
            """System prompt for the agent."""
            return f"You are {agent_name}."
        
        # Register handoff tool conditionally based on enable_handoff parameter
        if enable_handoff:
            @self.tool()
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
        
        
        
    
    async def start(self, port: int = 8080, host: str = "0.0.0.0") -> str:
        """Start MCP server and return MCP URI.
        
        Args:
            port: Port to run the MCP server on
            host: Host to bind to
            
        Returns:
            MCP URI (e.g., "http://localhost:8080/mcp")
        """
        self.settings.host = host
        self.settings.port = port
        self.settings.json_response = True
        
        # Start server in background
        import asyncio
        asyncio.create_task(self.run_streamable_http_async())
        
        # Give server a moment to start
        await asyncio.sleep(0.1)
        
        # Return MCP URI
        return f"http://{host}:{port}/mcp"
    
    def run(self, transport: str = "http"):
        """Run the MCP server (blocking).
        
        Args:
            transport: Transport type - "http" or "stdio"
        """
        if transport == "stdio":
            # Use stdio transport (for subprocess execution)
            # FastMCP's run() method handles stdio synchronously
            super().run(transport="stdio")
        else:
            # Use HTTP transport
            import asyncio
            asyncio.run(self.run_streamable_http_async())
    
    async def run_async(self, transport: str = "http"):
        """Run the MCP server asynchronously (non-blocking).
        
        Args:
            transport: Transport type - "http" or "stdio"
        """
        if transport == "stdio":
            # For stdio, use the synchronous run() in a thread
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, super().run, "stdio")
        else:
            await self.run_streamable_http_async()
