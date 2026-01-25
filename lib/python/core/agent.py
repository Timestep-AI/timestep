"""Agent class - A2A Server that contains Loop internally."""

import os
from contextlib import asynccontextmanager
from typing import Dict, Optional
from fastapi import FastAPI
import uvicorn
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, TransportProtocol

from timestep.core.loop import Loop
from timestep.core.environment import Environment


class Agent:
    """Agent is an A2A Server that contains Loop (AgentExecutor) internally.
    
    The Agent:
    1. Contains Loop (AgentExecutor) internally
    2. Exposes agent URI via A2A protocol
    3. Responds to A2A requests
    4. Maps context_id to environment URI
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        model: str = "gpt-4o-mini",
        context_id_to_environment_uri: Optional[Dict[str, str]] = None,
        human_in_loop: bool = False,
    ):
        self.agent_id = agent_id
        self.name = name
        self.model = model
        self.context_id_to_environment_uri = context_id_to_environment_uri or {}
        self.human_in_loop = human_in_loop
        
        # Create default Environment instance
        self.default_environment = Environment(
            environment_id=f"{agent_id}-env",
            context_id="default",  # Default context_id
            agent_id=agent_id,
            human_in_loop=human_in_loop,
        )
        
        # Loop (AgentExecutor) is inside Agent
        self.loop = Loop(
            agent_id=agent_id,
            model=model,
            context_id_to_environment_uri=self.context_id_to_environment_uri,
            human_in_loop=human_in_loop,
        )
        
        # Create A2A server with Loop as AgentExecutor
        self.handler = DefaultRequestHandler(
            agent_executor=self.loop,
            task_store=InMemoryTaskStore(),
        )
        
        # Create agent card
        self.agent_card = self._create_agent_card()
        
        # Create A2A JSON-RPC app (primary)
        # A2AFastAPIApplication handles JSON-RPC requests at / (POST)
        # This is what A2AClient uses (JSON-RPC transport)
        self.app = A2AFastAPIApplication(
            agent_card=self.agent_card,
            http_handler=self.handler,
        )
        
        # Build A2A app - this returns a FastAPI app with:
        # - Agent card endpoint at /.well-known/agent-card.json
        # - JSON-RPC endpoint at / (POST)
        self.fastapi_app = self.app.build()
        
        # Get MCP app - streamable_http_app() returns a Starlette app with route at /mcp
        # Include routes directly so /mcp stays as /mcp (not /mcp/mcp)
        mcp_app = self.default_environment.streamable_http_app()
        for route in mcp_app.routes:
            self.fastapi_app.routes.append(route)
        
        # Initialize MCP task group on startup (required for streamable HTTP)
        # This is what run_streamable_http_async() does internally
        @self.fastapi_app.on_event("startup")
        async def startup_event():
            import anyio
            tg = anyio.create_task_group()
            await tg.__aenter__()
            self._mcp_task_group = tg
            
            # Set task group on session manager
            session_manager = self.default_environment.session_manager
            if hasattr(session_manager, '_task_group') and session_manager._task_group is None:
                session_manager._task_group = tg
        
        @self.fastapi_app.on_event("shutdown")
        async def shutdown_event():
            if hasattr(self, '_mcp_task_group') and self._mcp_task_group:
                try:
                    await self._mcp_task_group.__aexit__(None, None, None)
                except Exception:
                    pass
    
    def _create_agent_card(self) -> AgentCard:
        """Create an agent card for this agent."""
        base_url = os.getenv("A2A_BASE_URL", "http://localhost:9999")
        return AgentCard(
            name=self.name,
            version="1.0.0",
            description=f"{self.name} agent",
            url=f"{base_url}",
            preferred_transport=TransportProtocol.http_json,
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[AgentSkill(
                id=self.agent_id,
                name=self.name,
                description=f"{self.name} agent",
                tags=[],
            )],
        )
    
    async def start(self, port: int = 9999, host: str = "0.0.0.0") -> str:
        """Start A2A server and return agent URI.
        
        Args:
            port: Port to run the A2A server on
            host: Host to bind to
            
        Returns:
            Agent URI (e.g., "http://localhost:9999")
        """
        # Convert bind address to HTTP hostname
        http_host = "localhost" if host == "0.0.0.0" else host
        
        # Set base URL environment variable if not set
        if not os.getenv("A2A_BASE_URL"):
            os.environ["A2A_BASE_URL"] = f"http://{http_host}:{port}"
        
        # Update default environment URI to use mounted path
        # Use localhost instead of 0.0.0.0 for HTTP requests
        default_env_uri = f"http://{http_host}:{port}/mcp"
        # Set default context_id to use mounted environment if not already set
        if None not in self.context_id_to_environment_uri and "" not in self.context_id_to_environment_uri:
            self.context_id_to_environment_uri[None] = default_env_uri
            # Update Loop's mapping
            self.loop.context_id_to_environment_uri[None] = default_env_uri
        
        # Start server
        config = uvicorn.Config(
            app=self.fastapi_app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        
        # Start in background
        import asyncio
        asyncio.create_task(server.serve())
        
        # Return agent URI
        return f"http://{http_host}:{port}"
    
    def run(self, port: int = 9999, host: str = "0.0.0.0"):
        """Run the A2A server (blocking)."""
        # Convert bind address to HTTP hostname
        http_host = "localhost" if host == "0.0.0.0" else host
        
        # Set base URL environment variable if not set
        if not os.getenv("A2A_BASE_URL"):
            os.environ["A2A_BASE_URL"] = f"http://{http_host}:{port}"
        
        # Update default environment URI to use mounted path
        # Use localhost instead of 0.0.0.0 for HTTP requests
        default_env_uri = f"http://{http_host}:{port}/mcp"
        # Set default context_id to use mounted environment if not already set
        if None not in self.context_id_to_environment_uri and "" not in self.context_id_to_environment_uri:
            self.context_id_to_environment_uri[None] = default_env_uri
            # Update Loop's mapping
            self.loop.context_id_to_environment_uri[None] = default_env_uri
        
        uvicorn.run(self.fastapi_app, host=host, port=port)
