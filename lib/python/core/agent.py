"""Agent class - A2A Server that contains Loop internally."""

import os
from typing import Dict, Optional
from fastapi import FastAPI
import uvicorn
from a2a.server.apps.rest.fastapi_app import A2ARESTFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, TransportProtocol

from timestep.core.loop import Loop


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
        trace_to_file: str = "traces.jsonl",
    ):
        self.agent_id = agent_id
        self.name = name
        self.model = model
        self.context_id_to_environment_uri = context_id_to_environment_uri or {}
        self.human_in_loop = human_in_loop
        self.trace_to_file = trace_to_file
        
        # Loop (AgentExecutor) is inside Agent
        self.loop = Loop(
            agent_id=agent_id,
            model=model,
            context_id_to_environment_uri=self.context_id_to_environment_uri,
            human_in_loop=human_in_loop,
            trace_to_file=trace_to_file,
        )
        
        # Create A2A server with Loop as AgentExecutor
        self.handler = DefaultRequestHandler(
            agent_executor=self.loop,
            task_store=InMemoryTaskStore(),
        )
        
        # Create agent card
        self.agent_card = self._create_agent_card()
        
        # Create A2A app
        self.app = A2ARESTFastAPIApplication(
            agent_card=self.agent_card,
            http_handler=self.handler,
        )
        
        # Create FastAPI app
        self.fastapi_app = FastAPI()
        self.a2a_app = self.app.build()
        
        # Mount A2A app
        self.fastapi_app.mount(f"/agents/{agent_id}", self.a2a_app)
        
        # Add agent card endpoint
        @self.fastapi_app.get(f"/agents/{agent_id}/.well-known/agent-card.json")
        async def get_agent_card():
            return self.agent_card.model_dump(mode="json")
    
    def _create_agent_card(self) -> AgentCard:
        """Create an agent card for this agent."""
        base_url = os.getenv("A2A_BASE_URL", "http://localhost:8000")
        return AgentCard(
            name=self.name,
            version="1.0.0",
            description=f"{self.name} agent",
            url=f"{base_url}/agents/{self.agent_id}",
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
    
    async def start(self, port: int = 8000, host: str = "0.0.0.0") -> str:
        """Start A2A server and return agent URI.
        
        Args:
            port: Port to run the A2A server on
            host: Host to bind to
            
        Returns:
            Agent URI (e.g., "http://localhost:8000/agents/{agent_id}")
        """
        # Set base URL environment variable if not set
        if not os.getenv("A2A_BASE_URL"):
            os.environ["A2A_BASE_URL"] = f"http://{host}:{port}"
        
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
        return f"http://{host}:{port}/agents/{self.agent_id}"
    
    def run(self, port: int = 8000, host: str = "0.0.0.0"):
        """Run the A2A server (blocking)."""
        # Set base URL environment variable if not set
        if not os.getenv("A2A_BASE_URL"):
            os.environ["A2A_BASE_URL"] = f"http://{host}:{port}"
        
        uvicorn.run(self.fastapi_app, host=host, port=port)
