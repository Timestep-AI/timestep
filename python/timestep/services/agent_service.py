"""Agent service for business logic and orchestration."""

from __future__ import annotations

from typing import Any, Dict
from uuid import UUID

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import DatabaseTaskStore
from a2a.types import AgentCapabilities, AgentCard
from starlette.applications import Starlette

from timestep.services.environment import Environment
from timestep.services.executor import TimestepAgentExecutor
from timestep.stores.agent_store import AgentStore
from timestep.stores.session import FileSession
from timestep.utils.exceptions import AgentConfigError


class AgentService:
    """Service for agent business logic and orchestration."""
    
    def __init__(self, agent_store: AgentStore):
        """Initialize agent service.
        
        Args:
            agent_store: AgentStore instance for persistence
        """
        self.agent_store = agent_store
        
        # Executor cache: agent_id -> TimestepAgentExecutor
        self._executor_cache: Dict[str, TimestepAgentExecutor] = {}
        
        # A2A app cache: agent_id -> Starlette app
        self._a2a_app_cache: Dict[str, Starlette] = {}
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            Created agent dictionary with ID
            
        Raises:
            AgentConfigError: If agent configuration is invalid
        """
        # Validate agent config
        Environment.validate_agent_config(agent_config)
        
        # Persist agent
        return await self.agent_store.create_agent(agent_config)
    
    async def get_agent(self, agent_id: UUID) -> Dict[str, Any] | None:
        """Get agent by ID.
        
        Args:
            agent_id: Agent UUID
            
        Returns:
            Agent dictionary or None if not found
        """
        return await self.agent_store.get_agent(agent_id)
    
    async def list_agents(self) -> list[Dict[str, Any]]:
        """List all agents.
        
        Returns:
            List of agent dictionaries
        """
        return await self.agent_store.list_agents()
    
    async def update_agent(
        self,
        agent_id: UUID,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """Update an existing agent.
        
        Args:
            agent_id: Agent UUID
            agent_config: Updated agent configuration dictionary
            
        Returns:
            Updated agent dictionary or None if not found
            
        Raises:
            AgentConfigError: If agent configuration is invalid
        """
        # Validate agent config
        Environment.validate_agent_config(agent_config)
        
        # Update agent
        agent = await self.agent_store.update_agent(agent_id, agent_config)
        
        if agent is not None:
            # Invalidate caches for this agent
            self._invalidate_caches(str(agent_id))
        
        return agent
    
    async def delete_agent(self, agent_id: UUID) -> bool:
        """Delete an agent.
        
        Args:
            agent_id: Agent UUID
            
        Returns:
            True if deleted, False if not found
        """
        deleted = await self.agent_store.delete_agent(agent_id)
        
        if deleted:
            # Remove from caches
            self._invalidate_caches(str(agent_id))
        
        return deleted
    
    def get_executor(
        self,
        agent_id: UUID,
        task_store: DatabaseTaskStore,
    ) -> TimestepAgentExecutor:
        """Get or create executor for an agent (cached).
        
        Args:
            agent_id: Agent UUID
            task_store: DatabaseTaskStore instance
            
        Returns:
            TimestepAgentExecutor instance
            
        Raises:
            ValueError: If agent not found
        """
        agent_id_str = str(agent_id)
        
        if agent_id_str not in self._executor_cache:
            # Get agent data
            import asyncio
            agent_data = asyncio.run(self.agent_store.get_agent(agent_id))
            if agent_data is None:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Create agent config from database data
            agent_config = {
                "id": agent_id_str,
                "name": agent_data["name"],
                "model": agent_data["model"],
                "instructions": agent_data["instructions"],
                "tools": agent_data.get("tools", []),
                "handoffs": agent_data.get("handoffs", []),
                "guardrails": agent_data.get("guardrails", []),
            }
            
            # Create session for the agent
            session = FileSession(
                agent_name=agent_data["name"],
                conversation_id=agent_id_str,
                agent_instructions=agent_data["instructions"],
            )
            
            # Create executor
            executor = TimestepAgentExecutor(
                agent_config=agent_config,
                session=session,
            )
            
            self._executor_cache[agent_id_str] = executor
        
        return self._executor_cache[agent_id_str]
    
    async def get_a2a_app(
        self,
        agent_id: UUID,
        task_store: DatabaseTaskStore,
        base_url: str,
    ) -> Starlette:
        """Get or create A2A app for an agent (cached).
        
        Args:
            agent_id: Agent UUID
            task_store: DatabaseTaskStore instance
            base_url: Base URL for the agent
            
        Returns:
            Starlette app configured for this agent
            
        Raises:
            ValueError: If agent not found
        """
        agent_id_str = str(agent_id)
        
        if agent_id_str not in self._a2a_app_cache:
            # Get agent data
            agent_data = await self.agent_store.get_agent(agent_id)
            if agent_data is None:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Get or create executor
            executor = await self.get_executor(agent_id, task_store)
            
            # Create request handler
            request_handler = DefaultRequestHandler(
                agent_executor=executor,
                task_store=task_store,
            )
            
            # Create agent cards
            public_card = self.create_agent_card(agent_id, base_url, extended=False)
            extended_card = self.create_agent_card(agent_id, base_url, extended=True)
            
            # Create A2A application
            a2a_app = A2AStarletteApplication(
                agent_card=public_card,
                http_handler=request_handler,
                extended_agent_card=extended_card,
            )
            
            self._a2a_app_cache[agent_id_str] = a2a_app.build()
        
        return self._a2a_app_cache[agent_id_str]
    
    async def create_agent_card(
        self,
        agent_id: UUID,
        base_url: str,
        extended: bool = False,
    ) -> AgentCard:
        """Create agent card.
        
        Args:
            agent_id: Agent UUID
            base_url: Base URL for the agent
            extended: Whether to create extended card
            
        Returns:
            AgentCard object
            
        Raises:
            ValueError: If agent not found
        """
        agent_data = await self.agent_store.get_agent(agent_id)
        if agent_data is None:
            raise ValueError(f"Agent {agent_id} not found")
        
        name = agent_data["name"]
        
        if extended:
            name = f"{name} - Extended Edition"
        
        url = f"{base_url}/agents/{agent_id}/"
        
        return AgentCard(
            name=name,
            description=agent_data.get("description", ""),
            url=url,
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[],
            supports_authenticated_extended_card=True,
        )
    
    def _invalidate_caches(self, agent_id_str: str) -> None:
        """Invalidate caches for an agent.
        
        Args:
            agent_id_str: Agent ID as string
        """
        if agent_id_str in self._executor_cache:
            del self._executor_cache[agent_id_str]
        if agent_id_str in self._a2a_app_cache:
            del self._a2a_app_cache[agent_id_str]

