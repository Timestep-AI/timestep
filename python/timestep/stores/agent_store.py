"""Agent storage layer for PostgreSQL."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class AgentModel(Base):
    """SQLAlchemy model for agents table."""
    
    __tablename__ = "agents"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4, server_default=text("gen_random_uuid()"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    instructions: Mapped[str] = mapped_column(Text, nullable=False)
    tools: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    handoffs: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    guardrails: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("now()"), onupdate=text("now()"))


class AgentStore:
    """Agent storage operations."""
    
    def __init__(self, engine: AsyncEngine):
        """Initialize agent store.
        
        Args:
            engine: SQLAlchemy async engine
        """
        self.engine = engine
        self.async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async def create_tables(self) -> None:
        """Create agent tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            Created agent dictionary with ID
        """
        async with self.async_session_maker() as session:
            agent = AgentModel(
                name=agent_config["name"],
                model=agent_config["model"],
                instructions=agent_config["instructions"],
                tools=agent_config.get("tools"),
                handoffs=agent_config.get("handoffs"),
                guardrails=agent_config.get("guardrails"),
            )
            session.add(agent)
            await session.commit()
            await session.refresh(agent)
            return self._model_to_dict(agent)
    
    async def get_agent(self, agent_id: UUID) -> Optional[Dict[str, Any]]:
        """Get agent by ID.
        
        Args:
            agent_id: Agent UUID
            
        Returns:
            Agent dictionary or None if not found
        """
        async with self.async_session_maker() as session:
            result = await session.get(AgentModel, agent_id)
            if result is None:
                return None
            return self._model_to_dict(result)
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents.
        
        Returns:
            List of agent dictionaries
        """
        from sqlalchemy import select
        
        async with self.async_session_maker() as session:
            result = await session.execute(select(AgentModel))
            agents = result.scalars().all()
            return [self._model_to_dict(agent) for agent in agents]
    
    async def update_agent(self, agent_id: UUID, agent_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing agent.
        
        Args:
            agent_id: Agent UUID
            agent_config: Updated agent configuration dictionary
            
        Returns:
            Updated agent dictionary or None if not found
        """
        async with self.async_session_maker() as session:
            agent = await session.get(AgentModel, agent_id)
            if agent is None:
                return None
            
            agent.name = agent_config["name"]
            agent.model = agent_config["model"]
            agent.instructions = agent_config["instructions"]
            agent.tools = agent_config.get("tools")
            agent.handoffs = agent_config.get("handoffs")
            agent.guardrails = agent_config.get("guardrails")
            agent.updated_at = datetime.utcnow()
            
            await session.commit()
            await session.refresh(agent)
            return self._model_to_dict(agent)
    
    async def delete_agent(self, agent_id: UUID) -> bool:
        """Delete an agent.
        
        Args:
            agent_id: Agent UUID
            
        Returns:
            True if deleted, False if not found
        """
        async with self.async_session_maker() as session:
            agent = await session.get(AgentModel, agent_id)
            if agent is None:
                return False
            
            await session.delete(agent)
            await session.commit()
            return True
    
    def _model_to_dict(self, agent: AgentModel) -> Dict[str, Any]:
        """Convert AgentModel to dictionary.
        
        Args:
            agent: AgentModel instance
            
        Returns:
            Agent dictionary
        """
        return {
            "id": str(agent.id),
            "name": agent.name,
            "model": agent.model,
            "instructions": agent.instructions,
            "tools": agent.tools,
            "handoffs": agent.handoffs,
            "guardrails": agent.guardrails,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
        }

