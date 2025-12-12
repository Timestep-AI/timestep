"""PostgreSQL AgentStore implementation for managing agents."""

import json
import os
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg


class Agent:
    """Agent model."""

    def __init__(
        self,
        id: str,
        name: str,
        description: str | None = None,
        tools: list[str] | None = None,
        model: str = "gpt-4.1",
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.description = description
        self.tools = tools or []
        self.model = model
        self.created_at = created_at
        self.updated_at = updated_at


class PostgresAgentStore:
    """PostgreSQL-backed AgentStore implementation."""

    def __init__(self, connection_string: str | None = None) -> None:
        """Initialize PostgreSQL AgentStore.

        Args:
            connection_string: PostgreSQL connection string. If None, constructs from env vars.
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            user = os.getenv("POSTGRES_USER", "timestep")
            password = os.getenv("POSTGRES_PASSWORD", "timestep")
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "timestep")
            self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db}"

        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=20,
            )
        return self._pool

    async def get_agent(self, agent_id: str) -> Agent | None:
        """Retrieve agent configuration by ID.

        Args:
            agent_id: Agent identifier.

        Returns:
            Agent instance or None if not found.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, name, description, tools, model, created_at, updated_at FROM agents WHERE id = $1",
                agent_id,
            )

            if not row:
                return None

            return Agent(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                tools=row["tools"] or [],
                model=row["model"] or "gpt-4.1",
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    async def list_agents(self) -> list[Agent]:
        """List all agents.

        Returns:
            List of Agent instances.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, description, tools, model, created_at, updated_at FROM agents ORDER BY created_at ASC"
            )

            return [
                Agent(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    tools=row["tools"] or [],
                    model=row["model"] or "gpt-4.1",
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def create_agent(
        self, agent: Agent
    ) -> Agent:
        """Create a new agent.

        Args:
            agent: Agent instance (without created_at/updated_at).

        Returns:
            Created Agent instance with timestamps.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO agents (id, name, description, tools, model)
                   VALUES ($1, $2, $3, $4, $5)
                   RETURNING id, name, description, tools, model, created_at, updated_at""",
                agent.id,
                agent.name,
                agent.description,
                json.dumps(agent.tools),
                agent.model,
            )

            return Agent(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                tools=row["tools"] or [],
                model=row["model"] or "gpt-4.1",
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def get_default_agents(self) -> list[Agent]:
        """Get default agent configurations.

        Returns:
            List of default Agent instances.
        """
        return [
            Agent(
                id="personal-assistant",
                name="Personal Assistant",
                description="Personal assistant with web search and agent handoff capabilities",
                tools=["handoff", "web_search"],
                model="gpt-4.1",
            ),
            Agent(
                id="weather-assistant",
                name="Weather Assistant",
                description="Specialized weather information agent",
                tools=["get_weather"],
                model="gpt-4.1",
            ),
        ]

