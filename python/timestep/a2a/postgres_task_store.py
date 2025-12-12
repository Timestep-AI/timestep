"""PostgreSQL TaskStore implementation for A2A protocol."""

import json
import os
from typing import Any
from uuid import UUID

import asyncpg
from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskStore
from a2a.types import Task
from openai.types.chat import ChatCompletionMessageParam


class PostgresTaskStore(TaskStore):
    """PostgreSQL-backed TaskStore implementation."""

    def __init__(self, connection_string: str | None = None, agent_id: str | None = None) -> None:
        """Initialize PostgreSQL TaskStore.

        Args:
            connection_string: PostgreSQL connection string. If None, constructs from env vars.
            agent_id: Optional agent ID to associate with tasks.
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
        self.agent_id = agent_id

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=20,
            )
        return self._pool

    async def save(self, task: Task, context: ServerCallContext | None = None) -> None:
        """Save a task to the database."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Ensure context exists
            await self._ensure_context(conn, task.context_id)

            # Upsert task (openai_messages will be saved separately via save_openai_messages)
            await conn.execute(
                """
                INSERT INTO tasks (id, context_id, agent_id, data, updated_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO UPDATE
                SET data = $4, agent_id = $3, updated_at = CURRENT_TIMESTAMP
                """,
                str(task.id),
                str(task.context_id),
                self.agent_id,
                json.dumps(task.model_dump(mode="json")),
            )

            # Update context updated_at
            await conn.execute(
                "UPDATE contexts SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                str(task.context_id),
            )

    async def get(self, task_id: str, context: ServerCallContext | None = None) -> Task | None:
        """Get a task by ID (required by TaskStore interface)."""
        return await self.load(task_id)

    async def load(self, task_id: str) -> Task | None:
        """Load a task by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM tasks WHERE id = $1", task_id
            )
            if row is None:
                return None
            task_data = json.loads(row["data"])
            return Task.model_validate(task_data)

    async def load_by_context_id(self, context_id: str) -> list[Task]:
        """Load all tasks for a given context ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM tasks WHERE context_id = $1 ORDER BY created_at ASC",
                context_id,
            )
            return [Task.model_validate(json.loads(row["data"])) for row in rows]

    async def list_contexts(self, parent_id: str | None = None) -> list[dict[str, Any]]:
        """List all contexts, optionally filtered by parent_id."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT id, created_at, updated_at, metadata, parent_context_id
                FROM contexts
            """
            params: list[str] = []
            if parent_id:
                query += " WHERE parent_context_id = $1"
                params.append(parent_id)
            query += " ORDER BY updated_at DESC"
            
            rows = await conn.fetch(query, *params)
            return [
                {
                    "id": str(row["id"]),
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "metadata": row["metadata"],
                    "parent_context_id": str(row["parent_context_id"]) if row["parent_context_id"] else None,
                }
                for row in rows
            ]

    async def create_context(
        self, 
        metadata: dict[str, Any] | None = None,
        parent_context_id: str | None = None
    ) -> dict[str, Any]:
        """Create a new context, optionally with a parent."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO contexts (metadata, parent_context_id)
                VALUES ($1, $2)
                RETURNING id, created_at, updated_at, metadata, parent_context_id
                """,
                json.dumps(metadata) if metadata else None,
                parent_context_id,
            )
            return {
                "id": str(row["id"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "metadata": row["metadata"],
                "parent_context_id": str(row["parent_context_id"]) if row["parent_context_id"] else None,
            }

    async def _ensure_context(self, conn: asyncpg.Connection, context_id: str) -> None:
        """Ensure a context exists in the database."""
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM contexts WHERE id = $1)", context_id
        )
        if not exists:
            await conn.execute("INSERT INTO contexts (id) VALUES ($1)", context_id)

    async def delete(self, task_id: str, context: ServerCallContext | None = None) -> None:
        """Delete a task by ID (required by TaskStore interface)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM tasks WHERE id = $1", task_id)

    async def delete_context(self, context_id: str) -> None:
        """Delete a context and all its tasks (CASCADE will handle tasks)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM contexts WHERE id = $1", context_id)

    async def save_openai_messages(
        self, task_id: str, messages: list[ChatCompletionMessageParam]
    ) -> None:
        """Save OpenAI messages for a task."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE tasks SET openai_messages = $1, updated_at = CURRENT_TIMESTAMP
                WHERE id = $2
                """,
                json.dumps(messages),
                task_id,
            )

    async def get_openai_messages(
        self, task_id: str
    ) -> list[ChatCompletionMessageParam] | None:
        """Get OpenAI messages for a task."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT openai_messages FROM tasks WHERE id = $1", task_id
            )
            if row is None or row["openai_messages"] is None:
                return None
            return json.loads(row["openai_messages"])

    async def get_openai_messages_by_context_id(
        self, context_id: str
    ) -> list[ChatCompletionMessageParam]:
        """Get all OpenAI messages for a context, merged chronologically and deduplicated."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT openai_messages FROM tasks
                WHERE context_id = $1 AND openai_messages IS NOT NULL
                ORDER BY created_at ASC
                """,
                context_id,
            )

            # Merge all messages from all tasks and deduplicate
            all_messages: list[ChatCompletionMessageParam] = []
            seen_messages: set[str] = set()

            for row in rows:
                messages = json.loads(row["openai_messages"])
                if messages and isinstance(messages, list):
                    for msg in messages:
                        # Create a unique key for deduplication
                        # For system messages, use role only (should be unique)
                        # For other messages, use role + content hash
                        if msg.get("role") == "system":
                            content = (
                                msg.get("content")
                                if isinstance(msg.get("content"), str)
                                else json.dumps(msg.get("content"))
                            )
                            message_key = f"system:{content}"
                        elif msg.get("role") == "tool":
                            tool_call_id = msg.get("tool_call_id", "")
                            content = (
                                msg.get("content")
                                if isinstance(msg.get("content"), str)
                                else json.dumps(msg.get("content"))
                            )
                            message_key = f"tool:{tool_call_id}:{content}"
                        elif msg.get("role") == "assistant":
                            # For assistant messages with tool_calls, use tool_call IDs
                            tool_calls = msg.get("tool_calls")
                            if tool_calls and isinstance(tool_calls, list):
                                tool_call_ids = sorted(
                                    [
                                        tc.get("id", "")
                                        for tc in tool_calls
                                        if isinstance(tc, dict) and tc.get("id")
                                    ]
                                )
                                message_key = f"assistant:tool_calls:{','.join(tool_call_ids)}"
                            else:
                                content = (
                                    msg.get("content")
                                    if isinstance(msg.get("content"), str)
                                    else json.dumps(msg.get("content"))
                                )
                                message_key = f"assistant:{content}"
                        else:
                            # user messages
                            content = (
                                msg.get("content")
                                if isinstance(msg.get("content"), str)
                                else json.dumps(msg.get("content"))
                            )
                            message_key = f"user:{content}"

                        # Only add if we haven't seen this message before
                        if message_key not in seen_messages:
                            seen_messages.add(message_key)
                            all_messages.append(msg)

            return all_messages

    async def get_context(self, context_id: str) -> dict[str, Any] | None:
        """Get a single context by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, created_at, updated_at, metadata, parent_context_id
                FROM contexts
                WHERE id = $1
                """,
                context_id,
            )
            if row is None:
                return None
            return {
                "id": str(row["id"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "metadata": row["metadata"],
                "parent_context_id": str(row["parent_context_id"]) if row["parent_context_id"] else None,
            }

    async def update_context(
        self, context_id: str, updates: dict[str, Any]
    ) -> dict[str, Any]:
        """Update context fields."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            updates_list: list[str] = []
            params: list[Any] = []
            param_index = 1

            if "parent_context_id" in updates:
                updates_list.append(f"parent_context_id = ${param_index}")
                params.append(updates["parent_context_id"] or None)
                param_index += 1

            if not updates_list:
                # No updates, just return existing context
                existing = await self.get_context(context_id)
                if not existing:
                    raise ValueError(f"Context not found: {context_id}")
                return existing

            updates_list.append("updated_at = CURRENT_TIMESTAMP")
            params.append(context_id)

            query = f"""
                UPDATE contexts
                SET {', '.join(updates_list)}
                WHERE id = ${param_index}
                RETURNING id, created_at, updated_at, metadata, parent_context_id
            """

            row = await conn.fetchrow(query, *params)
            if row is None:
                raise ValueError(f"Context not found: {context_id}")

            return {
                "id": str(row["id"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "metadata": row["metadata"],
                "parent_context_id": str(row["parent_context_id"]) if row["parent_context_id"] else None,
            }

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

