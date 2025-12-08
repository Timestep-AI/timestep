"""RunStateStore implementation using PostgreSQL."""

import json
import uuid
from typing import Optional, Any

from ..shared.db_connection import DatabaseConnection
from ..._vendored_imports import Agent, RunState


class RunStateStore:
    """Store for persisting run state using PostgreSQL."""
    
    SCHEMA_VERSION = "1.0"
    
    def __init__(
        self,
        agent: Agent,
        session_id: Optional[str] = None,
        connection_string: Optional[str] = None
    ):
        """
        Initialize RunStateStore.
        
        Note: If connection_string is not provided and DBOS is not configured,
        the connection will be initialized lazily when needed (which will configure DBOS
        and use PG_CONNECTION_URI if available).
        """
        """
        Initialize RunStateStore using DBOS's database connection if available.
        
        RunStateStore will use the same database as DBOS:
        - If PG_CONNECTION_URI is set, both use that PostgreSQL database
        - Otherwise, both use the same PostgreSQL database (via PG_CONNECTION_URI)
        
        Args:
            agent: Agent instance (required)
            session_id: Session ID to use as identifier (required, will be generated if not provided)
            connection_string: PostgreSQL connection string (optional, uses DBOS's connection if not provided)
        """
        if agent is None:
            raise ValueError("agent is required")
        
        self.agent = agent
        self.session_id = session_id
        self._connection_string = connection_string
        self.db: Optional[DatabaseConnection] = None
        self._connected = False
    
    async def _initialize_database(self) -> None:
        """Initialize the database connection."""
        connection_string = self._connection_string
        # Get connection string from DBOS if not explicitly provided
        if not connection_string:
            from ...config.dbos_config import get_dbos_connection_string, configure_dbos
            connection_string = get_dbos_connection_string()
            
            # If DBOS is not configured, auto-configure it (will use PG_CONNECTION_URI if available)
            if not connection_string:
                await configure_dbos()
                connection_string = get_dbos_connection_string()
        
        if not connection_string:
            raise ValueError(
                "No connection string provided. "
                "Either provide connection_string in options or ensure DBOS is configured with PG_CONNECTION_URI."
            )
        
        # Normalize connection string: remove +psycopg2 from scheme (asyncpg expects postgresql:// or postgres://)
        if connection_string.startswith('postgresql+psycopg2://'):
            connection_string = connection_string.replace('postgresql+psycopg2://', 'postgresql://', 1)
        elif connection_string.startswith('postgres+psycopg2://'):
            connection_string = connection_string.replace('postgres+psycopg2://', 'postgresql://', 1)
        
        # Use PostgreSQL connection string
        self.db = DatabaseConnection(connection_string=connection_string)
        self._connected = False
    
    async def _ensure_connected(self) -> None:
        """Ensure database connection is established."""
        if not self.db:
            await self._initialize_database()
        if not self._connected:
            connected = await self.db.connect()
            if not connected:
                raise RuntimeError(
                    "Failed to connect to database. "
                    "Check PG_CONNECTION_URI environment variable or ensure DBOS is configured."
                )
            self._connected = True
    
    async def _ensure_session_id(self) -> str:
        """Ensure we have a session_id, creating one if needed."""
        if self.session_id:
            return self.session_id
        
        # Generate a new session_id
        self.session_id = str(uuid.uuid4())
        await self._ensure_connected()
        
        return self.session_id
    
    async def save(self, state: Any) -> None:
        """
        Save state to database.
        
        Args:
            state: RunState instance to save
        """
        await self._ensure_connected()
        session_id = await self._ensure_session_id()
        
        # Convert state to JSON
        state_json = state.to_json()
        
        # Determine state type
        state_type = "interrupted" if state_json.get("interruptions") else "checkpoint"
        
        # Mark previous states as inactive
        await self.db.execute(
            """
            UPDATE run_states
            SET is_active = false
            WHERE run_id = $1 AND is_active = true
            """,
            session_id
        )
        
        # Insert new state
        await self.db.execute(
            """
            INSERT INTO run_states (run_id, state_type, schema_version, state_data, is_active)
            VALUES ($1, $2, $3, $4, true)
            """,
            session_id,
            state_type,
            self.SCHEMA_VERSION,
            json.dumps(state_json)
        )
    
    async def load(self) -> Any:
        """
        Load active state from database.
        
        Returns:
            RunState instance
        """
        await self._ensure_connected()
        session_id = await self._ensure_session_id()
        
        # Fetch active state
        row = await self.db.fetchrow(
            """
            SELECT state_data, state_type, created_at
            FROM run_states
            WHERE run_id = $1 AND is_active = true
            ORDER BY created_at DESC
            LIMIT 1
            """,
            session_id
        )
        
        if not row:
            raise FileNotFoundError(
                f"No active state found for session_id: {session_id}. "
                "Make sure you've saved a state first."
            )
        
        # Update resumed_at timestamp
        await self.db.execute(
            """
            UPDATE run_states
            SET resumed_at = NOW()
            WHERE run_id = $1 AND is_active = true
            """,
            session_id
        )
        
        # Deserialize state
        state_json = row["state_data"]
        if isinstance(state_json, str):
            state_json = json.loads(state_json)
        
        return await RunState.from_json(self.agent, state_json)
    
    async def clear(self) -> None:
        """Mark state as inactive (soft delete)."""
        if not self.session_id:
            return
        
        try:
            await self._ensure_connected()
            await self.db.execute(
                """
                UPDATE run_states
                SET is_active = false
                WHERE run_id = $1
                """,
                self.session_id
            )
        except Exception:
            # If database is not available, silently fail (graceful degradation)
            pass
    
    async def close(self) -> None:
        """Close database connection."""
        if self.db:
            await self.db.disconnect()
        self._connected = False
