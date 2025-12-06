"""RunStateStore implementation using PGLite by default."""

import json
import uuid
from typing import Optional, Any
from datetime import datetime

from .db_connection import DatabaseConnection, DatabaseType
from ._vendored_imports import Agent, RunState


class RunStateStore:
    """Store for persisting run state using PGLite (default) or PostgreSQL."""
    
    SCHEMA_VERSION = "1.0"
    
    def __init__(
        self,
        agent: Agent,
        session_id: Optional[str] = None,
        connection_string: Optional[str] = None,
        use_pglite: Optional[bool] = None,
        pglite_path: Optional[str] = None
    ):
        """
        Initialize RunStateStore with PGLite by default.
        
        Args:
            agent: Agent instance (required)
            session_id: Session ID to use as identifier (required, will be generated if not provided)
            connection_string: PostgreSQL connection string (optional, uses PGLite if not provided)
            use_pglite: Whether to use PGLite (defaults to True if no connection_string)
            pglite_path: Path for PGLite data directory (defaults to app directory)
        """
        if agent is None:
            raise ValueError("agent is required")
        
        self.agent = agent
        self.session_id = session_id
        
        # Use session ID in PGLite path to avoid concurrent access issues
        # Each session gets its own database file
        if not pglite_path and not connection_string and (use_pglite is not False):
            from .app_dir import get_pglite_dir
            session_id_for_path = session_id or 'default'
            pglite_path = str(get_pglite_dir(session_id_for_path))
        
        # Default to PGLite if no connection string provided
        self.db = DatabaseConnection(
            connection_string=connection_string,
            use_pglite=use_pglite,  # None means auto-detect (defaults to PGLite)
            pglite_path=pglite_path
        )
        self._connected = False
    
    async def _ensure_connected(self) -> None:
        """Ensure database connection is established."""
        if not self._connected:
            connected = await self.db.connect()
            if not connected:
                raise RuntimeError(
                    "Failed to connect to database. "
                    "Check PG_CONNECTION_URI environment variable or ensure PGLite dependencies are installed."
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
        await self.db.disconnect()
        self._connected = False

