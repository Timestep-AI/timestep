"""Database-backed RunStateStore implementation."""

import json
import uuid
from typing import Optional, Any
from datetime import datetime

from .db_connection import DatabaseConnection, DatabaseType
from ._vendored_imports import Agent, RunState


class DatabaseRunStateStore:
    """Database-backed store for persisting run state."""
    
    SCHEMA_VERSION = "1.0"
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        agent: Agent = None,
        connection_string: Optional[str] = None,
        use_pglite: bool = False,
        pglite_path: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize database-backed RunStateStore.
        
        Args:
            run_id: UUID of the run (optional, will be generated if not provided)
            agent: Agent instance (required)
            connection_string: PostgreSQL connection string
            use_pglite: Whether to use PGLite (not yet supported)
            pglite_path: Path for PGLite data directory
            session_id: Session ID to use as identifier (alternative to run_id)
        """
        if agent is None:
            raise ValueError("agent is required")
        
        self.agent = agent
        self.run_id = run_id or session_id  # Use session_id as fallback
        self.db = DatabaseConnection(
            connection_string=connection_string,
            use_pglite=use_pglite,
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
                    "Check TIMESTEP_DB_URL environment variable or use file-based storage."
                )
            self._connected = True
    
    async def _ensure_run_id(self) -> str:
        """Ensure we have a run_id, creating a minimal run record if needed."""
        if self.run_id:
            return self.run_id
        
        # Generate a new run_id
        self.run_id = str(uuid.uuid4())
        
        # For MVP, we'll create a minimal run record if the runs table exists
        # Otherwise, we'll use the run_id directly (the foreign key constraint
        # will be handled by the application layer)
        try:
            await self._ensure_connected()
            
            # Check if runs table exists and create a minimal run if needed
            # This is a simplified approach for MVP
            # In production, you'd want to properly create a run record
            pass
        except Exception:
            # If we can't create a run record, we'll still use the run_id
            # The database constraint will need to be handled at the application level
            pass
        
        return self.run_id
    
    async def save(self, state: Any) -> None:
        """
        Save state to database.
        
        Args:
            state: RunState instance to save
        """
        await self._ensure_connected()
        run_id = await self._ensure_run_id()
        
        # Convert state to JSON
        state_json = state.to_json()
        
        # Determine state type
        # For MVP, we'll use 'interrupted' if there are interruptions, otherwise 'checkpoint'
        state_type = "interrupted" if state_json.get("interruptions") else "checkpoint"
        
        # Mark previous states as inactive
        await self.db.execute(
            """
            UPDATE run_states
            SET is_active = false
            WHERE run_id = $1 AND is_active = true
            """,
            run_id
        )
        
        # Insert new state
        await self.db.execute(
            """
            INSERT INTO run_states (run_id, state_type, schema_version, state_data, is_active)
            VALUES ($1, $2, $3, $4, true)
            """,
            run_id,
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
        run_id = await self._ensure_run_id()
        
        # Fetch active state
        row = await self.db.fetchrow(
            """
            SELECT state_data, state_type, created_at
            FROM run_states
            WHERE run_id = $1 AND is_active = true
            ORDER BY created_at DESC
            LIMIT 1
            """,
            run_id
        )
        
        if not row:
            raise FileNotFoundError(
                f"No active state found for run_id: {run_id}. "
                "Make sure you've saved a state first."
            )
        
        # Update resumed_at timestamp
        await self.db.execute(
            """
            UPDATE run_states
            SET resumed_at = NOW()
            WHERE run_id = $1 AND is_active = true
            """,
            run_id
        )
        
        # Deserialize state
        state_json = row["state_data"]
        if isinstance(state_json, str):
            state_json = json.loads(state_json)
        
        return await RunState.from_json(self.agent, state_json)
    
    async def clear(self) -> None:
        """Mark state as inactive (soft delete)."""
        if not self.run_id:
            return
        
        try:
            await self._ensure_connected()
            await self.db.execute(
                """
                UPDATE run_states
                SET is_active = false
                WHERE run_id = $1
                """,
                self.run_id
            )
        except Exception:
            # If database is not available, silently fail (graceful degradation)
            pass
    
    async def close(self) -> None:
        """Close database connection."""
        await self.db.disconnect()
        self._connected = False

