"""Database connection management for Timestep."""

import os
import asyncio
from typing import Optional, Any
from enum import Enum


class DatabaseType(Enum):
    """Database type enumeration."""
    POSTGRESQL = "postgresql"
    PGLITE = "pglite"
    NONE = "none"


class DatabaseConnection:
    """Manages database connections for Timestep."""
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        use_pglite: bool = False,
        pglite_path: Optional[str] = None
    ):
        """
        Initialize database connection.
        
        Args:
            connection_string: PostgreSQL connection string (e.g., postgresql://user:pass@host/db)
            use_pglite: Whether to use PGLite for local development
            pglite_path: Path for PGLite data directory
        """
        self.connection_string = connection_string or os.environ.get("TIMESTEP_DB_URL")
        self.use_pglite = use_pglite or os.environ.get("TIMESTEP_USE_PGLITE", "").lower() == "true"
        self.pglite_path = pglite_path or os.environ.get("TIMESTEP_PGLITE_PATH", "./pglite_data")
        self._connection: Optional[Any] = None
        self._db_type: DatabaseType = DatabaseType.NONE
        
    async def connect(self) -> bool:
        """
        Connect to database. Returns True if successful, False otherwise.
        
        Returns:
            True if connection successful, False otherwise
        """
        # Try PostgreSQL first if connection string is provided
        if self.connection_string and not self.use_pglite:
            try:
                return await self._connect_postgresql()
            except Exception:
                pass
        
        # Try PGLite if enabled
        if self.use_pglite:
            try:
                return await self._connect_pglite()
            except Exception:
                pass
        
        return False
    
    async def _connect_postgresql(self) -> bool:
        """Connect to PostgreSQL database."""
        try:
            import asyncpg
            
            # Parse connection string
            self._connection = await asyncpg.connect(self.connection_string)
            self._db_type = DatabaseType.POSTGRESQL
            
            # Test connection
            await self._connection.fetchval("SELECT 1")
            return True
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgreSQL support. "
                "Install it with: pip install asyncpg"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
    
    async def _connect_pglite(self) -> bool:
        """Connect to PGLite database."""
        try:
            # For MVP, we'll use a subprocess approach or note that PGLite Python bindings
            # may not be available. For now, we'll raise a NotImplementedError.
            # In production, this would use PGLite Python bindings when available.
            raise NotImplementedError(
                "PGLite Python bindings are not yet available. "
                "Use PostgreSQL for now, or file-based storage as fallback."
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PGLite: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self._connection:
            if self._db_type == DatabaseType.POSTGRESQL:
                await self._connection.close()
            self._connection = None
            self._db_type = DatabaseType.NONE
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connection is not None and self._db_type != DatabaseType.NONE
    
    @property
    def connection(self) -> Any:
        """Get database connection."""
        if not self.is_connected:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._connection
    
    @property
    def db_type(self) -> DatabaseType:
        """Get database type."""
        return self._db_type
    
    async def execute(self, query: str, *args) -> Any:
        """Execute a query."""
        if self._db_type == DatabaseType.POSTGRESQL:
            return await self._connection.execute(query, *args)
        else:
            raise NotImplementedError(f"Execute not implemented for {self._db_type}")
    
    async def fetch(self, query: str, *args) -> list:
        """Fetch rows from a query."""
        if self._db_type == DatabaseType.POSTGRESQL:
            return await self._connection.fetch(query, *args)
        else:
            raise NotImplementedError(f"Fetch not implemented for {self._db_type}")
    
    async def fetchrow(self, query: str, *args) -> Optional[dict]:
        """Fetch a single row from a query."""
        if self._db_type == DatabaseType.POSTGRESQL:
            return await self._connection.fetchrow(query, *args)
        else:
            raise NotImplementedError(f"Fetchrow not implemented for {self._db_type}")
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value from a query."""
        if self._db_type == DatabaseType.POSTGRESQL:
            return await self._connection.fetchval(query, *args)
        else:
            raise NotImplementedError(f"Fetchval not implemented for {self._db_type}")

