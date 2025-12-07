"""Database connection management for Timestep."""

import os
from pathlib import Path
from typing import Optional, Any
from enum import Enum
from .app_dir import get_pglite_dir
from .pglite_sidecar.client import PGliteSidecarClient


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
        use_pglite: Optional[bool] = None,
        pglite_path: Optional[str] = None
    ):
        """
        Initialize database connection.
        
        Args:
            connection_string: PostgreSQL connection string (e.g., postgresql://user:pass@host/db)
            use_pglite: Whether to use PGLite (required if connection_string is None)
            pglite_path: Path for PGLite data directory (required if use_pglite is True)
        """
        if connection_string:
            # PostgreSQL mode
            self.connection_string = connection_string
            self.use_pglite = False
            self.pglite_path = None
        elif use_pglite:
            # PGLite mode - path is required
            if not pglite_path:
                raise ValueError("pglite_path is required when use_pglite=True")
            self.connection_string = None
            self.use_pglite = True
            self.pglite_path = pglite_path
        else:
            # No configuration provided
            raise ValueError(
                "Either connection_string must be provided for PostgreSQL, "
                "or use_pglite=True with pglite_path for PGLite"
            )
        self._connection: Optional[Any] = None
        self._db_type: DatabaseType = DatabaseType.NONE
        self._pglite_sidecar: Optional[PGliteSidecarClient] = None
        self._pglite_started = False
        
    async def connect(self) -> bool:
        """
        Connect to database. Returns True if successful, False otherwise.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.connection_string and not self.use_pglite:
            try:
                return await self._connect_postgresql()
            except Exception as e:
                raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
        
        if self.use_pglite:
            try:
                return await self._connect_pglite()
            except Exception as e:
                raise ConnectionError(f"Failed to connect to PGLite: {e}")
        
        raise ValueError("No connection configuration provided")
    
    async def _connect_postgresql(self) -> bool:
        """Connect to PostgreSQL database."""
        try:
            import asyncpg
            
            # Parse connection string
            self._connection = await asyncpg.connect(self.connection_string)
            self._db_type = DatabaseType.POSTGRESQL
            
            # Test connection
            await self._connection.fetchval("SELECT 1")
            
            # Initialize schema (pass self so it uses our unified interface)
            from .schema import initialize_schema
            await initialize_schema(self)
            
            return True
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgreSQL support. "
                "Install it with: pip install asyncpg"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
    
    async def _connect_pglite(self) -> bool:
        """Connect to PGLite database via sidecar."""
        try:
            # Initialize PGLite path
            pglite_path = Path(self.pglite_path)
            pglite_path.mkdir(parents=True, exist_ok=True)
            
            # Start sidecar
            self._pglite_sidecar = PGliteSidecarClient(str(pglite_path))
            await self._pglite_sidecar.start()
            self._pglite_started = True
            
            # Store path for reference, connection is the sidecar
            self._connection = self._pglite_sidecar
            self._db_type = DatabaseType.PGLITE
            
            # Test connection
            await self._pglite_sidecar.query("SELECT 1", None)
            
            # Initialize schema (pass self so it uses our unified interface)
            from .schema import initialize_schema
            await initialize_schema(self)
            
            return True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PGLite: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self._connection:
            if self._db_type == DatabaseType.POSTGRESQL:
                await self._connection.close()
            elif self._db_type == DatabaseType.PGLITE:
                # Stop the sidecar
                if self._pglite_sidecar:
                    await self._pglite_sidecar.stop()
                    self._pglite_sidecar = None
                    self._pglite_started = False
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
        elif self._db_type == DatabaseType.PGLITE:
            if not self._pglite_sidecar:
                raise RuntimeError("PGLite sidecar not initialized")
            # PGLite uses query method which returns result
            params = list(args) if args else None
            result = await self._pglite_sidecar.query(query, params)
            # Return rowCount if available, otherwise None
            return result.get('rowCount', None) if isinstance(result, dict) else None
        else:
            raise NotImplementedError(f"Execute not implemented for {self._db_type}")
    
    async def fetch(self, query: str, *args) -> list:
        """Fetch rows from a query."""
        if self._db_type == DatabaseType.POSTGRESQL:
            return await self._connection.fetch(query, *args)
        elif self._db_type == DatabaseType.PGLITE:
            if not self._pglite_sidecar:
                raise RuntimeError("PGLite sidecar not initialized")
            params = list(args) if args else None
            return await self._pglite_sidecar.fetch(query, params)
        else:
            raise NotImplementedError(f"Fetch not implemented for {self._db_type}")
    
    async def fetchrow(self, query: str, *args) -> Optional[dict]:
        """Fetch a single row from a query."""
        if self._db_type == DatabaseType.POSTGRESQL:
            return await self._connection.fetchrow(query, *args)
        elif self._db_type == DatabaseType.PGLITE:
            if not self._pglite_sidecar:
                raise RuntimeError("PGLite sidecar not initialized")
            params = list(args) if args else None
            return await self._pglite_sidecar.fetchrow(query, params)
        else:
            raise NotImplementedError(f"Fetchrow not implemented for {self._db_type}")
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value from a query."""
        if self._db_type == DatabaseType.POSTGRESQL:
            return await self._connection.fetchval(query, *args)
        elif self._db_type == DatabaseType.PGLITE:
            if not self._pglite_sidecar:
                raise RuntimeError("PGLite sidecar not initialized")
            params = list(args) if args else None
            return await self._pglite_sidecar.fetchval(query, params)
        else:
            raise NotImplementedError(f"Fetchval not implemented for {self._db_type}")

