"""Database connection management for Timestep."""

import os
from pathlib import Path
from typing import Optional, Any
from enum import Enum
from .app_dir import get_pglite_dir
from .postgres_helper import try_local_postgres
from .pglite_sidecar.client import PGliteSidecarClient


class DatabaseType(Enum):
    """Database type enumeration."""
    POSTGRESQL = "postgresql"
    PGLITE = "pglite"
    NONE = "none"


class PGLiteConnection:
    """PGLite connection wrapper using sidecar process."""
    
    def __init__(self, pglite_path: str):
        self.pglite_path = Path(pglite_path)
        self.pglite_path.mkdir(parents=True, exist_ok=True)
        self._sidecar: Optional[PGliteSidecarClient] = None
        self._started = False
    
    async def _ensure_started(self) -> None:
        """Ensure the sidecar is started."""
        if not self._started:
            self._sidecar = PGliteSidecarClient(str(self.pglite_path))
            await self._sidecar.start()
            self._started = True
    
    async def query(self, sql: str, params: Optional[list] = None) -> dict:
        """Execute a query via PGLite sidecar."""
        await self._ensure_started()
        if not self._sidecar:
            raise RuntimeError("Sidecar not initialized")
        return await self._sidecar.query(sql, params)
    
    async def execute(self, sql: str, *args) -> None:
        """Execute a query (for DDL statements or DML with parameters)."""
        await self._ensure_started()
        if not self._sidecar:
            raise RuntimeError("Sidecar not initialized")
        if args:
            await self._sidecar.execute(sql, list(args))
        else:
            await self._sidecar.execute(sql, None)
    
    async def fetch(self, sql: str, *args) -> list:
        """Fetch rows from a query."""
        await self._ensure_started()
        if not self._sidecar:
            raise RuntimeError("Sidecar not initialized")
        return await self._sidecar.fetch(sql, list(args) if args else None)
    
    async def fetchrow(self, sql: str, *args) -> Optional[dict]:
        """Fetch a single row from a query."""
        await self._ensure_started()
        if not self._sidecar:
            raise RuntimeError("Sidecar not initialized")
        return await self._sidecar.fetchrow(sql, list(args) if args else None)
    
    async def fetchval(self, sql: str, *args) -> Any:
        """Fetch a single value from a query."""
        await self._ensure_started()
        if not self._sidecar:
            raise RuntimeError("Sidecar not initialized")
        return await self._sidecar.fetchval(sql, list(args) if args else None)
    
    async def close(self) -> None:
        """Close connection and stop sidecar."""
        if self._sidecar:
            await self._sidecar.stop()
            self._sidecar = None
            self._started = False


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
            use_pglite: Whether to use PGLite (defaults to True if no connection string)
            pglite_path: Path for PGLite data directory
        """
        self.connection_string = connection_string or os.environ.get("TIMESTEP_DB_URL")
        # Default to PGLite if no connection string provided
        if use_pglite is None:
            use_pglite = not self.connection_string or os.environ.get("TIMESTEP_USE_PGLITE", "").lower() == "true"
        self.use_pglite = use_pglite
        # Use app directory for PGLite storage if path not explicitly provided
        if pglite_path:
            self.pglite_path = pglite_path
        else:
            env_path = os.environ.get("TIMESTEP_PGLITE_PATH")
            if env_path:
                self.pglite_path = env_path
            else:
                # Default to app directory
                self.pglite_path = str(get_pglite_dir())
        self._connection: Optional[Any] = None
        self._db_type: DatabaseType = DatabaseType.NONE
        
    async def connect(self) -> bool:
        """
        Connect to database. Returns True if successful, False otherwise.
        
        Connection priority:
        1. Explicit connection string (TIMESTEP_DB_URL or connection_string parameter)
        2. Local Postgres on localhost:5432 (auto-detect)
        3. PGLite with sidecar (fallback)
        
        Returns:
            True if connection successful, False otherwise
        """
        # 1. Try explicit connection string first (remote or local)
        if self.connection_string and not self.use_pglite:
            try:
                return await self._connect_postgresql()
            except Exception as e:
                # If explicit connection fails, don't fall back - raise the error
                raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
        
        # 2. Try local Postgres (auto-detect)
        if not self.use_pglite:
            try:
                local_conn = await try_local_postgres()
                if local_conn:
                    self.connection_string = local_conn
                    return await self._connect_postgresql()
            except Exception:
                # Local Postgres not available, continue to fallback
                pass
        
        # 3. Fall back to PGLite (with sidecar for performance)
        if self.use_pglite or not self.connection_string:
            try:
                return await self._connect_pglite()
            except Exception as e:
                raise ConnectionError(f"Failed to connect to PGLite: {e}")
        
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
            
            # Initialize schema
            from .schema import initialize_schema
            await initialize_schema(self._connection)
            
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
            self._connection = PGLiteConnection(self.pglite_path)
            self._db_type = DatabaseType.PGLITE
            
            # Test connection (this will start the sidecar)
            await self._connection.query("SELECT 1")
            
            # Initialize schema
            from .schema import initialize_schema
            await initialize_schema(self._connection)
            
            return True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PGLite: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self._connection:
            if self._db_type == DatabaseType.POSTGRESQL:
                await self._connection.close()
            elif self._db_type == DatabaseType.PGLITE:
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
        elif self._db_type == DatabaseType.PGLITE:
            # PGLite uses query method which returns result
            result = await self._connection.query(query, list(args) if args else None)
            # Return rowCount if available, otherwise None
            return result.get('rowCount', None) if isinstance(result, dict) else None
        else:
            raise NotImplementedError(f"Execute not implemented for {self._db_type}")
    
    async def fetch(self, query: str, *args) -> list:
        """Fetch rows from a query."""
        if self._db_type == DatabaseType.POSTGRESQL:
            return await self._connection.fetch(query, *args)
        elif self._db_type == DatabaseType.PGLITE:
            return await self._connection.fetch(query, *args)
        else:
            raise NotImplementedError(f"Fetch not implemented for {self._db_type}")
    
    async def fetchrow(self, query: str, *args) -> Optional[dict]:
        """Fetch a single row from a query."""
        if self._db_type == DatabaseType.POSTGRESQL:
            return await self._connection.fetchrow(query, *args)
        elif self._db_type == DatabaseType.PGLITE:
            return await self._connection.fetchrow(query, *args)
        else:
            raise NotImplementedError(f"Fetchrow not implemented for {self._db_type}")
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value from a query."""
        if self._db_type == DatabaseType.POSTGRESQL:
            return await self._connection.fetchval(query, *args)
        elif self._db_type == DatabaseType.PGLITE:
            return await self._connection.fetchval(query, *args)
        else:
            raise NotImplementedError(f"Fetchval not implemented for {self._db_type}")

