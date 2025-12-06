"""Database connection management for Timestep."""

import os
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional, Any
from enum import Enum
from .app_dir import get_pglite_dir


class DatabaseType(Enum):
    """Database type enumeration."""
    POSTGRESQL = "postgresql"
    PGLITE = "pglite"
    NONE = "none"


class PGLiteConnection:
    """PGLite connection wrapper using subprocess."""
    
    def __init__(self, pglite_path: str):
        self.pglite_path = Path(pglite_path)
        self.pglite_path.mkdir(parents=True, exist_ok=True)
        self._node_modules_path = None
        
    def _find_pglite_module(self) -> str | None:
        """Find the @electric-sql/pglite module path."""
        import shutil
        
        # Try to find node_modules in common locations
        possible_paths = [
            # Current directory and parent directories
            Path.cwd() / "node_modules" / "@electric-sql" / "pglite",
            Path(__file__).parent.parent.parent / "node_modules" / "@electric-sql" / "pglite",
            # Global npm installation
            Path.home() / ".npm" / "global" / "node_modules" / "@electric-sql" / "pglite",
        ]
        
        # Also check if pglite is installed globally via npm
        try:
            import subprocess
            result = subprocess.run(
                ["npm", "root", "-g"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                global_node_modules = Path(result.stdout.strip())
                possible_paths.append(global_node_modules / "@electric-sql" / "pglite")
        except Exception:
            pass
        
        for path in possible_paths:
            if path.exists() and (path / "package.json").exists():
                return str(path)
        
        # If not found, return None to use require() directly
        return None
        
    async def query(self, sql: str, params: Optional[list] = None) -> dict:
        """Execute a query via PGLite Node.js subprocess."""
        import shutil
        
        # Check if node is available
        node_path = shutil.which('node')
        if not node_path:
            raise RuntimeError(
                "Node.js is required for PGLite support. "
                "Install Node.js from https://nodejs.org/ or use PostgreSQL via TIMESTEP_DB_URL"
            )
        
        # Find PGLite module
        pglite_module_path = self._find_pglite_module()
        
        # Convert path to absolute
        abs_path = str(self.pglite_path.resolve())
        
        # Create a script that tries to load PGLite from various locations
        if pglite_module_path:
            # Use explicit path
            require_path = f"require('{pglite_module_path}')"
        else:
            # Try to require from node_modules or global
            require_path = "require('@electric-sql/pglite')"
        
        script = f"""
        const {{ PGlite }} = {require_path};
        const path = require('path');
        const fs = require('fs');
        
        async function runQuery() {{
            const dbPath = {json.dumps(abs_path)};
            const db = new PGlite(dbPath);
            await db.waitReady;
            
            const sql = {json.dumps(sql)};
            const params = {json.dumps(params or [])};
            
            try {{
                const result = await db.query(sql, params);
                console.log(JSON.stringify({{ success: true, rows: result.rows, rowCount: result.rowCount }}));
            }} catch (error) {{
                console.error(JSON.stringify({{ success: false, error: error.message }}));
                process.exit(1);
            }} finally {{
                await db.close();
            }}
        }}
        
        runQuery().catch(err => {{
            console.error(JSON.stringify({{ success: false, error: err.message }}));
            process.exit(1);
        }});
        """
        
        # Run via node
        process = await asyncio.create_subprocess_exec(
            node_path,
            '-e', script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(__file__).parent.parent.parent)
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            if "Cannot find module" in error_msg:
                raise RuntimeError(
                    "@electric-sql/pglite is not installed. "
                    "Install it with: npm install -g @electric-sql/pglite "
                    "or: npm install @electric-sql/pglite (in the project directory)"
                )
            raise RuntimeError(f"PGLite query failed: {error_msg}")
        
        result = json.loads(stdout.decode())
        if not result.get('success'):
            raise RuntimeError(f"PGLite query error: {result.get('error')}")
        
        return result
    
    async def execute(self, sql: str, *args) -> None:
        """Execute a query (for DDL statements or DML with parameters)."""
        if args:
            await self.query(sql, list(args))
        else:
            await self.query(sql, None)
    
    async def fetch(self, sql: str, *args) -> list:
        """Fetch rows from a query."""
        result = await self.query(sql, list(args) if args else None)
        return result.get('rows', [])
    
    async def fetchrow(self, sql: str, *args) -> Optional[dict]:
        """Fetch a single row from a query."""
        rows = await self.fetch(sql, *args)
        return rows[0] if rows else None
    
    async def fetchval(self, sql: str, *args) -> Any:
        """Fetch a single value from a query."""
        row = await self.fetchrow(sql, *args)
        if row:
            return list(row.values())[0] if row else None
        return None
    
    async def close(self) -> None:
        """Close connection (no-op for subprocess approach)."""
        pass


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
        
        Returns:
            True if connection successful, False otherwise
        """
        # Try PostgreSQL first if connection string is explicitly provided
        if self.connection_string and not self.use_pglite:
            try:
                return await self._connect_postgresql()
            except Exception:
                pass
        
        # Default to PGLite (or if explicitly enabled)
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
        """Connect to PGLite database via subprocess."""
        try:
            self._connection = PGLiteConnection(self.pglite_path)
            self._db_type = DatabaseType.PGLITE
            
            # Test connection
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

