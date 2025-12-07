"""DBOS configuration for Timestep workflows."""

import os
import sys
from pathlib import Path
from typing import Optional
from dbos import DBOS, DBOSConfig
from .app_dir import get_pglite_dir


class DBOSContext:
    """Context for DBOS configuration and state."""
    
    def __init__(self):
        self._config: Optional[DBOSConfig] = None
        self._configured = False
        self._launched = False
        self._pglite_sidecar = None
        self._pglite_socket_server_info: Optional[dict] = None
    
    def get_connection_string(self) -> Optional[str]:
        """
        Get the DBOS connection string if configured.
        
        Returns:
            Connection string or None if not configured
        """
        if self._config:
            return self._config.get("system_database_url")
        return None
    
    @property
    def is_configured(self) -> bool:
        """Check if DBOS is configured."""
        return self._configured
    
    @property
    def is_launched(self) -> bool:
        """Check if DBOS is launched."""
        return self._launched


# Singleton instance
_dbos_context = DBOSContext()


def get_dbos_connection_string() -> Optional[str]:
    """
    Get the DBOS connection string if configured.
    
    Returns:
        Connection string or None if not configured
    """
    return _dbos_context.get_connection_string()


def configure_dbos(
    name: str = "timestep",
    system_database_url: Optional[str] = None
) -> None:
    """
    Configure DBOS for Timestep workflows.
    
    Uses PG_CONNECTION_URI for the system database if available,
    otherwise will use PGLite via socket server (started in ensure_dbos_launched).
    
    Args:
        name: Application name for DBOS (default: "timestep")
        system_database_url: Optional system database URL. If not provided,
            uses PG_CONNECTION_URI environment variable, or PGLite
    """
    # Get system database URL from parameter, env var, or default to PGLite
    db_url = system_database_url or os.environ.get("PG_CONNECTION_URI")
    
    # If no connection string provided, prepare for PGLite socket server
    if not db_url:
        # Use a SEPARATE PGLite database for DBOS system database
        # This is important because PGLite socket server holds an exclusive lock,
        # so we can't use the same instance as RunStateStore (which uses the sidecar)
        pglite_path = get_pglite_dir()
        # Use a different subdirectory to ensure separate instance
        db_path = pglite_path.parent / 'dbos_system' / 'dbos_system'
        
        # Use TCP connection (Unix sockets might have issues with sidecar coordination)
        socket_path = None  # Use TCP for simplicity
        port = 0  # Auto-assign port
        
        # Store socket server info for later startup
        _dbos_context._pglite_socket_server_info = {
            'path': socket_path,
            'port': port,
            'host': '127.0.0.1',
            'db_path': db_path
        }
        
        # Placeholder - will be updated in ensure_dbos_launched
        db_url = "postgresql://postgres:postgres@127.0.0.1:5432/postgres?sslmode=disable"
    
    # DBOS will use the same database but different schema (dbos schema)
    _dbos_context._config = {
        "name": name,
        "system_database_url": db_url,
    }
    
    DBOS(config=_dbos_context._config)
    _dbos_context._configured = True


async def ensure_dbos_launched() -> None:
    """
    Ensure DBOS is configured and launched. Safe to call multiple times.
    
    This should be called before using any DBOS workflows.
    """
    if not _dbos_context._configured:
        configure_dbos()
    
    # If using PGLite, start socket server before launching DBOS
    if _dbos_context._pglite_socket_server_info:
        # Import sidecar client
        from .pglite_sidecar.client import PGliteSidecarClient
        
        # Create or reuse sidecar client
        if _dbos_context._pglite_sidecar is None:
            _dbos_context._pglite_sidecar = PGliteSidecarClient(
                str(_dbos_context._pglite_socket_server_info['db_path'])
            )
            await _dbos_context._pglite_sidecar.start()
        
        # Start socket server and get connection string
        try:
            server_info = await _dbos_context._pglite_sidecar.start_socket_server(
                port=_dbos_context._pglite_socket_server_info['port'],
                host=_dbos_context._pglite_socket_server_info['host'],
                path=_dbos_context._pglite_socket_server_info['path']
            )
            connection_string = server_info.get('connectionString')
            if not connection_string:
                raise RuntimeError("Failed to get connection string from PGLite socket server")
            
            # Reconfigure DBOS with the actual connection string
            _dbos_context._config = {
                "name": "timestep",  # Use default name
                "system_database_url": connection_string,
            }
            DBOS(config=_dbos_context._config)
        except Exception as e:
            raise RuntimeError(f"Failed to start PGLite socket server for DBOS: {e}")
    
    if not _dbos_context._launched:
        DBOS.launch()
        _dbos_context._launched = True

