"""DBOS configuration for Timestep workflows."""

import os
import sys
from pathlib import Path
from typing import Optional
from dbos import DBOS, DBOSConfig
from .app_dir import get_pglite_dir

# Global state for PGLite socket server
_pglite_sidecar = None
_pglite_socket_server_info = None


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
    global _pglite_socket_server_info
    
    # Get system database URL from parameter, env var, or default to PGLite
    db_url = system_database_url or os.environ.get("PG_CONNECTION_URI")
    
    # If no connection string provided, prepare for PGLite socket server
    if not db_url:
        # Use a dedicated PGLite database for DBOS system database
        pglite_path = get_pglite_dir()
        db_path = pglite_path / 'dbos_system'
        
        # Use Unix socket for better performance (or TCP if on Windows)
        is_windows = sys.platform == 'win32'
        socket_path = None if is_windows else str(db_path.parent / '.s.PGSQL.5432')
        port = 5432 if is_windows else 0
        
        # Store socket server info for later startup
        _pglite_socket_server_info = {
            'path': socket_path,
            'port': port,
            'host': '127.0.0.1',
            'db_path': db_path
        }
        
        # Placeholder - will be updated in ensure_dbos_launched
        db_url = "postgresql://postgres:postgres@127.0.0.1:5432/postgres?sslmode=disable"
    
    # DBOS will use the same database but different schema (dbos schema)
    config: DBOSConfig = {
        "name": name,
        "system_database_url": db_url,
    }
    
    DBOS(config=config)


_dbos_configured = False
_dbos_launched = False


async def ensure_dbos_launched() -> None:
    """
    Ensure DBOS is configured and launched. Safe to call multiple times.
    
    This should be called before using any DBOS workflows.
    """
    global _dbos_configured, _dbos_launched, _pglite_sidecar, _pglite_socket_server_info
    
    if not _dbos_configured:
        configure_dbos()
        _dbos_configured = True
    
    # If using PGLite, start socket server before launching DBOS
    if _pglite_socket_server_info:
        # Import sidecar client
        from .pglite_sidecar.client import PGliteSidecarClient
        
        # Create or reuse sidecar client
        if _pglite_sidecar is None:
            _pglite_sidecar = PGliteSidecarClient(str(_pglite_socket_server_info['db_path']))
            await _pglite_sidecar.start()
        
        # Start socket server and get connection string
        try:
            server_info = await _pglite_sidecar.start_socket_server(
                port=_pglite_socket_server_info['port'],
                host=_pglite_socket_server_info['host'],
                path=_pglite_socket_server_info['path']
            )
            connection_string = server_info.get('connectionString')
            if not connection_string:
                raise RuntimeError("Failed to get connection string from PGLite socket server")
            
            # Reconfigure DBOS with the actual connection string
            config: DBOSConfig = {
                "name": "timestep",  # Use default name
                "system_database_url": connection_string,
            }
            DBOS(config=config)
        except Exception as e:
            raise RuntimeError(f"Failed to start PGLite socket server for DBOS: {e}")
    
    if not _dbos_launched:
        DBOS.launch()
        _dbos_launched = True

