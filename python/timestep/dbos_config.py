"""DBOS configuration for Timestep workflows."""

import os
from typing import Optional
from dbos import DBOS, DBOSConfig


def configure_dbos(
    name: str = "timestep",
    system_database_url: Optional[str] = None
) -> None:
    """
    Configure DBOS for Timestep workflows.
    
    Uses PG_CONNECTION_URI for the system database if available,
    otherwise uses a default SQLite database.
    
    Args:
        name: Application name for DBOS (default: "timestep")
        system_database_url: Optional system database URL. If not provided,
            uses PG_CONNECTION_URI environment variable, or defaults to SQLite
    """
    # Get system database URL from parameter, env var, or default to SQLite
    db_url = system_database_url or os.environ.get("PG_CONNECTION_URI")
    
    # If we have a PostgreSQL connection string, use it for DBOS system database
    # DBOS will use the same database but different schema (dbos schema)
    config: DBOSConfig = {
        "name": name,
        "system_database_url": db_url,
    }
    
    DBOS(config=config)


_dbos_configured = False
_dbos_launched = False


def ensure_dbos_launched() -> None:
    """
    Ensure DBOS is configured and launched. Safe to call multiple times.
    
    This should be called before using any DBOS workflows.
    """
    global _dbos_configured, _dbos_launched
    
    if not _dbos_configured:
        configure_dbos()
        _dbos_configured = True
    
    if not _dbos_launched:
        DBOS.launch()
        _dbos_launched = True

