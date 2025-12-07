"""DBOS configuration for Timestep workflows."""

import os
from typing import Optional
from dbos import DBOS, DBOSConfig
from testcontainers.postgres import PostgresContainer


class DBOSContext:
    """Context for DBOS configuration and state."""
    
    def __init__(self):
        self._config: Optional[DBOSConfig] = None
        self._configured = False
        self._launched = False
        self._postgres_container = None
    
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

    def set_config(self, config: DBOSConfig) -> None:
        """Set the DBOS configuration."""
        self._config = config

    def set_configured(self, value: bool) -> None:
        """Set the configured status."""
        self._configured = value

    def set_launched(self, value: bool) -> None:
        """Set the launched status."""
        self._launched = value

    def get_postgres_container(self):
        """Get the PostgreSQL container."""
        return self._postgres_container

    def set_postgres_container(self, container):
        """Set the PostgreSQL container."""
        self._postgres_container = container

    async def cleanup(self) -> None:
        """Clean up PostgreSQL container and DBOS resources."""
        import asyncio
        
        # Shutdown DBOS first to stop all background threads (queue workers, notification listeners)
        # Use shutdown() instead of destroy() to preserve the registry for next test
        if self._launched:
            try:
                # Try shutdown first (if available)
                if hasattr(DBOS, 'shutdown'):
                    DBOS.shutdown()
                else:
                    # Fallback to destroy but don't destroy registry
                    DBOS.destroy(destroy_registry=False)
            except Exception:
                # Ignore errors during shutdown - DBOS might already be shutting down
                pass
            self._launched = False
        
        # Give threads a moment to fully stop
        await asyncio.sleep(0.5)
        
        # Stop container after DBOS is fully shut down
        if self._postgres_container:
            self._postgres_container.stop()
            self._postgres_container = None
        
        self._configured = False
        self._config = None


# Singleton instance
_dbos_context = DBOSContext()


def get_dbos_connection_string() -> Optional[str]:
    """
    Get the DBOS connection string if configured.
    
    Returns:
        Connection string or None if not configured
    """
    return _dbos_context.get_connection_string()


async def configure_dbos(
    name: str = "timestep",
    system_database_url: Optional[str] = None
) -> None:
    """
    Configure DBOS for Timestep workflows.

    Uses PG_CONNECTION_URI for the system database if available,
    otherwise starts a Testcontainers PostgreSQL instance.

    Args:
        name: Application name for DBOS (default: "timestep")
        system_database_url: Optional system database URL. If not provided,
            uses PG_CONNECTION_URI environment variable, or Testcontainers PostgreSQL
    """
    # Only destroy if we're already configured (to avoid destroying on first call)
    if _dbos_context.is_configured:
        try:
            DBOS.destroy(destroy_registry=True)
        except Exception:
            # Ignore if DBOS doesn't exist yet
            pass
    
    # Get system database URL from parameter or env var
    db_url = system_database_url or os.environ.get("PG_CONNECTION_URI")
    
    # If no connection string provided, start Testcontainers PostgreSQL
    if not db_url:
        print("Starting Testcontainers PostgreSQL for DBOS...")
        postgres_container = PostgresContainer("postgres:15")
        postgres_container.start()
        _dbos_context.set_postgres_container(postgres_container)
        
        # Get connection URL from container
        # Testcontainers Python PostgresContainer provides get_connection_url() method
        db_url = postgres_container.get_connection_url()
        print(f"Testcontainers PostgreSQL started: {db_url}")
    
    # DBOS will use the same database but different schema (dbos schema)
    config: DBOSConfig = {
        "name": name,
        "system_database_url": db_url,
    }
    
    DBOS(config=config)
    _dbos_context.set_config(config)
    _dbos_context.set_configured(True)


async def ensure_dbos_launched() -> None:
    """
    Ensure DBOS is configured and launched. Safe to call multiple times.

    This should be called before using any DBOS workflows.
    """
    if not _dbos_context.is_configured:
        await configure_dbos()
    
    if not _dbos_context.is_launched:
        DBOS.launch()
        _dbos_context.set_launched(True)


def is_dbos_launched() -> bool:
    """
    Check if DBOS is launched.
    
    Returns:
        True if DBOS is launched, False otherwise
    """
    return _dbos_context.is_launched


async def cleanup_dbos() -> None:
    """
    Clean up PostgreSQL container and DBOS resources.
    Call this when shutting down the application.
    """
    await _dbos_context.cleanup()
