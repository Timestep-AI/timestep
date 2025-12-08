"""DBOS configuration for Timestep workflows."""

import os
from typing import Optional
from dbos import DBOS, DBOSConfig


class DBOSContext:
    """Context for DBOS configuration and state."""
    
    def __init__(self):
        self._config: Optional[DBOSConfig] = None
        self._configured = False
        self._launched = False
    
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

    async def cleanup(self) -> None:
        """Clean up DBOS resources."""
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
        
        
        # Reset configuration flags after cleanup
        # This ensures that if another test runs, it will properly configure DBOS
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

    Uses PG_CONNECTION_URI environment variable for the system database.
    For tests, run 'make test-setup' to start the test database.

    Args:
        name: Application name for DBOS (default: "timestep")
        system_database_url: Optional system database URL. If not provided,
            uses PG_CONNECTION_URI environment variable
    """
    # Note: configure_dbos() should be called after DBOS.destroy() in test fixtures
    # We don't destroy here to allow the caller to control when destruction happens
    # This follows the DBOS testing pattern: destroy -> configure -> reset -> launch
    
    # Get system database URL from parameter or env var
    db_url = system_database_url or os.environ.get("PG_CONNECTION_URI")
    
    if not db_url:
        raise ValueError(
            "PG_CONNECTION_URI not set. Run 'make test-setup' to start the test database, "
            "or set PG_CONNECTION_URI environment variable to your PostgreSQL connection string."
        )
    
    # DBOS will use the same database but different schema (dbos schema)
    config: DBOSConfig = {
        "name": name,
        "system_database_url": db_url,
    }
    
    DBOS(config=config)
    _dbos_context.set_config(config)
    
    # Note: We don't call reset_system_database() here because:
    # 1. It tries to drop the database, which fails if we're connected to it
    # 2. DBOS.launch() will handle schema initialization and migrations automatically
    # If you need to reset the system database, do it before calling configure_dbos()
    # by calling DBOS.destroy() and DBOS.reset_system_database() manually
    
    _dbos_context.set_configured(True)


async def ensure_dbos_launched() -> None:
    """
    Ensure DBOS is configured and launched. Safe to call multiple times.

    This should be called before using any DBOS workflows.
    """
    if not _dbos_context.is_configured:
        await configure_dbos()
    
    # Always try to launch DBOS, regardless of our flag.
    # This ensures DBOS is actually launched even if our flag is incorrect.
    # DBOS.launch() should handle being called multiple times gracefully.
    try:
        DBOS.launch()
        _dbos_context.set_launched(True)
    except Exception as e:
        # If launch fails, check if it's because DBOS is already launched
        error_msg = str(e).lower()
        if "already" in error_msg or "launched" in error_msg:
            # DBOS is already launched, just update our flag
            _dbos_context.set_launched(True)
        else:
            # Some other error occurred - if our flag said it was launched,
            # reset it since launch failed
            if _dbos_context.is_launched:
                _dbos_context.set_launched(False)
            raise
    
    # Verify DBOS is actually ready by checking if system database is accessible
    # This handles the case where DBOS.launch() returned early because it thought
    # it was already launched, but DBOS was actually shut down
    try:
        # Try to access the system database property - this will raise if not launched
        sys_db = DBOS._sys_db
        # If we got here, DBOS is actually launched and ready
        _dbos_context.set_launched(True)
    except (AttributeError, Exception) as verify_error:
        # System database is not accessible. This can happen if:
        # 1. DBOS was shut down by another test but DBOS's internal _launched flag wasn't reset
        # 2. DBOS.launch() returned early because it thought it was already launched
        # In this case, we need to force a proper re-initialization
        # First, try to shut down DBOS to reset its internal state
        try:
            if hasattr(DBOS, 'shutdown'):
                DBOS.shutdown()
            else:
                DBOS.destroy(destroy_registry=False)
        except Exception:
            # Ignore shutdown errors - DBOS might not be running
            pass
        
        # Reset our flags since DBOS is not actually ready
        _dbos_context.set_launched(False)
        
        # Now try to launch again - this should properly initialize DBOS
        try:
            DBOS.launch()
            _dbos_context.set_launched(True)
            # Verify it's actually ready now
            _ = DBOS._sys_db
        except Exception as e2:
            error_msg = str(e2).lower()
            if "already" in error_msg or "launched" in error_msg:
                # DBOS says it's already launched after shutdown - this shouldn't happen
                # but if it does, verify it's actually ready
                try:
                    _ = DBOS._sys_db
                    _dbos_context.set_launched(True)
                except Exception:
                    # Still can't access - something is seriously wrong
                    _dbos_context.set_configured(False)
                    raise verify_error
            else:
                # Some other error occurred
                _dbos_context.set_configured(False)
                raise


def is_dbos_launched() -> bool:
    """
    Check if DBOS is launched.
    
    Returns:
        True if DBOS is launched, False otherwise
    """
    return _dbos_context.is_launched


async def cleanup_dbos() -> None:
    """
    Clean up DBOS resources.
    Call this when shutting down the application.
    """
    await _dbos_context.cleanup()
