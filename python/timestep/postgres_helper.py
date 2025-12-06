"""Helper functions for local Postgres detection and setup."""

import os
import asyncio
from typing import Optional


async def get_local_postgres_connection(
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: Optional[str] = None,
    database: str = "timestep"
) -> Optional[str]:
    """
    Try to connect to local Postgres and create database if needed.
    
    Args:
        host: Postgres host (default: localhost)
        port: Postgres port (default: 5432)
        user: Postgres user (default: postgres)
        password: Postgres password (optional, can use PGPASSWORD env var)
        database: Database name to use/create (default: timestep)
    
    Returns:
        Connection string if successful, None otherwise
    """
    try:
        import asyncpg
    except ImportError:
        return None
    
    # Build connection string for postgres database (to create timestep DB)
    postgres_conn_str = f"postgresql://{user}"
    if password:
        postgres_conn_str += f":{password}"
    postgres_conn_str += f"@{host}:{port}/postgres"
    
    # Also try PGPASSWORD environment variable
    if not password:
        pgpassword = os.environ.get("PGPASSWORD")
        if pgpassword:
            postgres_conn_str = f"postgresql://{user}:{pgpassword}@{host}:{port}/postgres"
    
    try:
        # Try to connect to postgres database
        conn = await asyncpg.connect(postgres_conn_str, timeout=2.0)
        
        # Check if timestep database exists
        db_exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            database
        )
        
        if not db_exists:
            # Create the database
            # Note: CREATE DATABASE cannot be run in a transaction
            await conn.execute(f'CREATE DATABASE "{database}"')
        
        await conn.close()
        
        # Return connection string for the timestep database
        result_conn_str = f"postgresql://{user}"
        if password:
            result_conn_str += f":{password}"
        elif os.environ.get("PGPASSWORD"):
            result_conn_str += f":{os.environ.get('PGPASSWORD')}"
        result_conn_str += f"@{host}:{port}/{database}"
        
        return result_conn_str
        
    except (asyncpg.exceptions.InvalidPasswordError, 
            asyncpg.exceptions.InvalidAuthorizationSpecificationError):
        # Authentication failed - Postgres is running but credentials are wrong
        # Don't auto-create in this case
        return None
    except (asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.TooManyConnectionsError,
            OSError,
            asyncio.TimeoutError):
        # Postgres is not running or not accessible
        return None
    except Exception:
        # Any other error - assume Postgres is not available
        return None


async def try_local_postgres() -> Optional[str]:
    """
    Try to connect to local Postgres with common defaults.
    
    Returns:
        Connection string if successful, None otherwise
    """
    # Try with no password first (common for local dev)
    conn_str = await get_local_postgres_connection(password=None)
    if conn_str:
        return conn_str
    
    # Try with common passwords
    for password in ["postgres", "password", ""]:
        conn_str = await get_local_postgres_connection(password=password)
        if conn_str:
            return conn_str
    
    return None

