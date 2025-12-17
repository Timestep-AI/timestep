"""Session store for saving and loading Session configurations from database."""

import json
import uuid
from typing import Optional
from ..shared.db_connection import DatabaseConnection
from ..._vendored_imports import SessionABC


async def save_session(session: SessionABC) -> str:
    """
    Save a session configuration to the database.
    Manages database connection internally.
    
    Args:
        session: The Session object to save
        
    Returns:
        The session_id (UUID as string) - this is the database ID, not the session's internal ID
    """
    # Get connection string
    connection_string = os.environ.get("PG_CONNECTION_URI")
    if not connection_string:
        connection_string = os.environ.get("PG_CONNECTION_URI")
    if not connection_string:
        raise ValueError("DBOS connection string not available")
    
    # Create and manage database connection
    db = DatabaseConnection(connection_string=connection_string)
    await db.connect()
    try:
        return await _save_session_internal(session, db)
    finally:
        await db.disconnect()


async def _save_session_internal(session: SessionABC, db: DatabaseConnection) -> str:
    """
    Internal function that saves a session using an existing database connection.
    
    Args:
        session: The Session object to save
        db: DatabaseConnection instance (already connected)
        
    Returns:
        The session_id (UUID as string) - this is the database ID, not the session's internal ID
    """
    # Get the session's internal ID (e.g., conversation_id for OpenAI sessions)
    # For OpenAIConversationsSession, we need to call _get_session_id() to get/create the conversation_id
    session_internal_id = None
    if hasattr(session, '_get_session_id'):
        try:
            session_internal_id = await session._get_session_id()
        except Exception:
            # If _get_session_id fails, try other methods
            pass
    if not session_internal_id:
        if hasattr(session, 'conversation_id'):
            session_internal_id = session.conversation_id
        elif hasattr(session, '_session_id'):
            session_internal_id = session._session_id
        elif hasattr(session, 'session_id'):
            session_internal_id = session.session_id
        elif hasattr(session, 'id'):
            session_internal_id = session.id
    
    # Determine session type from class name
    session_type = session.__class__.__name__
    
    # Extract config data
    config_data = {}
    if hasattr(session, '__dict__'):
        # Try to serialize session attributes
        for key, value in session.__dict__.items():
            if not key.startswith('_') and not callable(value):
                try:
                    # Try to serialize the value
                    json.dumps(value, default=str)
                    config_data[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable values
                    pass
    
    # Check if session already exists by session_id
    existing = None
    if session_internal_id:
        existing = await db.fetchrow("""
            SELECT id FROM sessions WHERE session_id = $1
        """, session_internal_id)
    
    if existing:
        # Update existing session
        await db.execute("""
            UPDATE sessions
            SET session_type = $1, config_data = $2
            WHERE id = $3
        """, session_type, json.dumps(config_data), existing['id'])
        return existing['id']
    else:
        # Insert new session
        session_db_id = str(uuid.uuid4())
        await db.execute("""
            INSERT INTO sessions (id, session_id, session_type, config_data)
            VALUES ($1, $2, $3, $4)
        """, session_db_id, session_internal_id or str(session_db_id), session_type, json.dumps(config_data))
        return session_db_id


async def load_session(session_id: str) -> Optional[dict]:
    """
    Load a session configuration from the database.
    Manages database connection internally.
    
    Note: This function cannot fully reconstruct Session objects as they often require
    runtime connections (e.g., to OpenAI API). This function returns the session data,
    but the caller must reconstruct the Session object using the appropriate constructor.
    
    Args:
        session_id: The session ID (can be either the database UUID or the session's internal ID)
        
    Returns:
        A dict with session data, or None if not found
    """
    # Get connection string
    connection_string = os.environ.get("PG_CONNECTION_URI")
    if not connection_string:
        connection_string = os.environ.get("PG_CONNECTION_URI")
    if not connection_string:
        raise ValueError("DBOS connection string not available")
    
    # Create and manage database connection
    db = DatabaseConnection(connection_string=connection_string)
    await db.connect()
    try:
        return await _load_session_internal(session_id, db)
    finally:
        await db.disconnect()


async def _load_session_internal(session_id: str, db: DatabaseConnection) -> Optional[dict]:
    """
    Load a session configuration from the database.
    
    Note: This function cannot fully reconstruct Session objects as they often require
    runtime connections (e.g., to OpenAI API). This function returns the session data,
    but the caller must reconstruct the Session object using the appropriate constructor.
    
    Args:
        session_id: The session ID (can be either the database UUID or the session's internal ID)
        db: DatabaseConnection instance
        
    Returns:
        A dict with session data, or None if not found
    """
    # Try to load by database ID first
    session_row = await db.fetchrow("""
        SELECT * FROM sessions WHERE id = $1
    """, session_id)
    
    # If not found, try by session_id
    if not session_row:
        session_row = await db.fetchrow("""
            SELECT * FROM sessions WHERE session_id = $1
        """, session_id)
    
    if not session_row:
        return None
    
    # Return session data as dict
    config_data = json.loads(session_row['config_data']) if session_row['config_data'] else {}
    
    return {
        'id': session_row['id'],
        'session_id': session_row['session_id'],
        'session_type': session_row['session_type'],
        'config_data': config_data,
    }

