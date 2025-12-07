/** Session store for saving and loading Session configurations from database. */

import { randomUUID } from 'crypto';
import type { Session } from '@openai/agents';
import type { DatabaseConnection } from './db_connection.ts';

export async function saveSession(session: Session, db: DatabaseConnection): Promise<string> {
  /**
   * Save a session configuration to the database.
   *
   * @param session - The Session object to save
   * @param db - DatabaseConnection instance
   * @returns The session_id (UUID as string) - this is the database ID, not the session's internal ID
   */
  // Get the session's internal ID (e.g., conversation_id for OpenAI sessions)
  // For OpenAIConversationsSession, we need to call getSessionId() to get/create the conversation_id
  let sessionInternalId: string | null = null;
  if ('getSessionId' in session && typeof session.getSessionId === 'function') {
    try {
      sessionInternalId = await (session as any).getSessionId();
    } catch (e) {
      // If getSessionId fails, try other methods
    }
  }
  if (!sessionInternalId) {
    if ('conversationId' in session && session.conversationId) {
      sessionInternalId = session.conversationId as string;
    } else if ('sessionId' in session && session.sessionId) {
      sessionInternalId = session.sessionId as string;
    } else if ('id' in session && session.id) {
      sessionInternalId = session.id as string;
    }
  }

  // Determine session type from class name
  const sessionType = session.constructor?.name || 'UnknownSession';

  // Extract config data
  const configData: any = {};
  if (typeof session === 'object') {
    for (const [key, value] of Object.entries(session)) {
      if (!key.startsWith('_') && typeof value !== 'function') {
        try {
          // Try to serialize the value
          JSON.stringify(value);
          configData[key] = value;
        } catch (e) {
          // Skip non-serializable values
        }
      }
    }
  }

  // Check if session already exists by session_id
  let existing: any = null;
  if (sessionInternalId) {
    const existingResult = await db.query(
      `
      SELECT id FROM sessions WHERE session_id = $1
      `,
      [sessionInternalId]
    );
    if (existingResult.rows && existingResult.rows.length > 0) {
      existing = existingResult.rows[0];
    }
  }

  if (existing) {
    // Update existing session
    await db.query(
      `
      UPDATE sessions
      SET session_type = $1, config_data = $2
      WHERE id = $3
      `,
      [sessionType, JSON.stringify(configData), existing.id]
    );
    return existing.id;
  } else {
    // Insert new session
    const sessionDbId = randomUUID();
    await db.query(
      `
      INSERT INTO sessions (id, session_id, session_type, config_data)
      VALUES ($1, $2, $3, $4)
      `,
      [sessionDbId, sessionInternalId || sessionDbId, sessionType, JSON.stringify(configData)]
    );
    return sessionDbId;
  }
}

export async function loadSession(
  sessionId: string,
  db: DatabaseConnection
): Promise<{ id: string; session_id: string; session_type: string; config_data: any } | null> {
  /**
   * Load a session configuration from the database.
   *
   * Note: This function cannot fully reconstruct Session objects as they often require
   * runtime connections (e.g., to OpenAI API). This function returns the session data,
   * but the caller must reconstruct the Session object using the appropriate constructor.
   *
   * @param sessionId - The session ID (can be either the database UUID or the session's internal ID)
   * @param db - DatabaseConnection instance
   * @returns A dict with session data, or null if not found
   */
  // Try to load by database ID first
  let sessionResult = await db.query(
    `
    SELECT * FROM sessions WHERE id = $1
    `,
    [sessionId]
  );

  // If not found, try by session_id
  if (!sessionResult.rows || sessionResult.rows.length === 0) {
    sessionResult = await db.query(
      `
      SELECT * FROM sessions WHERE session_id = $1
      `,
      [sessionId]
    );
  }

  if (!sessionResult.rows || sessionResult.rows.length === 0) {
    return null;
  }

  const sessionRow = sessionResult.rows[0];

  // Return session data as object
  // PostgreSQL JSON columns are returned as objects by pg driver, not strings
  let configData: any = {};
  if (sessionRow.config_data) {
    if (typeof sessionRow.config_data === 'string') {
      try {
        configData = JSON.parse(sessionRow.config_data);
      } catch {
        configData = {};
      }
    } else if (typeof sessionRow.config_data === 'object' && sessionRow.config_data !== null) {
      configData = sessionRow.config_data;
    }
  }

  return {
    id: sessionRow.id,
    session_id: sessionRow.session_id,
    session_type: sessionRow.session_type,
    config_data: configData,
  };
}

