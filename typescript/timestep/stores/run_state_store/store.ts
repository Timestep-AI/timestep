/** RunStateStore implementation using PostgreSQL. */

import { DatabaseConnection } from '../shared/db_connection.ts';
import { Agent, RunState } from '@openai/agents';

export interface RunStateStoreOptions {
  agent: Agent;
  sessionId?: string;
  connectionString?: string;
}

export class RunStateStore {
  private static readonly SCHEMA_VERSION = '1.0';

  private agent: Agent;
  private sessionId?: string;
  private db: DatabaseConnection | null = null;
  private connected: boolean = false;
  private _connectionString?: string;

  constructor(options: RunStateStoreOptions) {
    if (!options.agent) {
      throw new Error('agent is required');
    }

    this.agent = options.agent;
    this.sessionId = options.sessionId;
    
    // Store options for lazy initialization
    this._connectionString = options.connectionString;
    this.db = null as any; // Will be initialized in ensureConnected
  }

  private async _initializeDatabase(): Promise<void> {
    if (this.db) {
      return; // Already initialized
    }

    // Get connection string from DBOS if not explicitly provided
    let connectionString = this._connectionString;
    if (!connectionString) {
      const { getDBOSConnectionString, configureDBOS } = await import('../../config/dbos_config.ts');
      connectionString = getDBOSConnectionString();
      
      // If DBOS isn't configured, configure it (will use PG_CONNECTION_URI if available)
      if (!connectionString) {
        await configureDBOS();
        connectionString = getDBOSConnectionString();
      }
    }
    
    if (!connectionString) {
      throw new Error(
        'No connection string provided. ' +
        'Either provide connectionString in options or ensure DBOS is configured with PG_CONNECTION_URI.'
      );
    }
    
    // Use PostgreSQL connection string
    this.db = new DatabaseConnection({ connectionString });
  }

  private async ensureConnected(): Promise<void> {
    if (!this.db) {
      await this._initializeDatabase();
    }
    if (!this.connected) {
      const db = this.getDb();
      const connected = await db.connect();
      if (!connected) {
        throw new Error(
          'Failed to connect to database. ' +
          'Check PG_CONNECTION_URI environment variable or ensure DBOS is configured.'
        );
      }
      this.connected = true;
    }
  }

  private getDb(): DatabaseConnection {
    if (!this.db) {
      throw new Error('Database not initialized. Call ensureConnected() first.');
    }
    return this.db;
  }

  private async ensureSessionId(): Promise<string> {
    if (this.sessionId) {
      return this.sessionId;
    }

    // Generate a new session_id
    this.sessionId = crypto.randomUUID();
    await this.ensureConnected();

    return this.sessionId;
  }

  async save(state: any): Promise<void> {
    await this.ensureConnected();
    const sessionId = await this.ensureSessionId();

    // Convert state to JSON string
    const stateString = state.toString();
    const stateJson = JSON.parse(stateString);

    // Determine state type
    const stateType = stateJson.interruptions?.length > 0 ? 'interrupted' : 'checkpoint';

    // Mark previous states as inactive
    await this.getDb().query(
      `UPDATE run_states
       SET is_active = false
       WHERE run_id = $1 AND is_active = true`,
      [sessionId]
    );

    // Insert new state
    await this.getDb().query(
      `INSERT INTO run_states (run_id, state_type, schema_version, state_data, is_active)
       VALUES ($1, $2, $3, $4, true)`,
      [sessionId, stateType, RunStateStore.SCHEMA_VERSION, JSON.stringify(stateJson)]
    );
  }

  async load(): Promise<any> {
    await this.ensureConnected();
    const sessionId = await this.ensureSessionId();

    // Fetch active state
    const result = await this.getDb().query(
      `SELECT state_data, state_type, created_at
       FROM run_states
       WHERE run_id = $1 AND is_active = true
       ORDER BY created_at DESC
       LIMIT 1`,
      [sessionId]
    );

    if (!result.rows || result.rows.length === 0) {
      throw new Error(
        `No active state found for session_id: ${sessionId}. ` +
        "Make sure you've saved a state first."
      );
    }

    const row = result.rows[0];

    // Update resumed_at timestamp
    await this.getDb().query(
      `UPDATE run_states
       SET resumed_at = NOW()
       WHERE run_id = $1 AND is_active = true`,
      [sessionId]
    );

    // Deserialize state
    let stateJson = row.state_data;
    if (typeof stateJson === 'string') {
      stateJson = JSON.parse(stateJson);
    }

    // Convert to RunState
    const stateString = JSON.stringify(stateJson);
    return await RunState.fromString(this.agent, stateString);
  }

  async clear(): Promise<void> {
    if (!this.sessionId) {
      return;
    }

    try {
      await this.ensureConnected();
      await this.getDb().query(
        `UPDATE run_states
         SET is_active = false
         WHERE run_id = $1`,
        [this.sessionId]
      );
    } catch (e) {
      // If database is not available, silently fail (graceful degradation)
    }
  }

  async close(): Promise<void> {
    if (this.db) {
      await this.db.disconnect();
    }
    this.connected = false;
  }
}
