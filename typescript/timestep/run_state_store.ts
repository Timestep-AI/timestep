/** RunStateStore implementation using PGLite by default. */

import { DatabaseConnection } from './db_connection.ts';
import { Agent, RunState } from '@openai/agents';
import { getPgliteDir } from './app_dir.ts';

export interface RunStateStoreOptions {
  agent: Agent;
  runId?: string;
  sessionId?: string;
  connectionString?: string;
  usePglite?: boolean;
  pglitePath?: string;
}

export class RunStateStore {
  private static readonly SCHEMA_VERSION = '1.0';

  private agent: Agent;
  private runId?: string;
  private db: DatabaseConnection;
  private connected: boolean = false;

  constructor(options: RunStateStoreOptions) {
    if (!options.agent) {
      throw new Error('agent is required');
    }

    this.agent = options.agent;
    this.runId = options.runId || options.sessionId;
    
    // Use session ID in PGLite path to avoid concurrent access issues
    // Each session gets its own database file
    // This must match Python's get_pglite_dir() implementation exactly
    let pglitePath = options.pglitePath;
    if (!pglitePath && !options.connectionString && (options.usePglite !== false)) {
      const sessionId = options.sessionId || options.runId || 'default';
      pglitePath = getPgliteDir({ sessionId });
    }
    
    // Default to PGLite if no connection string provided
    this.db = new DatabaseConnection({
      connectionString: options.connectionString,
      usePglite: options.usePglite,  // undefined means auto-detect (defaults to PGLite)
      pglitePath: pglitePath,
    });
  }

  private async ensureConnected(): Promise<void> {
    if (!this.connected) {
      const connected = await this.db.connect();
      if (!connected) {
        throw new Error(
          'Failed to connect to database. ' +
          'Check TIMESTEP_DB_URL environment variable or ensure PGLite dependencies are installed.'
        );
      }
      this.connected = true;
    }
  }

  private async ensureRunId(): Promise<string> {
    if (this.runId) {
      return this.runId;
    }

    // Generate a new run_id
    this.runId = crypto.randomUUID();
    await this.ensureConnected();

    return this.runId;
  }

  async save(state: any): Promise<void> {
    await this.ensureConnected();
    const runId = await this.ensureRunId();

    // Convert state to JSON string
    const stateString = state.toString();
    const stateJson = JSON.parse(stateString);

    // Determine state type
    const stateType = stateJson.interruptions?.length > 0 ? 'interrupted' : 'checkpoint';

    // Mark previous states as inactive
    await this.db.query(
      `UPDATE run_states
       SET is_active = false
       WHERE run_id = $1 AND is_active = true`,
      [runId]
    );

    // Insert new state
    await this.db.query(
      `INSERT INTO run_states (run_id, state_type, schema_version, state_data, is_active)
       VALUES ($1, $2, $3, $4, true)`,
      [runId, stateType, RunStateStore.SCHEMA_VERSION, JSON.stringify(stateJson)]
    );
  }

  async load(): Promise<any> {
    await this.ensureConnected();
    const runId = await this.ensureRunId();

    // Fetch active state
    const result = await this.db.query(
      `SELECT state_data, state_type, created_at
       FROM run_states
       WHERE run_id = $1 AND is_active = true
       ORDER BY created_at DESC
       LIMIT 1`,
      [runId]
    );

    if (!result.rows || result.rows.length === 0) {
      throw new Error(
        `No active state found for run_id: ${runId}. ` +
        "Make sure you've saved a state first."
      );
    }

    const row = result.rows[0];

    // Update resumed_at timestamp
    await this.db.query(
      `UPDATE run_states
       SET resumed_at = NOW()
       WHERE run_id = $1 AND is_active = true`,
      [runId]
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
    if (!this.runId) {
      return;
    }

    try {
      await this.ensureConnected();
      await this.db.query(
        `UPDATE run_states
         SET is_active = false
         WHERE run_id = $1`,
        [this.runId]
      );
    } catch (e) {
      // If database is not available, silently fail (graceful degradation)
    }
  }

  async close(): Promise<void> {
    await this.db.disconnect();
    this.connected = false;
  }
}

