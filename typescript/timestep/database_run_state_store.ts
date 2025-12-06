/** Database-backed RunStateStore implementation. */

import { DatabaseConnection } from './db_connection.ts';
import { Agent, RunState } from '@openai/agents';

export interface DatabaseRunStateStoreOptions {
  runId?: string;
  agent: Agent;
  connectionString?: string;
  usePglite?: boolean;
  pglitePath?: string;
  sessionId?: string;
}

export class DatabaseRunStateStore {
  private static readonly SCHEMA_VERSION = '1.0';

  private agent: Agent;
  private runId?: string;
  private db: DatabaseConnection;
  private connected: boolean = false;

  constructor(options: DatabaseRunStateStoreOptions) {
    if (!options.agent) {
      throw new Error('agent is required');
    }

    this.agent = options.agent;
    this.runId = options.runId || options.sessionId;
    this.db = new DatabaseConnection({
      connectionString: options.connectionString,
      usePglite: options.usePglite,
      pglitePath: options.pglitePath,
    });
  }

  private async ensureConnected(): Promise<void> {
    if (!this.connected) {
      const connected = await this.db.connect();
      if (!connected) {
        throw new Error(
          'Failed to connect to database. ' +
          'Check TIMESTEP_DB_URL environment variable or use file-based storage.'
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
    // For MVP, we'll use a simple UUID generation
    // In production, you'd want to use a proper UUID library
    this.runId = crypto.randomUUID();

    // For MVP, we'll create a minimal run record if the runs table exists
    // Otherwise, we'll use the run_id directly
    try {
      await this.ensureConnected();
      // Check if runs table exists and create a minimal run if needed
      // This is a simplified approach for MVP
    } catch (e) {
      // If we can't create a run record, we'll still use the run_id
      // The database constraint will need to be handled at the application level
    }

    return this.runId;
  }

  async save(state: any): Promise<void> {
    await this.ensureConnected();
    const runId = await this.ensureRunId();

    // Convert state to JSON string
    // RunState has a toString() method that returns JSON string
    const stateString = state.toString();
    const stateJson = JSON.parse(stateString);

    // Determine state type
    // For MVP, we'll use 'interrupted' if there are interruptions, otherwise 'checkpoint'
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
      [runId, stateType, DatabaseRunStateStore.SCHEMA_VERSION, JSON.stringify(stateJson)]
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

