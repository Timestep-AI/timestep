/** PostgreSQL TaskStore implementation for A2A protocol. */

import { Pool, type PoolClient } from 'pg';
import type { Task, TaskStore } from '@a2a-js/sdk/server';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';

export interface Context {
  id: string;
  created_at: Date;
  updated_at: Date;
  metadata?: Record<string, unknown>;
  parent_context_id?: string;
}

export class PostgresTaskStore implements TaskStore {
  private pool: Pool;
  private agentId?: string;

  constructor(connectionString?: string, agentId?: string) {
    const connString =
      connectionString ||
      process.env.DATABASE_URL ||
      `postgresql://${process.env.POSTGRES_USER || 'timestep'}:${process.env.POSTGRES_PASSWORD || 'timestep'}@${process.env.POSTGRES_HOST || 'localhost'}:${process.env.POSTGRES_PORT || '5432'}/${process.env.POSTGRES_DB || 'timestep'}`;

    this.pool = new Pool({
      connectionString: connString,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    this.pool.on('error', (err) => {
      console.error('Unexpected error on idle client', err);
    });
    
    this.agentId = agentId;
  }

  setAgentId(agentId: string): void {
    /** Set the agent ID for this task store instance. */
    this.agentId = agentId;
  }

  async save(task: Task): Promise<void> {
    const client = await this.pool.connect();
    try {
      // Ensure context exists
      await this.ensureContext(client, task.contextId);

      // Upsert task (openai_messages will be saved separately via saveOpenAIMessages)
      await client.query(
        `INSERT INTO tasks (id, context_id, agent_id, data, updated_at)
         VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
         ON CONFLICT (id) DO UPDATE
         SET data = $4, agent_id = $3, updated_at = CURRENT_TIMESTAMP`,
        [task.id, task.contextId, this.agentId || null, JSON.stringify(task)]
      );

      // Update context updated_at
      await client.query(
        `UPDATE contexts SET updated_at = CURRENT_TIMESTAMP WHERE id = $1`,
        [task.contextId]
      );
    } finally {
      client.release();
    }
  }

  async saveOpenAIMessages(
    taskId: string,
    messages: ChatCompletionMessageParam[]
  ): Promise<void> {
    /** Save OpenAI messages for a task. */
    const client = await this.pool.connect();
    try {
      await client.query(
        `UPDATE tasks SET openai_messages = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2`,
        [JSON.stringify(messages), taskId]
      );
    } finally {
      client.release();
    }
  }

  async getOpenAIMessages(
    taskId: string
  ): Promise<ChatCompletionMessageParam[] | undefined> {
    /** Get OpenAI messages for a task. */
    const client = await this.pool.connect();
    try {
      const result = await client.query(
        'SELECT openai_messages FROM tasks WHERE id = $1',
        [taskId]
      );
      if (result.rows.length === 0 || !result.rows[0].openai_messages) {
        return undefined;
      }
      return result.rows[0].openai_messages as ChatCompletionMessageParam[];
    } finally {
      client.release();
    }
  }

  async getOpenAIMessagesByContextId(
    contextId: string
  ): Promise<ChatCompletionMessageParam[]> {
    /** Get all OpenAI messages for a context. Use the most recent save (by updated_at) since messages are saved incrementally. */
    const client = await this.pool.connect();
    try {
      // Get the most recently updated task's messages - this will have the most complete conversation
      const result = await client.query(
        `SELECT openai_messages, id as task_id FROM tasks 
         WHERE context_id = $1 AND openai_messages IS NOT NULL
         ORDER BY updated_at DESC
         LIMIT 1`,
        [contextId]
      );
      
      if (result.rows.length === 0) {
        return [];
      }
      
      const messages = result.rows[0].openai_messages as ChatCompletionMessageParam[];
      if (messages && Array.isArray(messages)) {
        // Store taskId in each message for later retrieval
        const taskId = result.rows[0].task_id;
        // Add taskId as a property on each message (we'll extract it later)
        return messages.map(msg => {
          // Store taskId in a way we can retrieve it
          return { ...msg, _taskId: taskId } as ChatCompletionMessageParam & { _taskId?: string };
        });
      }
      
      return [];
    } finally {
      client.release();
    }
  }

  async load(taskId: string): Promise<Task | undefined> {
    const client = await this.pool.connect();
    try {
      const result = await client.query('SELECT data FROM tasks WHERE id = $1', [
        taskId,
      ]);
      if (result.rows.length === 0) {
        return undefined;
      }
      return result.rows[0].data as Task;
    } finally {
      client.release();
    }
  }

  async loadByContextId(contextId: string): Promise<Task[]> {
    const client = await this.pool.connect();
    try {
      const result = await client.query(
        'SELECT data FROM tasks WHERE context_id = $1 ORDER BY created_at ASC',
        [contextId]
      );
      return result.rows.map((row) => row.data as Task);
    } finally {
      client.release();
    }
  }

  async listContexts(parentId?: string): Promise<Context[]> {
    /** List all contexts, optionally filtered by parent_id. */
    const client = await this.pool.connect();
    try {
      let query = `SELECT id, created_at, updated_at, metadata, parent_context_id
         FROM contexts`;
      const params: unknown[] = [];
      
      if (parentId) {
        query += ` WHERE parent_context_id = $1`;
        params.push(parentId);
      }
      
      query += ` ORDER BY updated_at DESC`;
      
      const result = await client.query(query, params);
      return result.rows.map((row) => ({
        id: row.id,
        created_at: row.created_at,
        updated_at: row.updated_at,
        metadata: row.metadata || undefined,
        parent_context_id: row.parent_context_id || undefined,
      }));
    } finally {
      client.release();
    }
  }

  async createContext(metadata?: Record<string, unknown>, parentContextId?: string): Promise<Context> {
    /** Create a new context, optionally with a parent. */
    const client = await this.pool.connect();
    try {
      const result = await client.query(
        `INSERT INTO contexts (metadata, parent_context_id)
         VALUES ($1, $2)
         RETURNING id, created_at, updated_at, metadata, parent_context_id`,
        [metadata ? JSON.stringify(metadata) : null, parentContextId || null]
      );
      const row = result.rows[0];
      return {
        id: row.id,
        created_at: row.created_at,
        updated_at: row.updated_at,
        metadata: row.metadata || undefined,
        parent_context_id: row.parent_context_id || undefined,
      };
    } finally {
      client.release();
    }
  }

  private async ensureContext(
    client: PoolClient,
    contextId: string
  ): Promise<void> {
    const result = await client.query(
      'SELECT id FROM contexts WHERE id = $1',
      [contextId]
    );
    if (result.rows.length === 0) {
      await client.query('INSERT INTO contexts (id) VALUES ($1)', [contextId]);
    }
  }

  async getContext(contextId: string): Promise<Context | null> {
    /** Get a single context by ID. */
    const client = await this.pool.connect();
    try {
      const result = await client.query(
        `SELECT id, created_at, updated_at, metadata, parent_context_id
         FROM contexts
         WHERE id = $1`,
        [contextId]
      );
      if (result.rows.length === 0) {
        return null;
      }
      const row = result.rows[0];
      return {
        id: row.id,
        created_at: row.created_at,
        updated_at: row.updated_at,
        metadata: row.metadata || undefined,
        parent_context_id: row.parent_context_id || undefined,
      };
    } finally {
      client.release();
    }
  }

  async updateContext(contextId: string, updates: { parent_context_id?: string }): Promise<Context> {
    /** Update context fields. */
    const client = await this.pool.connect();
    try {
      const updatesList: string[] = [];
      const params: unknown[] = [];
      let paramIndex = 1;

      if (updates.parent_context_id !== undefined) {
        updatesList.push(`parent_context_id = $${paramIndex}`);
        params.push(updates.parent_context_id || null);
        paramIndex++;
      }

      if (updatesList.length === 0) {
        // No updates, just return existing context
        const existing = await this.getContext(contextId);
        if (!existing) {
          throw new Error(`Context not found: ${contextId}`);
        }
        return existing;
      }

      updatesList.push(`updated_at = CURRENT_TIMESTAMP`);
      params.push(contextId);

      const result = await client.query(
        `UPDATE contexts
         SET ${updatesList.join(', ')}
         WHERE id = $${paramIndex}
         RETURNING id, created_at, updated_at, metadata, parent_context_id`,
        params
      );

      if (result.rows.length === 0) {
        throw new Error(`Context not found: ${contextId}`);
      }

      const row = result.rows[0];
      return {
        id: row.id,
        created_at: row.created_at,
        updated_at: row.updated_at,
        metadata: row.metadata || undefined,
        parent_context_id: row.parent_context_id || undefined,
      };
    } finally {
      client.release();
    }
  }

  async deleteContext(contextId: string): Promise<void> {
    /** Delete a context and all its tasks (CASCADE will handle tasks). */
    const client = await this.pool.connect();
    try {
      await client.query('DELETE FROM contexts WHERE id = $1', [contextId]);
    } finally {
      client.release();
    }
  }

  async close(): Promise<void> {
    await this.pool.end();
  }
}

