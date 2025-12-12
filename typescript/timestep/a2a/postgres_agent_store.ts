/** PostgreSQL AgentStore implementation for managing agents. */

import { Pool } from 'pg';

export interface Agent {
  id: string;
  name: string;
  description?: string;
  tools: string[];
  model: string;
  created_at: Date;
  updated_at: Date;
}

export class PostgresAgentStore {
  private pool: Pool;

  constructor(connectionString?: string) {
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
  }

  async getAgent(agentId: string): Promise<Agent | null> {
    const client = await this.pool.connect();
    try {
      const result = await client.query(
        'SELECT id, name, description, tools, model, created_at, updated_at FROM agents WHERE id = $1',
        [agentId]
      );

      if (result.rows.length === 0) {
        return null;
      }

      const row = result.rows[0];
      return {
        id: row.id,
        name: row.name,
        description: row.description || undefined,
        tools: row.tools || [],
        model: row.model || 'gpt-4.1',
        created_at: row.created_at,
        updated_at: row.updated_at,
      };
    } finally {
      client.release();
    }
  }

  async listAgents(): Promise<Agent[]> {
    const client = await this.pool.connect();
    try {
      const result = await client.query(
        'SELECT id, name, description, tools, model, created_at, updated_at FROM agents ORDER BY created_at ASC'
      );

      return result.rows.map((row) => ({
        id: row.id,
        name: row.name,
        description: row.description || undefined,
        tools: row.tools || [],
        model: row.model || 'gpt-4.1',
        created_at: row.created_at,
        updated_at: row.updated_at,
      }));
    } finally {
      client.release();
    }
  }

  async createAgent(agent: Omit<Agent, 'created_at' | 'updated_at'>): Promise<Agent> {
    const client = await this.pool.connect();
    try {
      const result = await client.query(
        `INSERT INTO agents (id, name, description, tools, model)
         VALUES ($1, $2, $3, $4, $5)
         RETURNING id, name, description, tools, model, created_at, updated_at`,
        [agent.id, agent.name, agent.description || null, JSON.stringify(agent.tools), agent.model]
      );

      const row = result.rows[0];
      return {
        id: row.id,
        name: row.name,
        description: row.description || undefined,
        tools: row.tools || [],
        model: row.model || 'gpt-4.1',
        created_at: row.created_at,
        updated_at: row.updated_at,
      };
    } finally {
      client.release();
    }
  }

  getDefaultAgents(): Array<Omit<Agent, 'created_at' | 'updated_at'>> {
    return [
      {
        id: 'personal-assistant',
        name: 'Personal Assistant',
        description: 'Personal assistant with web search and agent handoff capabilities',
        tools: ['handoff', 'web_search'],
        model: 'gpt-4.1',
      },
      {
        id: 'weather-assistant',
        name: 'Weather Assistant',
        description: 'Specialized weather information agent',
        tools: ['get_weather'],
        model: 'gpt-4.1',
      },
    ];
  }
}

