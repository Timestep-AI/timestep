export { OllamaModel } from './ollama_model.ts';
export { OllamaModelProvider, type OllamaModelProviderOptions } from './ollama_model_provider.ts';
export { MultiModelProvider, MultiModelProviderMap } from './multi_model_provider.ts';
export { webSearch } from './tools.ts';
export { DatabaseRunStateStore } from './database_run_state_store.ts';

import { Agent, Runner, Session, RunState, MaxTurnsExceededError, ModelBehaviorError, UserError, AgentsError } from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';
import * as fs from 'fs/promises';

export class InterruptionException extends Error {
  constructor(message: string = 'Agent execution interrupted for approval') {
    super(message);
    this.name = 'InterruptionException';
  }
}

export class RunStateStore {
  private filePath: string;
  private agent: Agent;

  constructor(filePath: string, agent: Agent) {
    this.filePath = filePath;
    this.agent = agent;
  }

  async save(state: any): Promise<void> {
    await fs.writeFile(this.filePath, state.toString(), 'utf-8');
  }

  async load(): Promise<any> {
    const content = await fs.readFile(this.filePath, 'utf-8');
    const { RunState } = await import('@openai/agents');
    return await RunState.fromString(this.agent, content);
  }

  async clear(): Promise<void> {
    try {
      await fs.unlink(this.filePath);
    } catch {
      // Ignore if file doesn't exist
    }
  }
}

export async function createRunStateStore(
  agent: Agent,
  options: {
    filePath?: string;
    runId?: string;
    sessionId?: string;
    connectionString?: string;
    useDatabase?: boolean;
  }
): Promise<RunStateStore | DatabaseRunStateStore> {
  /**
   * Factory function to create a RunStateStore (file-based or database-backed).
   *
   * Auto-selects the appropriate storage backend:
   * 1. Database if TIMESTEP_DB_URL is set or connectionString is provided
   * 2. File-based storage as fallback
   *
   * @param agent - Agent instance
   * @param options - Configuration options
   * @returns RunStateStore or DatabaseRunStateStore instance
   */
  const { filePath, runId, sessionId, connectionString, useDatabase } = options;

  // Auto-detect: try database first if connection string is available
  const shouldUseDatabase =
    useDatabase !== undefined
      ? useDatabase
      : Boolean(connectionString || Deno.env.get('TIMESTEP_DB_URL'));

  if (shouldUseDatabase) {
    try {
      const store = new DatabaseRunStateStore({
        runId,
        agent,
        connectionString,
        sessionId,
      });
      // Test connection by trying to ensure connected
      // This will throw if connection fails
      await (store as any).ensureConnected();
      return store;
    } catch (e) {
      // Fallback to file-based if database connection fails
      if (!filePath) {
        throw new Error(
          'Database connection failed and no filePath provided. ' +
          'Either provide a valid database connection or a filePath for file-based storage.'
        );
      }
      return new RunStateStore(filePath, agent);
    }
  } else {
    // Use file-based storage
    if (!filePath) {
      throw new Error('filePath is required for file-based storage');
    }
    return new RunStateStore(filePath, agent);
  }
}

export async function consumeResult(result: any): Promise<any> {
  /**
   * Consume all events from a result (streaming or non-streaming).
   *
   * @param result - RunResult or StreamedRunResult from runAgent
   * @returns The same result object after consuming stream (if applicable)
   */
  if ('toTextStream' in result) {
    const stream = result.toTextStream({ compatibleWithNodeStreams: true });
    for await (const _ of stream) {
      // Consume chunks
    }
    await result.completed;
  }
  return result;
}

export async function runAgent(
  agent: Agent,
  runInput: AgentInputItem[] | RunState<any, any>,
  session: Session,
  stream: boolean
): Promise<any> {
  const runner = new Runner();

  const sessionInputCallback = async (existingItems: AgentInputItem[], newInput: AgentInputItem[]): Promise<AgentInputItem[]> => {
    return [...existingItems, ...newInput];
  };

  try {
    if (stream) {
      const result = await runner.run(agent, runInput, {
        session,
        sessionInputCallback,
        stream: true
      });
      return result;
    } else {
      const result = await runner.run(agent, runInput, {
        session,
        sessionInputCallback
      });
      return result;
    }
  } catch (e) {
    if (e instanceof MaxTurnsExceededError) {
      console.error('MaxTurnsExceededError:', e.message);
      throw e;
    } else if (e instanceof ModelBehaviorError) {
      console.error('ModelBehaviorError:', e.message);
      throw e;
    } else if (e instanceof UserError) {
      console.error('UserError:', e.message);
      throw e;
    } else if (e instanceof AgentsError) {
      console.error('AgentsError:', e.message);
      throw e;
    } else {
      throw e;
    }
  }
}
