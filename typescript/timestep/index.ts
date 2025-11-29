export { OllamaModel } from './ollama_model.ts';
export { OllamaModelProvider, type OllamaModelProviderOptions } from './ollama_model_provider.ts';
export { MultiModelProvider, MultiModelProviderMap } from './multi_model_provider.ts';
export { webSearch } from './tools.ts';

import { Agent, Runner, Session, RunState } from '@openai/agents';
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
}
