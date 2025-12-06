export { OllamaModel } from './ollama_model.ts';
export { OllamaModelProvider, type OllamaModelProviderOptions } from './ollama_model_provider.ts';
export { MultiModelProvider, MultiModelProviderMap } from './multi_model_provider.ts';
export { webSearch } from './tools.ts';
export { RunStateStore } from './run_state_store.ts';

import { Agent, Runner, Session, RunState, MaxTurnsExceededError, ModelBehaviorError, UserError, AgentsError } from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';

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
