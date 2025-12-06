export { OllamaModel } from './ollama_model.ts';
export { OllamaModelProvider, type OllamaModelProviderOptions } from './ollama_model_provider.ts';
export { MultiModelProvider, MultiModelProviderMap } from './multi_model_provider.ts';
export { webSearch } from './tools.ts';
export { RunStateStore } from './run_state_store.ts';
export { runAgentWorkflow, queueAgentWorkflow, createScheduledAgentWorkflow } from './workflows.ts';
export { configureDBOS, ensureDBOSLaunched } from './dbos_config.ts';

import { Agent, Runner, Session, RunState, MaxTurnsExceededError, ModelBehaviorError, UserError, AgentsError } from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';

export async function defaultResultProcessor(result: any): Promise<any> {
  /**
   * Default result processor that consumes all events from a result (streaming or non-streaming).
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
  stream: boolean,
  resultProcessor?: (result: any) => Promise<any>
): Promise<any> {
  /**
   * Run an agent with the given session and stream setting.
   * 
   * @param agent - The agent to run
   * @param runInput - Input items or RunState for the agent
   * @param session - Session for managing conversation state
   * @param stream - Whether to stream the results
   * @param resultProcessor - Optional function to process the result. Defaults to defaultResultProcessor
   *   which consumes all streaming events and waits for completion. Pass undefined to skip processing.
   * @returns The processed result from the agent run
   */
  const processor = resultProcessor ?? defaultResultProcessor;
  const runner = new Runner();

  const sessionInputCallback = async (existingItems: AgentInputItem[], newInput: AgentInputItem[]): Promise<AgentInputItem[]> => {
    return [...existingItems, ...newInput];
  };

  try {
    let result;
    if (stream) {
      result = await runner.run(agent, runInput, {
        session,
        sessionInputCallback,
        stream: true
      });
    } else {
      result = await runner.run(agent, runInput, {
        session,
        sessionInputCallback
      });
    }

    // Apply result processor
    result = await processor(result);
    return result;
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
