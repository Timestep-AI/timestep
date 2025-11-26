export { OllamaModel } from './ollama_model.ts';
export { OllamaModelProvider, type OllamaModelProviderOptions } from './ollama_model_provider.ts';
export { MultiModelProvider, MultiModelProviderMap } from './multi_model_provider.ts';

import { Agent, Runner, Session } from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';

export async function runAgent(agent: Agent, runInput: AgentInputItem[], session: Session, stream: boolean): Promise<void> {
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
    const textStream = result.toTextStream({
      compatibleWithNodeStreams: true,
    });
    for await (const chunk of textStream) {
      // Consume stream chunks
    }
    await result.completed;
  } else {
    await runner.run(agent, runInput, { session, sessionInputCallback });
  }
}
