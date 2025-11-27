export { OllamaModel } from './ollama_model.ts';
export { OllamaModelProvider, type OllamaModelProviderOptions } from './ollama_model_provider.ts';
export { MultiModelProvider, MultiModelProviderMap } from './multi_model_provider.ts';

import { Agent, Runner, Session } from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';

export type ApprovalCallback = (interruption: any) => Promise<boolean>;

export async function runAgent(
  agent: Agent,
  runInput: AgentInputItem[],
  session: Session,
  stream: boolean,
  approvalCallback?: ApprovalCallback
): Promise<void> {
  const runner = new Runner();
  
  const sessionInputCallback = async (existingItems: AgentInputItem[], newInput: AgentInputItem[]): Promise<AgentInputItem[]> => {
    return [...existingItems, ...newInput];
  };
  
  if (stream) {
    let result = await runner.run(agent, runInput, {
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
    
    // Handle interruptions
    while (result.interruptions?.length) {
      const state = result.state;
      for (const interruption of result.interruptions) {
        const approved = approvalCallback 
          ? await approvalCallback(interruption)
          : true; // Auto-approve if no callback provided
        if (approved) {
          state.approve(interruption);
        } else {
          state.reject(interruption);
        }
      }
      
      // Resume execution
      result = await runner.run(agent, state, {
        session,
        sessionInputCallback,
        stream: true
      });
      const resumeTextStream = result.toTextStream({
        compatibleWithNodeStreams: true,
      });
      for await (const chunk of resumeTextStream) {
        // Consume stream chunks
      }
      await result.completed;
    }
  } else {
    let result = await runner.run(agent, runInput, { session, sessionInputCallback });
    
    // Handle interruptions
    while (result.interruptions?.length) {
      const state = result.state;
      for (const interruption of result.interruptions) {
        const approved = approvalCallback 
          ? await approvalCallback(interruption)
          : true; // Auto-approve if no callback provided
        if (approved) {
          state.approve(interruption);
        } else {
          state.reject(interruption);
        }
      }
      
      // Resume execution
      result = await runner.run(agent, state, { session, sessionInputCallback });
    }
  }
}
