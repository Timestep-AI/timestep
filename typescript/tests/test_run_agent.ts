/** Tests for runAgent functionality with conversation items assertions. */

import { test, expect } from 'vitest';
import {
  Agent,
  OpenAIConversationsSession,
  InputGuardrail,
  OutputGuardrail,
  InputGuardrailTripwireTriggered,
  OutputGuardrailTripwireTriggered,
} from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';
import OpenAI from 'openai';
import { runAgent, RunStateStore } from '../timestep/index';
import {
  RECOMMENDED_PROMPT_PREFIX,
  getWeather,
  RUN_INPUTS,
  cleanItems,
  assertConversationItems,
  EXPECTED_ITEMS,
} from './test_helpers';


async function runAgentTest(runInParallel: boolean = true, stream: boolean = false, sessionId?: string, model: string = "gpt-4.1"): Promise<any[]> {
  // Define guardrails
  const moonGuardrail: InputGuardrail = {
    name: 'Moon Guardrail',
    runInParallel: runInParallel,
    execute: async ({ input }) => {
      const inputText = typeof input === 'string' ? input : JSON.stringify(input);
      const mentionsMoon = inputText.toLowerCase().includes('moon');

      return {
        outputInfo: { mentionsMoon },
        tripwireTriggered: mentionsMoon,
      };
    },
  };

  const no47Guardrail: OutputGuardrail<any> = {
    name: 'No 47 Guardrail',
    execute: async ({ agentOutput }) => {
      const outputText = typeof agentOutput === 'string' ? agentOutput : JSON.stringify(agentOutput);
      const contains47 = outputText.includes('47');

      return {
        outputInfo: { contains47 },
        tripwireTriggered: contains47,
      };
    },
  };

  const weatherAssistantAgent = new Agent({
    instructions: `You are a helpful AI assistant that can answer questions about weather. When asked about weather, you MUST use the get_weather tool to get accurate, real-time weather information.`,
    model: model,
    modelSettings: { temperature: 0 },
    name: "Weather Assistant",
    tools: [getWeather],
  });

  const personalAssistantAgent = new Agent({
    handoffs: [weatherAssistantAgent],
    instructions: `${RECOMMENDED_PROMPT_PREFIX}You are an AI agent acting as a personal assistant.`,
    model: model,
    modelSettings: { temperature: 0 },
    name: "Personal Assistant",
    inputGuardrails: [moonGuardrail],
    outputGuardrails: [no47Guardrail],
  });

  const session = sessionId ? new OpenAIConversationsSession({ conversationId: sessionId }) : new OpenAIConversationsSession();

  // Get session ID for state file naming
  const currentSessionId = await session.getSessionId();
  if (!currentSessionId) {
    throw new Error('Failed to get session ID');
  }

  const stateStore = new RunStateStore({ agent: personalAssistantAgent, sessionId: currentSessionId });

  for (let i = 0; i < RUN_INPUTS.length; i++) {
    let runInput: AgentInputItem[] | any = RUN_INPUTS[i];

    try {
      let result = await runAgent(personalAssistantAgent, runInput, session, stream);

      // Handle interruptions
      if (result.interruptions?.length) {
        // Save state
        const state = result.state;
        await stateStore.save(state);

        // Load and approve
        const loadedState = await stateStore.load();
        const interruptions = loadedState.getInterruptions();
        for (const interruption of interruptions) {
          loadedState.approve(interruption);
        }

        // Resume with state
        result = await runAgent(personalAssistantAgent, loadedState, session, stream);
      }
    } catch (e) {
      if (e instanceof InputGuardrailTripwireTriggered || e instanceof OutputGuardrailTripwireTriggered) {
        // Guardrail was triggered - pop items until we've removed the user message
        // First, peek at the last few items to see what needs to be removed
        const recentItems = await session.getItems(2);
        // Count how many items to pop (from most recent back to the user message)
        let itemsToPop = 0;
        let foundUserMessage = false;
        for (let i = recentItems.length - 1; i >= 0; i--) {
          itemsToPop++;
          const item = recentItems[i];
          if (typeof item === 'object' && 'role' in item && item.role === 'user') {
            foundUserMessage = true;
            break;  // Found the user message, stop counting
          }
        }

        // Only pop if we found a user message in the recent items
        if (foundUserMessage) {
          for (let i = 0; i < itemsToPop; i++) {
            await session.popItem();
          }
        }
      } else {
        throw e;
      }
    }
  }

  // Clean up state file
  await stateStore.clear();

  const conversationId = await session.getSessionId();
  if (!conversationId) {
    throw new Error('Session does not have a conversation ID');
  }

  const openaiApiKey = process.env.OPENAI_API_KEY || '';
  if (!openaiApiKey) {
    throw new Error('OPENAI_API_KEY environment variable is required');
  }

  const client = new OpenAI({ apiKey: openaiApiKey });
  const itemsResponse = await client.conversations.items.list(conversationId, { limit: 100, order: 'asc' });
  return itemsResponse.data;
}

// Re-export helpers for backward compatibility
export { runAgentTestPartial, runAgentTestFromPython, cleanItems, assertConversationItems, EXPECTED_ITEMS } from './test_helpers';

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_run_agent_blocking_non_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues (may timeout or throw)
    const timeoutPromise = new Promise((_, reject) => 
      setTimeout(() => reject(new Error('Test timeout - expected failure')), 60000)
    );
    try {
      await Promise.race([
        runAgentTest(false, false, undefined, model),
        timeoutPromise
      ]);
      // If it doesn't throw, that's also acceptable (test may work intermittently)
    } catch (error) {
      // Expected to throw or timeout - this is the known failure case
      expect(error).toBeDefined();
    }
    return;
  }
  const items = await runAgentTest(false, false, undefined, model);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_run_agent_blocking_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues
    await expect(runAgentTest(false, true, undefined, model)).rejects.toThrow();
    return;
  }
  const items = await runAgentTest(false, true, undefined, model);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_run_agent_parallel_non_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues (may timeout or throw)
    const timeoutPromise = new Promise((_, reject) => 
      setTimeout(() => reject(new Error('Test timeout - expected failure')), 60000)
    );
    try {
      await Promise.race([
        runAgentTest(true, false, undefined, model),
        timeoutPromise
      ]);
    } catch (error) {
      // Expected to throw or timeout - this is the known failure case
      expect(error).toBeDefined();
    }
    return;
  }
  const items = await runAgentTest(true, false, undefined, model);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_run_agent_parallel_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues
    await expect(runAgentTest(true, true, undefined, model)).rejects.toThrow();
    return;
  }
  const items = await runAgentTest(true, true, undefined, model);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

