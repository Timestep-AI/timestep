/** Tests for runAgent functionality with conversation items assertions. */

import { Agent, OpenAIConversationsSession, Runner, tool, RunState } from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';
import OpenAI from 'openai';
import { runAgent, consumeResult, RunStateStore } from '../timestep/index';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const RECOMMENDED_PROMPT_PREFIX = "# System context\nYou are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.\n"

const getWeather = tool({
  name: 'get_weather',
  description: 'returns weather info for the specified city.',
  parameters: {
    type: 'object',
    properties: {
      city: { type: 'string' }
    },
    required: ['city'],
    additionalProperties: false
  } as any,
  needsApproval: async (_ctx: any, args: any) => {
    // Require approval for Berkeley
    return args.city?.includes('Berkeley') || false;
  },
  execute: async (args: any): Promise<string> => {
    return `The weather in ${args.city} is sunny`;
  }
});

const RUN_INPUTS: AgentInputItem[][] = [
  [
    { type: "message", role: "user", content: [{ type: "input_text", text: "What's 2+2?" }] }
  ],
  [
    { type: "message", role: "user", content: [{ type: "input_text", text: "What's the weather in Oakland?" }] }
  ],
  [
    { type: "message", role: "user", content: [{ type: "input_text", text: "What's three times that number you calculated earlier?" }] }
  ],
  [
    { type: "message", role: "user", content: [{ type: "input_text", text: "What's the weather in Berkeley?" }] }
  ]
];

function cleanItems(items: any[]): any[] {
  function removeId(obj: any): any {
    if (Array.isArray(obj)) {
      return obj.map(removeId);
    }
    if (obj && typeof obj === 'object') {
      const result: any = {};
      for (const [key, value] of Object.entries(obj)) {
        if (key !== 'id' && key !== 'status' && key !== 'call_id' && key !== 'annotations' && key !== 'logprobs') {
          result[key] = removeId(value);
        }
      }
      return result;
    }
    return obj;
  }
  
  return items.map(removeId);
}


async function runAgentTest(stream: boolean = false): Promise<any[]> {
  const weatherAssistantAgent = new Agent({
    instructions: `You are a helpful AI assistant that can answer questions about weather. When asked about weather, you MUST use the get_weather tool to get accurate, real-time weather information.`,
    model: "gpt-4.1",
    modelSettings: { temperature: 0 },
    name: "Weather Assistant",
    tools: [getWeather],
  });

  const personalAssistantAgent = new Agent({
    handoffs: [weatherAssistantAgent],
    instructions: `${RECOMMENDED_PROMPT_PREFIX}You are an AI agent acting as a personal assistant.`,
    model: "gpt-4.1",
    modelSettings: { temperature: 0 },
    name: "Personal Assistant",
  });

  const session = new OpenAIConversationsSession();

  // Get session ID for state file naming
  const sessionId = await session.getSessionId();
  if (!sessionId) {
    throw new Error('Failed to get session ID');
  }

  const stateFilePath = path.join(__dirname, '../../data', `agent_state_${sessionId}.json`);
  const stateStore = new RunStateStore(stateFilePath, personalAssistantAgent);

  for (let i = 0; i < RUN_INPUTS.length; i++) {
    let runInput: AgentInputItem[] | any = RUN_INPUTS[i];

    let result = await runAgent(personalAssistantAgent, runInput, session, stream);
    result = await consumeResult(result);

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
      result = await consumeResult(result);
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

const EXPECTED_ITEMS = [
  {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text: "What's 2+2?" }]
  },
  {
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text: "2 + 2 = 4." }]
  },
  {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text: "What's the weather in Oakland?" }]
  },
  {
    type: "function_call",
    name: "transfer_to_Weather_Assistant",
    arguments: "{}"
  },
  {
    type: "function_call_output",
    output: '{"assistant":"Weather Assistant"}'
  },
  {
    type: "function_call",
    name: "get_weather",
    arguments: '{"city":"Oakland"}'
  },
  {
    type: "function_call_output",
    output: "The weather in Oakland is sunny"
  },
  {
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text: "sunny" }]
  },
  {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text: "What's three times that number you calculated earlier?" }]
  },
  {
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text: "12" }]
  },
  {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text: "What's the weather in Berkeley?" }]
  },
  {
    type: "function_call",
    name: "transfer_to_Weather_Assistant",
    arguments: "{}"
  },
  {
    type: "function_call_output",
    output: '{"assistant":"Weather Assistant"}'
  },
  {
    type: "function_call",
    name: "get_weather",
    arguments: '{"city":"Berkeley"}'
  },
  {
    type: "function_call_output",
    output: "The weather in Berkeley is sunny"
  },
  {
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text: "sunny" }]
  }
];

function assertEqual(actual: any, expected: any, message: string): void {
  function sortKeys(obj: any): any {
    if (obj === null || typeof obj !== 'object') {
      return obj;
    }
    if (Array.isArray(obj)) {
      return obj.map(sortKeys);
    }
    const sorted: any = {};
    for (const key of Object.keys(obj).sort()) {
      sorted[key] = sortKeys(obj[key]);
    }
    return sorted;
  }
  
  const actualSorted = sortKeys(actual);
  const expectedSorted = sortKeys(expected);
  if (JSON.stringify(actualSorted) !== JSON.stringify(expectedSorted)) {
    throw new Error(`${message}\nExpected: ${JSON.stringify(expected, null, 2)}\nActual: ${JSON.stringify(actual, null, 2)}`);
  }
}

function assertConversationItems(cleaned: any[], expected: any[]): void {
  if (cleaned.length !== expected.length) {
    throw new Error(`Expected ${expected.length} items, got ${cleaned.length}`);
  }
  
  for (let i = 0; i < cleaned.length; i++) {
    const cleanedItem = cleaned[i];
    const expectedItem = expected[i];
    
    // For assistant messages with output_text, check that actual text contains expected text
    if (cleanedItem.type === "message" && 
        cleanedItem.role === "assistant" && 
        expectedItem.type === "message" && 
        expectedItem.role === "assistant") {
      // Extract text from both actual and expected
      const actualText = cleanedItem.content
        .filter((block: any) => block.type === "output_text")
        .map((block: any) => block.text || "")
        .join(" ");
      const expectedText = expectedItem.content
        .filter((block: any) => block.type === "output_text")
        .map((block: any) => block.text || "")
        .join(" ");
      // Check that actual contains expected (case-insensitive for flexibility)
      if (!actualText.toLowerCase().includes(expectedText.toLowerCase())) {
        throw new Error(`Item ${i} text mismatch: expected '${expectedText}' to be contained in '${actualText}'`);
      }
      // Also check structure matches
      if (cleanedItem.type !== expectedItem.type || cleanedItem.role !== expectedItem.role) {
        throw new Error(`Item ${i} structure mismatch: type or role doesn't match`);
      }
    } else {
      // For all other items, exact match
      assertEqual(cleanedItem, expectedItem, `Item ${i} mismatch`);
    }
  }
}

async function runTest(name: string, testFn: () => Promise<void>): Promise<void> {
  try {
    console.log(`Running test: ${name}`);
    await testFn();
    console.log(`✓ ${name} passed`);
  } catch (error: any) {
    console.error(`✗ ${name} failed:`, error.message);
    throw error;
  }
}

async function testRunAgentNonStreaming(): Promise<void> {
  const items = await runAgentTest(false);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
}

async function testRunAgentStreaming(): Promise<void> {
  const items = await runAgentTest(true);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
}

(async () => {
  try {
    await runTest('test_run_agent_non_streaming', testRunAgentNonStreaming);
    await runTest('test_run_agent_streaming', testRunAgentStreaming);
    console.log('\nAll tests passed!');
  } catch (error) {
    console.error('\nTests failed!');
    process.exit(1);
  }
})();

