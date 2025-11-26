/** Tests for runAgent functionality with conversation items assertions. */

import { Agent, OpenAIConversationsSession, Runner, tool } from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';
import OpenAI from 'openai';
import { runAgent } from '../timestep/index';

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
  execute: async (args: any): Promise<string> => {
    return `The weather in ${args.city} is sunny`;
  }
});

function cleanItems(items: any[]): any[] {
  function removeId(obj: any): any {
    if (Array.isArray(obj)) {
      return obj.map(removeId);
    }
    if (obj && typeof obj === 'object') {
      const result: any = {};
      for (const [key, value] of Object.entries(obj)) {
        if (key !== 'id' && key !== 'status' && key !== 'call_id') {
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
    instructions: `${RECOMMENDED_PROMPT_PREFIX}You are a helpful AI assistant that can answer questions about weather. When asked about weather, you MUST use the get_weather tool to get accurate, real-time weather information.`,
    model: "gpt-4.1",
    modelSettings: { temperature: 0 },
    name: "Weather Assistant",
    tools: [getWeather],
  });

  const personalAssistantAgent = new Agent({
    handoffs: [weatherAssistantAgent],
    instructions: "# System context\n\nYou are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.\n\nYou are an AI agent acting as a personal assistant.",
    model: "gpt-4.1",
    modelSettings: { temperature: 0 },
    name: "Personal Assistant",
  });

  const session = new OpenAIConversationsSession();

  const runInput: AgentInputItem[] = [{ type: "message", role: "user", content: [{ type: "input_text", text: "What's the weather in Oakland?" }] }];
  await runAgent(personalAssistantAgent, runInput, session, stream);

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
    content: [{ type: "output_text", text: "" }]  // Text may vary
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
  
  assertEqual(cleaned[0], expected[0], "First item (user message) mismatch");
  assertEqual(cleaned[1], expected[1], "Second item (handoff call) mismatch");
  assertEqual(cleaned[2], expected[2], "Third item (handoff output) mismatch");
  assertEqual(cleaned[3], expected[3], "Fourth item (get_weather call) mismatch");
  assertEqual(cleaned[4], expected[4], "Fifth item (get_weather output) mismatch");
  
  // Last message content may vary, just check structure
  if (cleaned[5].type !== expected[5].type || cleaned[5].role !== expected[5].role || !cleaned[5].content || cleaned[5].content.length === 0) {
    throw new Error(`Last item structure mismatch: ${JSON.stringify(cleaned[5], null, 2)}`);
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

