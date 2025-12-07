/** Helper functions and constants for agent tests (no test definitions). */

import {
  Agent,
  OpenAIConversationsSession,
  tool,
  InputGuardrail,
  OutputGuardrail,
  InputGuardrailTripwireTriggered,
  OutputGuardrailTripwireTriggered,
} from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';
import OpenAI from 'openai';
import { runAgent, RunStateStore } from '../timestep/index';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function imageToBase64(imagePath: string): string {
  const imageBuffer = fs.readFileSync(imagePath);
  return imageBuffer.toString('base64');
}

export const RECOMMENDED_PROMPT_PREFIX = "# System context\nYou are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.\n"

export const getWeather = tool({
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

// File paths for test media
const IMAGE_PATH = path.join(__dirname, '../../data', 'image_bison.jpg');

// Encode files once at module level
export const IMAGE_BASE64 = imageToBase64(IMAGE_PATH);

export const RUN_INPUTS: AgentInputItem[][] = [
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
  ],
  [
    { type: "message", role: "user", content: [{ type: "input_text", text: "What's the weather on The Dark Side of the Moon?" }] }
  ],
  [
    { type: "message", role: "user", content: [{ type: "input_text", text: "What's four times the last number we calculated minus one?" }] }
  ],
  [
    { type: "message", role: "user", content: [{ type: "input_text", text: "What's four times the last number we calculated minus six?" }] }
  ],
  [
    {
      type: "message",
      role: "user",
      content: [
        {
          type: "input_image",
          image: `data:image/jpeg;base64,${IMAGE_BASE64}`,
          detail: 'auto',
        },
        {
          type: "input_text",
          text: "What do you see in this image?"
        }
      ],
    },
  ]
];

export function cleanItems(items: any[]): any[] {
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

export async function runAgentTestPartial(runInParallel: boolean = true, stream: boolean = false, sessionId?: string, startIndex: number = 0, endIndex?: number): Promise<{ sessionId: string; connectionString?: string }> {
  if (endIndex === undefined) {
    endIndex = RUN_INPUTS.length;
  }

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

  for (let i = startIndex; i < endIndex!; i++) {
    let runInput: AgentInputItem[] | any = RUN_INPUTS[i];

    try {
      let result = await runAgent(personalAssistantAgent, runInput, session, stream);

      // Handle interruptions - save state but don't approve
      if (result.interruptions?.length) {
        // Save state (this will initialize the database connection if needed)
        const state = result.state;
        await stateStore.save(state);
        
        // Get the database connection string that was used
        // This ensures Python can connect to the same database
        const { getDBOSConnectionString } = await import('../timestep/dbos_config.ts');
        const connectionString = getDBOSConnectionString();
        
        // Return session ID and connection string without approving
        return { sessionId: currentSessionId, connectionString };
      }
    } catch (e) {
      if (e instanceof InputGuardrailTripwireTriggered || e instanceof OutputGuardrailTripwireTriggered) {
        // Guardrail was triggered - pop items until we've removed the user message
        const recentItems = await session.getItems(2);
        let itemsToPop = 0;
        let foundUserMessage = false;
        for (let i = recentItems.length - 1; i >= 0; i--) {
          itemsToPop++;
          const item = recentItems[i];
          if (typeof item === 'object' && 'role' in item && item.role === 'user') {
            foundUserMessage = true;
            break;
          }
        }

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

  // If we got here without interruption, get connection string anyway (may be undefined if DBOS wasn't configured)
  const { getDBOSConnectionString } = await import('../timestep/dbos_config.ts');
  const connectionString = getDBOSConnectionString();
  return { sessionId: currentSessionId, connectionString };
}

export async function runAgentTestFromPython(runInParallel: boolean = true, stream: boolean = false, sessionId: string, connectionString?: string): Promise<any[]> {
  // Configure DBOS before loading state (only need configuration, not launch, since we're just using the database connection)
  // Use provided connectionString if available (for TS->TS tests), otherwise use PG_CONNECTION_URI from env (for Python->TS tests)
  const { configureDBOS } = await import('../timestep/dbos_config.ts');
  const configOptions: any = { name: 'timestep-test' };
  if (connectionString) {
    configOptions.systemDatabaseUrl = connectionString;
  } else {
    // Log for debugging - check if env var is set
    const envConnStr = process.env['PG_CONNECTION_URI'];
    if (envConnStr) {
      console.log(`Using PG_CONNECTION_URI from environment: ${envConnStr.substring(0, 50)}...`);
    }
  }
  console.log('Configuring DBOS...');
  await configureDBOS(configOptions);
  console.log('DBOS configured');
  // Note: We don't need to launch DBOS here - RunStateStore only needs the connection string, not the workflow system
  
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
    inputGuardrails: [moonGuardrail],
    outputGuardrails: [no47Guardrail],
  });

  // Use the same session ID
  const session = new OpenAIConversationsSession({ conversationId: sessionId });

  const stateStore = new RunStateStore({ agent: personalAssistantAgent, sessionId: sessionId });

  // Load state saved by Python
  const loadedState = await stateStore.load();
  const interruptions = loadedState.getInterruptions();
  for (const interruption of interruptions) {
    loadedState.approve(interruption);
  }

  // Resume with state
  let result = await runAgent(personalAssistantAgent, loadedState, session, stream);

  // Continue with remaining inputs (indices 4-7)
  for (let i = 4; i < RUN_INPUTS.length; i++) {
    let runInput: AgentInputItem[] | any = RUN_INPUTS[i];

    try {
      result = await runAgent(personalAssistantAgent, runInput, session, stream);

      // Handle any new interruptions
      if (result.interruptions?.length) {
        const state = result.state;
        await stateStore.save(state);
        const loadedState = await stateStore.load();
        const interruptions = loadedState.getInterruptions();
        for (const interruption of interruptions) {
          loadedState.approve(interruption);
        }
        result = await runAgent(personalAssistantAgent, loadedState, session, stream);
      }
    } catch (e) {
      if (e instanceof InputGuardrailTripwireTriggered || e instanceof OutputGuardrailTripwireTriggered) {
        const recentItems = await session.getItems(2);
        let itemsToPop = 0;
        let foundUserMessage = false;
        for (let i = recentItems.length - 1; i >= 0; i--) {
          itemsToPop++;
          const item = recentItems[i];
          if (typeof item === 'object' && 'role' in item && item.role === 'user') {
            foundUserMessage = true;
            break;
          }
        }

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

export const EXPECTED_ITEMS = [
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
  },
  {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text: "What's four times the last number we calculated minus six?" }]
  },
  {
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text: "42" }]
  },
  {
    type: "message",
    role: "user",
    content: [
      {
        type: "input_image",
        image: `data:image/jpeg;base64,${IMAGE_BASE64}`,
        detail: 'auto',
      },
      {
        type: "input_text",
        text: "What do you see in this image?"
      }
    ]
  },
  {
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text: "bison" }]
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

function normalizeText(text: string): string {
  // Convert to lowercase
  let normalized = text.toLowerCase();
  
  // Normalize mathematical operators
  normalized = normalized.replace(/\bequals\b/g, "=");
  normalized = normalized.replace(/\bis equal to\b/g, "=");
  normalized = normalized.replace(/\bequal to\b/g, "=");
  
  // Normalize whitespace (multiple spaces to single space)
  normalized = normalized.replace(/\s+/g, " ");
  
  // Trim and remove trailing punctuation variations (keep the content but normalize)
  normalized = normalized.trim();
  
  return normalized;
}

export function assertConversationItems(cleaned: any[], expected: any[]): void {
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
      // Normalize both texts before comparison
      const actualNormalized = normalizeText(actualText);
      const expectedNormalized = normalizeText(expectedText);
      // Check that either normalized text contains the other (for flexibility with LLM variability)
      if (!(expectedNormalized.includes(actualNormalized) || actualNormalized.includes(expectedNormalized))) {
        throw new Error(`Item ${i} text mismatch: expected '${expectedText}' and actual '${actualText}' do not contain each other`);
      }
      // Also check structure matches
      if (cleanedItem.type !== expectedItem.type || cleanedItem.role !== expectedItem.role) {
        throw new Error(`Item ${i} structure mismatch: type or role doesn't match`);
      }
    } else if (cleanedItem.type === "function_call" && expectedItem.type === "function_call") {
      // For function_call items, compare arguments as JSON objects (not strings)
      if (cleanedItem.type !== expectedItem.type) {
        throw new Error(`Item ${i} type mismatch: ${cleanedItem.type} !== ${expectedItem.type}`);
      }
      // Function names may differ in casing between languages (e.g., transfer_to_Weather_Assistant vs transfer_to_weather_assistant)
      if (cleanedItem.name.toLowerCase() !== expectedItem.name.toLowerCase()) {
        throw new Error(`Item ${i} name mismatch: ${cleanedItem.name} !== ${expectedItem.name}`);
      }
      // Parse and compare JSON arguments
      const actualArgs = JSON.parse(cleanedItem.arguments);
      const expectedArgs = JSON.parse(expectedItem.arguments);
      if (JSON.stringify(actualArgs) !== JSON.stringify(expectedArgs)) {
        throw new Error(`Item ${i} arguments mismatch: ${JSON.stringify(actualArgs)} !== ${JSON.stringify(expectedArgs)}`);
      }
    } else if (cleanedItem.type === "message" &&
               cleanedItem.role === "user" &&
               expectedItem.type === "message" &&
               expectedItem.role === "user") {
      // For user messages, check structure but handle images specially
      if (cleanedItem.type !== expectedItem.type) {
        throw new Error(`Item ${i} type mismatch`);
      }
      if (cleanedItem.role !== expectedItem.role) {
        throw new Error(`Item ${i} role mismatch`);
      }
      if (cleanedItem.content.length !== expectedItem.content.length) {
        throw new Error(`Item ${i} content length mismatch`);
      }

      // Check each content block
      for (let j = 0; j < cleanedItem.content.length; j++) {
        const actualBlock = cleanedItem.content[j];
        const expectedBlock = expectedItem.content[j];

        if (actualBlock.type !== expectedBlock.type) {
          throw new Error(`Item ${i} content block ${j} type mismatch`);
        }

        if (actualBlock.type === "input_image") {
          // For images, only check type and detail, not image/file_id (API converts data URIs to file_ids)
          if ("detail" in expectedBlock && actualBlock.detail !== expectedBlock.detail) {
            throw new Error(`Item ${i} content block ${j} detail mismatch`);
          }
        } else if (actualBlock.type === "input_text") {
          if (actualBlock.text !== expectedBlock.text) {
            throw new Error(`Item ${i} content block ${j} text mismatch`);
          }
        } else {
          assertEqual(actualBlock, expectedBlock, `Item ${i} content block ${j} mismatch`);
        }
      }
    } else if (cleanedItem.type === "function_call_output" && expectedItem.type === "function_call_output") {
      // For function_call_output items, parse JSON output if present
      if (cleanedItem.type !== expectedItem.type) {
        throw new Error(`Item ${i} type mismatch: ${cleanedItem.type} !== ${expectedItem.type}`);
      }
      if (cleanedItem.output && expectedItem.output) {
        try {
          const actualOutput = JSON.parse(cleanedItem.output);
          const expectedOutput = JSON.parse(expectedItem.output);
          if (JSON.stringify(actualOutput) !== JSON.stringify(expectedOutput)) {
            throw new Error(`Item ${i} output mismatch: ${JSON.stringify(actualOutput)} !== ${JSON.stringify(expectedOutput)}`);
          }
        } catch (e) {
          // If not JSON, compare as strings
          if (cleanedItem.output !== expectedItem.output) {
            throw new Error(`Item ${i} output mismatch: ${cleanedItem.output} !== ${expectedItem.output}`);
          }
        }
      } else {
        assertEqual(cleanedItem, expectedItem, `Item ${i} mismatch`);
      }
    } else {
      // For all other items, exact match
      assertEqual(cleanedItem, expectedItem, `Item ${i} mismatch`);
    }
  }
}

