import { z } from 'zod';
import readline from 'node:readline/promises';
import { createWriteStream, mkdirSync } from 'node:fs';
import { Agent, Runner, tool, OpenAIProvider } from '@openai/agents';
import { RunConfig } from '@openai/agents-core';
import { MultiProvider, MultiProviderMap } from './multi_provider';
import { OllamaModelProvider } from './ollama_model_provider';


// Prompt user for yes/no confirmation
async function confirm(question: string): Promise<boolean> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  const answer = await rl.question(`${question} (y/n): `);
  rl.close();
  return ['y', 'yes'].includes(answer.trim().toLowerCase());
}

async function main(modelId: string, openaiUseResponses: boolean = false) {
  // Create and return the stream without consuming it
  const getWeatherTool = tool({
    name: 'get_weather',
    description: 'Get the weather for a given city',
    parameters: z.object({ city: z.string() }),
    needsApproval: async (_ctx, { city }) => city.includes('Oakland'),
    async execute({ city }) {
      return `The weather in ${city} is sunny.`;
    },
  });

  const weatherAgent = new Agent({
    model: modelId,
    name: 'Weather agent',
    instructions: 'You provide weather information.',
    handoffDescription: 'Handles weather-related queries',
    tools: [getWeatherTool],
  });

  // const getTemperatureTool = tool({
  //   name: 'get_temperature',
  //   description: 'Get the temperature for a given city',
  //   parameters: z.object({
  //     city: z.string(),
  //   }),
  //   needsApproval: async (_ctx, { city }) => city.includes('Oakland'),
  //   execute: async ({ city }) => {
  //     return `The temperature in ${city} is 20° Celsius`;
  //   },
  // });

  const mainAgent = new Agent({
    model: modelId,
    name: 'Main agent',
    instructions:
      'You are a general assistant. For weather questions, call the weather agent tool with a short input string and then answer.',
    handoffs: [weatherAgent],
    tools: [
      // getTemperatureTool,
      // weatherAgent.asTool({
      //   toolName: 'ask_weather_agent',
      //   toolDescription:
      //     'Ask the weather agent about locations by passing a short input.',
      //   // Require approval when the generated input mentions San Francisco.
      //   needsApproval: async (_ctx, { input }) =>
      //     input.includes('San Francisco'),
      // }),
    ],
  });

  const modelProviderMap = new MultiProviderMap();

  modelProviderMap.addProvider("ollama", new OllamaModelProvider({
    apiKey: process.env.OLLAMA_API_KEY,
  }));

  // Add Anthropic provider using OpenAI interface
  modelProviderMap.addProvider("anthropic", new OpenAIProvider({
    apiKey: process.env.ANTHROPIC_API_KEY,
    baseURL: "https://api.anthropic.com/v1/",
    useResponses: false
  }));

  // Use MultiProvider for model selection
  const modelProvider = new MultiProvider({
    provider_map: modelProviderMap,
    openai_use_responses: openaiUseResponses,
  });

  const runConfig: RunConfig = {
    modelProvider: modelProvider,
    // groupId: contextId,
    // traceId: trace.traceId, // Since we're associating the traceId with the run, then the history will be associated with the trace (task)
    traceIncludeSensitiveData: true,
    tracingDisabled: false,
  };

  const runner = new Runner(runConfig);

  const stream = await runner.run(
    mainAgent,
    'What is the weather and temperature in San Francisco and Oakland? Use available tools as needed.',
    { stream: true },
  );

  return stream;
}

async function runTestClient(modelId: string, openaiUseResponses: boolean = false) {
  // Get the stream
  const stream = await main(modelId, openaiUseResponses);

  // Create filename in data folder (without timestamp)
  const modelName = modelId.replace(':', '_'); // Replace colon with underscore for filename
  const modelNameForFile = modelName.replace('/', '_');

  // Only include openai_use_responses flag for OpenAI models (no slash in model name)
  const isOpenAIModel = !modelId.includes('/');
  const filename = isOpenAIModel
    ? `data/${modelNameForFile}.${openaiUseResponses}.jsonl`
    : `data/${modelNameForFile}.jsonl`;

  // Ensure data directory exists
  mkdirSync('data', { recursive: true });

  // Demonstrate usage by consuming the stream
  const fileStream = createWriteStream(filename, { flags: 'w' });

  for await (const chunk of stream) {
    // Write each chunk as a JSON line to the file
    fileStream.write(JSON.stringify(chunk) + '\n');

    // Handle different types of stream events with improved formatting
    if ('name' in chunk) {
      switch (chunk.name) {
        case 'handoff_occurred':
          const handoffOccurredItem = (chunk as any).item;
          if (handoffOccurredItem?.sourceAgent?.name && handoffOccurredItem?.targetAgent?.name) {
            const sourceAgent = handoffOccurredItem.sourceAgent.name;
            const targetAgent = handoffOccurredItem.targetAgent.name;
            console.log(`✅ Handoff completed: ${sourceAgent} → ${targetAgent}`);
          }
          break;

        case 'handoff_requested':
          const handoffRequestItem = (chunk as any).item;
          if (handoffRequestItem?.rawItem?.providerData?.function?.name && handoffRequestItem?.agent?.name) {
            const handoffFunction = handoffRequestItem.rawItem.providerData.function.name;
            const handoffArgs = handoffRequestItem.rawItem.providerData.function.arguments;
            const handoffAgent = handoffRequestItem.agent.name;
            console.log(`\n🔄 Handoff requested`);
            console.log(`   From: ${handoffAgent}`);
            console.log(`   Function: ${handoffFunction}`);
            console.log(`   Arguments: ${JSON.stringify(handoffArgs, null, 2)}`);
          }
          break;

        case 'tool_approval_requested':
          const toolApprovalItem = (chunk as any).item;
          if (toolApprovalItem?.rawItem?.providerData?.function?.name && toolApprovalItem?.agent?.name) {
            const toolName = toolApprovalItem.rawItem.providerData.function.name;
            const toolArguments = toolApprovalItem.rawItem.providerData.function.arguments;
            const toolAgent = toolApprovalItem.agent.name;
            console.log(`\n🔐 TOOL APPROVAL REQUIRED`);
            console.log(`   Agent: ${toolAgent}`);
            console.log(`   Tool: ${toolName}`);
            console.log(`   Arguments: ${JSON.stringify(toolArguments, null, 2)}`);
            console.log(`   Status: Waiting for approval...\n`);
          }
          break;

        case 'tool_called':
          const toolCallItem = (chunk as any).item;
          if (toolCallItem?.rawItem?.providerData?.function?.name && toolCallItem?.agent?.name) {
            const calledToolName = toolCallItem.rawItem.providerData.function.name;
            const calledToolArgs = toolCallItem.rawItem.providerData.function.arguments;
            const calledToolAgent = toolCallItem.agent.name;
            console.log(`🔧 Tool called: ${calledToolName}`);
            console.log(`   Agent: ${calledToolAgent}`);
            console.log(`   Arguments: ${JSON.stringify(calledToolArgs, null, 2)}`);
          }
          break;

        case 'tool_output':
          const toolOutputItem = (chunk as any).item;
          if (toolOutputItem?.rawItem?.name && toolOutputItem?.agent?.name) {
            const outputToolName = toolOutputItem.rawItem.name;
            const toolResult = toolOutputItem.output;
            const outputAgent = toolOutputItem.agent.name;
            console.log(`✅ Tool output from ${outputToolName}:`);
            console.log(`   Agent: ${outputAgent}`);
            console.log(`   Result: ${toolResult}`);
          }
          break;

        default:
          // Uncomment the line below to see all events
          // console.log(`📝 Event: ${chunk.name}`);
          break;
      }
    }
  }

  // Close the file stream when done
  fileStream.end();

  await stream.completed;

  // while (stream.interruptions?.length) {
  //   console.log(
  //     'Human-in-the-loop: approval required for the following tool calls:',
  //   );
  //   const state = stream.state;
  //   for (const interruption of stream.interruptions) {
  //     const ok = await confirm(
  //       `Agent ${interruption.agent.name} would like to use the tool ${interruption.rawItem.name} with "${interruption.rawItem.arguments}". Do you approve?`,
  //     );
  //     if (ok) {
  //       state.approve(interruption);
  //     } else {
  //       state.reject(interruption);
  //     }
  //   }

  //   // Resume execution with streaming output
  //   stream = await runner.run(mainAgent, state, { stream: true });
  //   const textStream = stream.toTextStream({ compatibleWithNodeStreams: true });
  //   textStream.pipe(process.stdout);
  //   await stream.completed;
  // }

  console.log('\n\nDone');
}


if (import.meta.url === `file://${process.argv[1]}`) {
  const modelId = process.env.MODEL_ID;
  const openaiUseResponses = (process.env.OPENAI_USE_RESPONSES ?? 'false').toLowerCase() === 'true';

  if (!modelId) {
    console.error('Missing required env var MODEL_ID');
    console.error('Usage: MODEL_ID="gpt-5|anthropic/claude-sonnet-4-5|ollama/smollm2:1.7b|ollama/gpt-oss:120b-cloud" pnpm run start');
    process.exit(1);
  }

  runTestClient(modelId, openaiUseResponses).catch(console.error);
}
