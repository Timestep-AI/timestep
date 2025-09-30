import { z } from 'zod';
import readline from 'node:readline/promises';
import { createWriteStream, mkdirSync, readFileSync, writeFileSync, existsSync, unlinkSync } from 'node:fs';
import { Agent, Runner, tool, OpenAIProvider, AgentInputItem, StreamRunOptions } from '@openai/agents';
import { RunConfig, RunState, RunToolApprovalItem } from '@openai/agents-core';
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

type AgentInput = string | RunState<undefined, Agent<unknown, "text">> | AgentInputItem[];

async function main(agent: Agent, agentInput: AgentInput, modelId: string, openaiUseResponses: boolean = false) {
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
    agent,
    agentInput,
    { stream: true },
  );

  return stream;
}

async function runTestClient(
  userInput: string,
  modelId: string,
  openaiUseResponses: boolean = false,
  approvalOptions: { alwaysApprove?: boolean; alwaysReject?: boolean } = {}
) {
  const { alwaysApprove = false, alwaysReject = false } = approvalOptions;
  const getWeatherTool = tool({
    name: 'get_weather',
    description: 'Get the weather for a given city',
    parameters: z.object({ city: z.string() }),
    needsApproval: true,
    // needsApproval: async (_ctx, { city }) => city.includes('Oakland'),
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

  const agent = new Agent({
    model: modelId,
    name: 'Main agent',
    instructions:
      'You are a general assistant. For weather questions, call the weather agent tool with a short input string and then answer.',
    handoffs: [weatherAgent],
    tools: [],
  });

  let agentInput: AgentInput = userInput;
  let isResumingFromSavedState = false;

  if (userInput === "approve" || userInput === "reject") {
    const isApproving = userInput === "approve";
    console.log(`[${userInput}] Loading saved state...`);
    const modelNameForFile = modelId.replace(':', '_').replace('/', '_');
    const stateFilename = `data/${modelNameForFile}.state.json`;
    const raw = readFileSync(stateFilename, 'utf8');
    const savedState = JSON.parse(raw);
    const runState = await RunState.fromString(
      agent,
      JSON.stringify(savedState),
    );
    console.log(`[${userInput}] Loaded state from file.`);

    // Find the tool approval item from generated items
    const approvalItem = runState._generatedItems
      .filter(item => item.type === 'tool_approval_item')
      .pop() as RunToolApprovalItem | undefined;

    if (!approvalItem) {
      throw new Error(`[${userInput}] No tool approval item found in state.`);
    }

    if (isApproving) {
      console.log('[approve] Approving tool call...');
      runState.approve(approvalItem, { alwaysApprove });
    } else {
      console.log('[reject] Rejecting tool call...');
      runState.reject(approvalItem, { alwaysReject });
    }

    agentInput = runState as RunState<undefined, Agent<unknown, "text">>;
    isResumingFromSavedState = true;
  }

  // Get the stream
  const stream = await main(agent, agentInput, modelId, openaiUseResponses);

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
  // Use append mode ('a') if resuming from saved state, otherwise overwrite ('w')
  const fileStream = createWriteStream(filename, { flags: isResumingFromSavedState ? 'a' : 'w' });
  let stateSavedForApproval = false;
  let shouldExitAfterSave = false;

  for await (const chunk of stream) {
    // console.log('chunk', chunk);

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

            if (!stateSavedForApproval) {
              // Mark that we need to exit after this approval
              stateSavedForApproval = true;
              shouldExitAfterSave = true;
            }
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
    if (shouldExitAfterSave) {
      // Save the state before exiting
      const state = (stream as any).state as RunState<any, any>;
      const stateFilename = `data/${modelNameForFile}.state.json`;
      try {
        const savedState = JSON.parse(JSON.stringify(state));
        writeFileSync(stateFilename, JSON.stringify(savedState, null, 2));
        console.log(`📝 Saved stream state to ${stateFilename}`);
      } catch (err) {
        console.error('Failed to serialize stream state:', err);
      }
      console.log('⏸️ Exiting run after saving approval state. Re-run with approval to continue.');
      break;
    }
  }

  // Close the file stream when done
  fileStream.end();

  await stream.completed;

  // If we completed without needing to exit for approval, delete any existing state file
  const stateFilename = `data/${modelNameForFile}.state.json`;
  if (!shouldExitAfterSave && existsSync(stateFilename)) {
    unlinkSync(stateFilename);
    console.log(`🗑️  Deleted state file ${stateFilename} (run completed successfully)`);
  }

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

  let userInput = "What is the weather and temperature in Oakland and San Francisco?";

  const modelName = modelId.replace(':', '_').replace('/', '_');
  const stateFilename = `data/${modelName}.state.json`;

  (async () => {
    // Main loop for handling chat and tool approvals
    while (true) {
      // Check if there's a saved state that needs approval
      if (existsSync(stateFilename)) {
        try {
          const raw = readFileSync(stateFilename, 'utf8');
          const savedState = JSON.parse(raw);

          // Find the tool approval item to show to user
          const generatedItems = savedState.generatedItems || [];
          const approvalItem = generatedItems
            .filter((item: any) => item.type === 'tool_approval_item')
            .pop();

          if (approvalItem) {
            const toolName = approvalItem.rawItem.name;
            const toolArgs = approvalItem.rawItem.arguments;
            const agentName = approvalItem.agent.name;

            // Ask user for confirmation
            const ok = await confirm(
              `Agent ${agentName} would like to use the tool ${toolName} with arguments ${toolArgs}. Do you approve?`
            );

            const decision = ok ? "approve" : "reject";

            // Ask if they want to always approve/reject this tool
            const always = await confirm(
              `Do you want to always ${decision} this tool for the rest of the run?`
            );

            const approvalOptions = ok
              ? { alwaysApprove: always }
              : { alwaysReject: always };

            await runTestClient(decision, modelId, openaiUseResponses, approvalOptions);
            continue; // Check again for more approvals
          }
        } catch (err) {
          console.error('Failed to load+rehydrate saved state for resume:', err);
        }
      }

      // No pending approvals, run with user input
      await runTestClient(userInput, modelId, openaiUseResponses);

      // Check if there's a saved state after the run
      if (!existsSync(stateFilename)) {
        // No saved state means the run completed without needing approval
        break;
      }
    }

    console.log('\n\nConversation completed.');
  })().catch(console.error);
}
