import { z } from 'zod';
import readline from 'node:readline/promises';
import { createWriteStream, mkdirSync, readFileSync, writeFileSync, existsSync, unlinkSync } from 'node:fs';
import { Agent, run, hostedMcpTool, RunToolApprovalItem, Runner, tool, OpenAIProvider, AgentInputItem } from '@openai/agents';
import { RunConfig, RunState } from '@openai/agents-core';
import { MultiProvider, MultiProviderMap } from './multi_provider';
import { OllamaModelProvider } from './ollama_model_provider';
import { fetchMcpTools } from './mcp_server_proxy';


// Prompt user for yes/no confirmation
async function confirm(rl: readline.Interface, question: string): Promise<boolean> {
  const answer = await rl.question(`${question} (y/n): `);
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

  // Fetch built-in tools for weather agent (only log on first run)
  const isFirstRun = userInput !== "approve" && userInput !== "reject";
  if (isFirstRun) {
    console.log('[MCP] Loading tools...');
  }

  // Configure approval policies
  const requireApproval = {
    never: { toolNames: ['search_codex_code', 'fetch_codex_documentation'] },
    always: { toolNames: ['get_weather', 'fetch_generic_url_content'] },
  };

  const weatherTools = await fetchMcpTools(null, true, requireApproval);

  // Create weather agent with built-in tools
  const weatherAgent = new Agent({
    model: modelId,
    name: 'Weather agent',
    instructions: 'You provide weather information.',
    handoffDescription: 'Handles weather-related queries',
    tools: weatherTools,
  });

  // Fetch remote MCP tools from the codex server
  const mcpTools = await fetchMcpTools('https://gitmcp.io/timestep-ai/timestep', false, requireApproval);

  if (isFirstRun) {
    console.log(`[MCP] Loaded ${weatherTools.length + mcpTools.length} tools\n`);
  }

  // Create main agent with remote MCP tools and weather handoff
  const agent = new Agent({
    model: modelId,
    name: 'Main Assistant',
    instructions:
      'You are a helpful assistant. For questions about the timestep-ai/timestep repository, use the MCP tools. For weather questions, hand off to the weather agent.',
    tools: mcpTools,
    handoffs: [weatherAgent],
  });

  let agentInput: AgentInput = userInput;
  let isResumingFromSavedState = false;

  if (userInput === "approve" || userInput === "reject") {
    const isApproving = userInput === "approve";
    const modelNameForFile = modelId.replace(':', '_').replace('/', '_');
    const stateFilename = `data/${modelNameForFile}.state.json`;
    const raw = readFileSync(stateFilename, 'utf8');
    const savedState = JSON.parse(raw);
    const runState = await RunState.fromString(
      agent,
      JSON.stringify(savedState),
    );

    // Find the tool approval item from generated items
    const approvalItem = runState._generatedItems
      .filter(item => item.type === 'tool_approval_item')
      .pop() as RunToolApprovalItem | undefined;

    if (!approvalItem) {
      throw new Error(`No tool approval item found in state.`);
    }

    if (isApproving) {
      runState.approve(approvalItem, { alwaysApprove });
    } else {
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

        case 'message_output_created':
          const messageItem = (chunk as any).item;
          if (messageItem?.rawItem?.content) {
            const content = messageItem.rawItem.content;
            // Handle array of content blocks
            if (Array.isArray(content)) {
              for (const part of content) {
                if ((part.type === 'text' || part.type === 'output_text') && part.text) {
                  process.stdout.write(part.text);
                }
              }
            }
            // Handle string content
            else if (typeof content === 'string') {
              process.stdout.write(content);
            }
          }
          break;

        default:
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
        // State saved silently
      } catch (err) {
        console.error('Failed to serialize stream state:', err);
      }
      break;
    }
  }

  // Close the file stream when done
  fileStream.end();

  // Only wait for stream completion if we didn't exit early for approval
  if (!shouldExitAfterSave) {
    await stream.completed;

    // Add newline after response
    console.log();

    // If we completed without needing to exit for approval, delete any existing state file
    const stateFilename = `data/${modelNameForFile}.state.json`;
    if (existsSync(stateFilename)) {
      unlinkSync(stateFilename);
    }
  }
}


if (import.meta.url === `file://${process.argv[1]}`) {
  const modelId = process.env.MODEL_ID;
  const openaiUseResponses = (process.env.OPENAI_USE_RESPONSES ?? 'false').toLowerCase() === 'true';

  if (!modelId) {
    console.error('Missing required env var MODEL_ID');
    console.error('Usage: MODEL_ID="gpt-5|anthropic/claude-sonnet-4-5|ollama/smollm2:1.7b|ollama/gpt-oss:120b-cloud" pnpm run start');
    process.exit(1);
  }

  const modelName = modelId.replace(':', '_').replace('/', '_');
  const stateFilename = `data/${modelName}.state.json`;

  (async () => {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    let currentInput: string | null = null;
    let justHandledApproval = false;

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
              rl,
              `Agent ${agentName} would like to use the tool ${toolName} with arguments ${toolArgs}. Do you approve?`
            );

            const decision = ok ? "approve" : "reject";

            // Ask if they want to always approve/reject this tool
            const always = await confirm(
              rl,
              `Do you want to always ${decision} this tool for the rest of the run?`
            );

            const approvalOptions = ok
              ? { alwaysApprove: always }
              : { alwaysReject: always };

            await runTestClient(decision, modelId, openaiUseResponses, approvalOptions);
            justHandledApproval = true;
            continue; // Check again for more approvals
          }
        } catch (err) {
          console.error('Failed to load+rehydrate saved state for resume:', err);
        }
      } else if (justHandledApproval) {
        // We just handled an approval and now there's no state file, meaning run completed
        justHandledApproval = false;

        const continueChat = await confirm(rl, 'Do you want to continue the conversation?');
        if (!continueChat) {
          break;
        }
        currentInput = await rl.question('You: ');
        if (!currentInput.trim()) {
          break;
        }
        continue; // Start next iteration with new input
      } else if (!currentInput) {
        // First time or no input yet, prompt for input
        currentInput = await rl.question('You: ');
        if (!currentInput.trim()) {
          break;
        }
      }

      // No pending approvals, run with user input
      try {
        await runTestClient(currentInput, modelId, openaiUseResponses);
      } catch (error) {
        console.error('\n❌ Error during run:', error.message);
      }

      // Check if there's a saved state after the run
      if (existsSync(stateFilename)) {
        // A saved state was created, which means approval is needed
        // Loop will continue to the top to handle the approval
        continue;
      }

      // No saved state means the run completed without needing approval
      const continueChat = await confirm(rl, 'Do you want to continue the conversation?');
      if (!continueChat) {
        break;
      }
      currentInput = await rl.question('You: ');
      if (!currentInput.trim()) {
        break;
      }
      // Loop will continue with new input
    }

    rl.close();
    console.log('\n\nConversation completed.');
  })().catch(console.error);
}
