#!/usr/bin/env node
/**
 * Test runner for TypeScript implementation.
 * Accepts a test case JSON file and executes it.
 */

import { readFileSync } from 'fs';
import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '../timestep/index.ts';

let Agent: any;
let Runner: any;
let agentsAvailable = false;

// Load agents library dynamically
async function loadAgents() {
  if (agentsAvailable) return;
  
  try {
    // Try importing from @openai/agents
    // @ts-ignore - Dynamic import, may not be available
    const agentsModule = await import('@openai/agents');
    if (agentsModule.Agent && agentsModule.Runner) {
      Agent = agentsModule.Agent;
      Runner = agentsModule.Runner;
      agentsAvailable = true;
    } else {
      agentsAvailable = false;
    }
  } catch (e: any) {
    // Agents library not available - log for debugging
    console.error('Failed to load @openai/agents:', e.message);
    agentsAvailable = false;
  }
}

interface TestCase {
  name: string;
  setup: {
    provider_type: string;
    provider_config?: {
      openai_api_key?: string;
      openai_base_url?: string;
      openai_organization?: string;
      openai_project?: string;
      openai_use_responses?: boolean;
      ollama_api_key?: string;
      ollama_base_url?: string;
      [key: string]: any; // Allow other config options
    };
  };
  input: {
    model_name: string;
    messages?: Array<{ role: string; content: string }>;
    run_agent?: boolean;
    stream?: boolean;
    agent_config?: {
      system_prompt?: string;
      temperature?: number;
      max_tokens?: number;
      tools?: Array<any>;
    };
    user_input?: string | any;
  };
  expected?: {
    model_name?: string;
    provider_type?: string;
    should_succeed?: boolean;
    agent_output?: {
      contains_text?: string[];
      excludes_text?: string[];
      min_length?: number;
      max_length?: number;
      exact_match?: string;
      matches_pattern?: string;
    };
  };
}

interface TestResult {
  test_name: string;
  implementation: string;
  status: 'Passed' | 'Failed' | 'Error' | 'Skipped';
  duration_ms: number;
  error: string | null;
  actual_result: any;
  assertion_failures: Array<{
    assertion: any;
    actual_value: any;
    reason: string;
  }>;
}

async function runTest(testCase: TestCase): Promise<TestResult> {
  const startTime = Date.now();

  try {
    // Load agents library if needed
    await loadAgents();
    // Setup provider based on test case
    const { setup, input, expected } = testCase;
    const providerType = setup.provider_type || 'multi';
    const providerConfig = setup.provider_config || {};

    let provider: MultiModelProvider | OllamaModelProvider;

    if (providerType === 'ollama') {
      provider = new OllamaModelProvider({
        apiKey: providerConfig.api_key,
        baseURL: providerConfig.base_url,
      });
    } else if (providerType === 'multi') {
      const providerMap = new MultiModelProviderMap();
      if (providerConfig.ollama_api_key) {
        providerMap.addProvider(
          'ollama',
          new OllamaModelProvider({ apiKey: providerConfig.ollama_api_key })
        );
      }
      // Use environment variable if test config has placeholder, otherwise use config value
      let openaiApiKey = providerConfig.openai_api_key || '';
      if (openaiApiKey === 'test-key' || !openaiApiKey) {
        // Try to get from environment (works in both Node.js and Deno)
        if (typeof process !== 'undefined' && process.env) {
          openaiApiKey = process.env.OPENAI_API_KEY || '';
        } else {
          // @ts-ignore - Deno may not be available
          if (typeof Deno !== 'undefined' && Deno.env) {
            // @ts-ignore
            openaiApiKey = Deno.env.get('OPENAI_API_KEY') || '';
          }
        }
      }
      
      // Build provider options with optional parameters
      const providerOptions: any = {
        provider_map: providerMap,
        openai_api_key: openaiApiKey,
      };
      
      // Add optional OpenAI provider options
      if (providerConfig.openai_base_url !== undefined) {
        providerOptions.openai_base_url = providerConfig.openai_base_url;
      }
      if (providerConfig.openai_organization !== undefined) {
        providerOptions.openai_organization = providerConfig.openai_organization;
      }
      if (providerConfig.openai_project !== undefined) {
        providerOptions.openai_project = providerConfig.openai_project;
      }
      if (providerConfig.openai_use_responses !== undefined) {
        providerOptions.openai_use_responses = providerConfig.openai_use_responses;
      }
      
      provider = new MultiModelProvider(providerOptions);
    } else {
      throw new Error(`Unknown provider type: ${providerType}`);
    }

    // Get input
    const modelName = input.model_name;
    const runAgent = input.run_agent || false;
    const stream = input.stream || false;

    const actualResult: any = {
      model_name: modelName,
      provider_type: providerType,
    };
    let agentOutput: string | undefined;

    if (runAgent) {
      if (!agentsAvailable) {
        throw new Error(
          '@openai/agents package is required for agent tests. Install with: npm install @openai/agents'
        );
      }

      // Create agent configuration
      const agentConfig = input.agent_config || {};
      const agentOptions: any = { 
        name: testCase.name || 'test_agent',
        model: modelName 
      };

      if (agentConfig.system_prompt) {
        agentOptions.system = agentConfig.system_prompt;
      }

      // Set temperature to 0 via ModelSettings (plain object, not a class)
      const temperature = agentConfig.temperature !== undefined ? agentConfig.temperature : 0;
      agentOptions.model_settings = { temperature };

      // Create agent
      const agent = new Agent(agentOptions);

      // Get user input
      let userInput = input.user_input;
      if (userInput === undefined) {
        // Fallback to messages if user_input not provided
        const messages = input.messages || [];
        userInput = messages[0]?.content || 'Hello';
      }

      // Convert user_input to string if it's an object
      if (typeof userInput === 'object') {
        userInput = userInput.content || JSON.stringify(userInput);
      } else if (typeof userInput !== 'string') {
        userInput = String(userInput);
      }

      // Create runner
      const runner = new Runner({ modelProvider: provider });

      // Run agent (streaming or non-streaming)
      try {
        const result = await runner.run(agent, userInput, { stream });

        // Extract output from result
        if (stream) {
          // For streaming, result is an async iterable of stream events
          agentOutput = '';
          for await (const chunk of result) {
            // Extract text deltas from streaming events
            // Check multiple possible event types and structures
            const eventType = chunk?.type || chunk?.data?.type;
            
            // Handle output_text_delta events
            if (eventType === 'output_text_delta' || chunk?.type === 'output_text_delta') {
              // Extract delta from various possible locations
              let delta: string | undefined;
              
              // Direct delta property
              if (chunk?.delta && typeof chunk.delta === 'string') {
                delta = chunk.delta;
              }
              // Nested in data
              else if (chunk?.data?.delta && typeof chunk.data.delta === 'string') {
                delta = chunk.data.delta;
              }
              // In data.event
              else if (chunk?.data?.event?.delta && typeof chunk.data.event.delta === 'string') {
                delta = chunk.data.event.delta;
              }
              // In providerData
              else if (chunk?.providerData?.delta && typeof chunk.providerData.delta === 'string') {
                delta = chunk.providerData.delta;
              }
              
              if (delta) {
                agentOutput += delta;
              }
            }
            // Handle raw_model_stream_event with output_text_delta
            else if (eventType === 'raw_model_stream_event' || chunk?.type === 'raw_model_stream_event') {
              const data = chunk.data || chunk;
              if (data?.type === 'output_text_delta' && data.delta) {
                agentOutput += data.delta;
              } else if (data?.event?.type === 'output_text_delta' && data.event.delta) {
                agentOutput += data.event.delta;
              }
            }
            // Also check for final output in response_done events (as fallback)
            else if (eventType === 'response_done' || chunk?.type === 'response_done') {
              const response = chunk.response || chunk.data?.response;
              if (response?.output && Array.isArray(response.output) && response.output.length > 0) {
                const lastItem = response.output[response.output.length - 1];
                if (lastItem?.content && Array.isArray(lastItem.content)) {
                  const textParts = lastItem.content
                    .filter((c: any) => c.type === 'output_text' && c.text)
                    .map((c: any) => c.text);
                  if (textParts.length > 0) {
                    // Use final text if we don't have deltas yet, or as fallback
                    if (!agentOutput) {
                      agentOutput = textParts.join('');
                    }
                  }
                }
              }
            }
          }
        } else {
          // Non-streaming: extract from result object
          // The result structure depends on the agents library version
          if (result?.messages && Array.isArray(result.messages) && result.messages.length > 0) {
            // Get the last assistant message
            const lastMessage = result.messages[result.messages.length - 1];
            if (lastMessage?.content) {
              // content might be an array or string
              if (Array.isArray(lastMessage.content)) {
                agentOutput = lastMessage.content
                  .map((c: any) => c.text || c.content || JSON.stringify(c))
                  .join('');
              } else {
                agentOutput = lastMessage.content;
              }
            } else if (typeof lastMessage === 'string') {
              agentOutput = lastMessage;
            } else {
              agentOutput = JSON.stringify(lastMessage);
            }
          } else if (result?.content) {
            agentOutput = result.content;
          } else if (result?.state?.currentStep?.output) {
            // Try to extract from state.currentStep.output
            agentOutput = result.state.currentStep.output;
          } else if (result?.state?.lastModelResponse?.output_text) {
            // Try to extract from state.lastModelResponse.output_text
            agentOutput = result.state.lastModelResponse.output_text;
          } else if (typeof result === 'string') {
            agentOutput = result;
          } else {
            // Last resort: stringify, but try to extract text if it's a JSON string
            const str = JSON.stringify(result);
            try {
              const parsed = JSON.parse(str);
              if (parsed.state?.currentStep?.output) {
                agentOutput = parsed.state.currentStep.output;
              } else if (parsed.state?.lastModelResponse?.output_text) {
                agentOutput = parsed.state.lastModelResponse.output_text;
              } else {
                agentOutput = str;
              }
            } catch {
              agentOutput = str;
            }
          }
        }

        actualResult.agent_output = agentOutput;
      } catch (e: any) {
        throw new Error(`Agent execution failed: ${e.message || String(e)}`);
      }
    } else {
      // Just verify model creation (original behavior)
      const model = await provider.getModel(modelName);
      // Model created successfully, no further action needed
    }

    const durationMs = Date.now() - startTime;

    // Check expected results
    const assertionFailures: TestResult['assertion_failures'] = [];

    // Validate model name
    if (expected?.model_name && modelName !== expected.model_name) {
      assertionFailures.push({
        assertion: { field: 'model_name', operator: 'equals', value: expected.model_name },
        actual_value: modelName,
        reason: `Expected model_name ${expected.model_name}, got ${modelName}`,
      });
    }

    // Validate agent output if provided
    if (runAgent && agentOutput !== undefined) {
      const agentOutputExpectation = expected?.agent_output;

      if (agentOutputExpectation) {
        // Check contains_text (case-insensitive)
        if (agentOutputExpectation.contains_text) {
          for (const text of agentOutputExpectation.contains_text) {
            if (!agentOutput.toLowerCase().includes(text.toLowerCase())) {
              assertionFailures.push({
                assertion: { field: 'agent_output', operator: 'contains', value: text },
                actual_value: agentOutput,
                reason: `Expected agent output to contain '${text}', but it didn't`,
              });
            }
          }
        }

        // Check excludes_text (case-insensitive)
        if (agentOutputExpectation.excludes_text) {
          for (const text of agentOutputExpectation.excludes_text) {
            if (agentOutput.toLowerCase().includes(text.toLowerCase())) {
              assertionFailures.push({
                assertion: { field: 'agent_output', operator: 'excludes', value: text },
                actual_value: agentOutput,
                reason: `Expected agent output to not contain '${text}', but it did`,
              });
            }
          }
        }

        // Check min_length
        if (agentOutputExpectation.min_length !== undefined) {
          const minLen = agentOutputExpectation.min_length;
          if (agentOutput.length < minLen) {
            assertionFailures.push({
              assertion: { field: 'agent_output', operator: 'min_length', value: minLen },
              actual_value: agentOutput.length,
              reason: `Expected agent output length >= ${minLen}, got ${agentOutput.length}`,
            });
          }
        }

        // Check max_length
        if (agentOutputExpectation.max_length !== undefined && agentOutputExpectation.max_length !== null) {
          const maxLen = agentOutputExpectation.max_length;
          if (agentOutput.length > maxLen) {
            assertionFailures.push({
              assertion: { field: 'agent_output', operator: 'max_length', value: maxLen },
              actual_value: agentOutput.length,
              reason: `Expected agent output length <= ${maxLen}, got ${agentOutput.length}`,
            });
          }
        }

        // Check exact_match (rarely used)
        if (agentOutputExpectation.exact_match) {
          const expectedText = agentOutputExpectation.exact_match;
          if (agentOutput !== expectedText) {
            assertionFailures.push({
              assertion: { field: 'agent_output', operator: 'equals', value: expectedText },
              actual_value: agentOutput,
              reason: 'Expected exact match, but got different output',
            });
          }
        }
      }
    }

    const status = assertionFailures.length === 0 ? 'Passed' : 'Failed';

    return {
      test_name: testCase.name,
      implementation: 'typescript',
      status,
      duration_ms: durationMs,
      error: null,
      actual_result: actualResult,
      assertion_failures: assertionFailures,
    };
  } catch (error: any) {
    const durationMs = Date.now() - startTime;
    return {
      test_name: testCase.name,
      implementation: 'typescript',
      status: 'Error',
      duration_ms: durationMs,
      error: error.message || String(error),
      actual_result: null,
      assertion_failures: [],
    };
  }
}

// Main execution
const testCaseFile = process.argv[2];
if (!testCaseFile) {
  console.error('Usage: run_test.ts <test_case.json>');
  process.exit(1);
}

const testCaseJson = readFileSync(testCaseFile, 'utf-8');
const testCase: TestCase = JSON.parse(testCaseJson);

runTest(testCase)
  .then((result) => {
    console.log(JSON.stringify(result));
  })
  .catch((error) => {
    console.error(JSON.stringify({
      test_name: testCase.name || 'unknown',
      implementation: 'typescript',
      status: 'Error',
      duration_ms: 0,
      error: error.message || String(error),
      actual_result: null,
      assertion_failures: [],
    }));
    process.exit(1);
  });

