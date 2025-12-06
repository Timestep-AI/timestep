/** Example demonstrating DBOS workflows for durable agent execution. */

import {
  runAgentWorkflow,
  queueAgentWorkflow,
  createScheduledAgentWorkflow,
  configureDBOS,
  ensureDBOSLaunched,
} from '../timestep/index.ts';
import {
  Agent,
  OpenAIConversationsSession,
  ModelSettings,
} from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';

async function exampleDurableWorkflow(): Promise<void> {
  /** Example: Run an agent in a durable workflow. */
  console.log('=== Example 1: Durable Workflow ===');
  
  // Configure and launch DBOS
  configureDBOS();
  await ensureDBOSLaunched();
  
  // Create agent and session
  const agent = new Agent({
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4.1',
    modelSettings: { temperature: 0 },
    name: 'Assistant',
  });
  const session = new OpenAIConversationsSession();
  
  // Prepare input
  const inputItems: AgentInputItem[] = [
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: "What's 2+2?" }] }
  ];
  
  // Run in durable workflow
  const result = await runAgentWorkflow(
    agent,
    inputItems,
    session,
    false,
    undefined,
    undefined,
    undefined,
    'example-workflow-1'  // Idempotency key
  );
  
  console.log(`Result: ${result.output}`);
  console.log();
}

async function exampleQueuedWorkflow(): Promise<void> {
  /** Example: Enqueue agent runs with rate limiting. */
  console.log('\n=== Example 2: Queued Workflow ===');
  
  await ensureDBOSLaunched();
  
  const agent = new Agent({
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4.1',
    modelSettings: { temperature: 0 },
    name: 'Assistant',
  });
  const session = new OpenAIConversationsSession();
  
  const inputItems: AgentInputItem[] = [
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: "What's 3+3?" }] }
  ];
  
  // Enqueue workflow (returns handle immediately)
  const handle = await queueAgentWorkflow(
    agent,
    inputItems,
    session,
    false,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    1,  // priority (higher priority)
    'example-queue-1'  // deduplication ID
  );
  
  // Wait for result when ready
  const result = await handle.getResult();
  console.log(`Result: ${result.output}`);
  console.log();
}

async function exampleScheduledWorkflow(): Promise<void> {
  /** Example: Schedule periodic agent runs. */
  console.log('\n=== Example 3: Scheduled Workflow ===');
  
  await ensureDBOSLaunched();
  
  const agent = new Agent({
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4.1',
    modelSettings: { temperature: 0 },
    name: 'Assistant',
  });
  const session = new OpenAIConversationsSession();
  
  const inputItems: AgentInputItem[] = [
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: "What's 4+4?" }] }
  ];
  
  // Create scheduled workflow (runs every 5 minutes)
  // Note: In production, you'd typically run this in a long-lived process
  createScheduledAgentWorkflow(
    '*/5 * * * *',  // Every 5 minutes
    agent,
    inputItems,
    session,
    false
  );
  
  console.log('Scheduled workflow created. It will run every 5 minutes.');
  console.log('Note: Keep the process running for scheduled workflows to execute.');
  console.log();
}

async function main(): Promise<void> {
  /** Run all examples. */
  // Check for API key
  const env = typeof process !== 'undefined' ? process.env : {};
  if (!env['OPENAI_API_KEY']) {
    console.log('Warning: OPENAI_API_KEY not set. Examples may fail.');
  }
  
  try {
    await exampleDurableWorkflow();
    await exampleQueuedWorkflow();
    await exampleScheduledWorkflow();
  } catch (e) {
    console.error('Error running examples:', e);
    if (e instanceof Error) {
      console.error(e.stack);
    }
  }
}

if (import.meta.main) {
  main().catch(console.error);
}

