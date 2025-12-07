/** Tests for DBOS workflow functionality. */

import { test, beforeAll } from 'vitest';
import {
  configureDBOS,
  ensureDBOSLaunched,
  runAgentWorkflow,
  queueAgentWorkflow,
  createScheduledAgentWorkflow,
} from '../timestep/index.ts';
import {
  Agent,
  OpenAIConversationsSession,
  ModelSettings,
} from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';

let dbosAvailable = false;

beforeAll(async () => {
  try {
    console.log('Configuring DBOS...');
    await configureDBOS({ name: 'timestep-test' });
    console.log('DBOS configured, launching...');
    await ensureDBOSLaunched();
    console.log('DBOS launched successfully');
    dbosAvailable = true;
  } catch (error: any) {
    console.error('DBOS setup failed:', error);
    // Exit immediately on failure to stop test suite
    process.exit(1);
  }
}, 120000); // 120 second timeout

test('test_configure_dbos', () => {
  if (!dbosAvailable) {
    return; // Skip if database not available
  }
  // Configuration is done in beforeAll
  // If we get here, configuration worked
});

test('test_run_agent_workflow_basic', async () => {
  const env = typeof process !== 'undefined' ? process.env : {};
  if (!env['OPENAI_API_KEY']) {
    return; // Skip if no API key
  }

  const agent = new Agent({
    instructions: 'You are a helpful assistant. Answer concisely.',
    model: 'gpt-4.1',
    modelSettings: { temperature: 0 },
    name: 'Test Assistant',
  });
  const session = new OpenAIConversationsSession();
  
  const inputItems: AgentInputItem[] = [
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: "Say 'hello' and nothing else." }] }
  ];
  
  const result = await runAgentWorkflow(
    agent,
    inputItems,
    session,
    false,
    undefined,
    undefined,
    undefined,
    'test-workflow-1'
  );
  
  if (!result || !result.output || !result.output.toLowerCase().includes('hello')) {
    throw new Error('Basic workflow test failed');
  }
});

test('test_queue_agent_workflow', async () => {
  const env = typeof process !== 'undefined' ? process.env : {};
  if (!env['OPENAI_API_KEY']) {
    return; // Skip if no API key
  }

  const agent = new Agent({
    instructions: 'You are a helpful assistant. Answer concisely.',
    model: 'gpt-4.1',
    modelSettings: { temperature: 0 },
    name: 'Test Assistant',
  });
  const session = new OpenAIConversationsSession();
  
  const inputItems: AgentInputItem[] = [
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: "Say 'queued' and nothing else." }] }
  ];
  
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
    1,
    'test-queue-1'
  );
  
  const result = await handle.getResult();
  if (!result || !result.output || !result.output.toLowerCase().includes('queued')) {
    throw new Error('Queued workflow test failed');
  }
});

test('test_create_scheduled_workflow', () => {
  const agent = new Agent({
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4.1',
    modelSettings: { temperature: 0 },
    name: 'Test Assistant',
  });
  const session = new OpenAIConversationsSession();
  
  const inputItems: AgentInputItem[] = [
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Hello' }] }
  ];
  
  // This should not raise an error
  createScheduledAgentWorkflow(
    '0 * * * *',  // Every hour
    agent,
    inputItems,
    session,
    false
  );
});

