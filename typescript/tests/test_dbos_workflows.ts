/** Tests for DBOS workflow functionality. */

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

// Simple test runner since we don't have a test framework set up
async function runTests(): Promise<void> {
  console.log('Running DBOS workflow tests...\n');
  
  // Setup
  configureDBOS({ name: 'timestep-test' });
  await ensureDBOSLaunched();
  console.log('✓ DBOS configured and launched');
  
  const env = typeof process !== 'undefined' ? process.env : {};
  const hasApiKey = !!env['OPENAI_API_KEY'];
  
  if (!hasApiKey) {
    console.log('\n⚠ Skipping tests that require OPENAI_API_KEY');
    return;
  }
  
  // Test 1: Basic workflow
  try {
    console.log('\nTest 1: Basic durable workflow...');
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
    
    if (result && result.output && result.output.toLowerCase().includes('hello')) {
      console.log('✓ Basic workflow test passed');
    } else {
      console.log('✗ Basic workflow test failed');
    }
  } catch (e) {
    console.log('✗ Basic workflow test failed:', e);
  }
  
  // Test 2: Queued workflow
  try {
    console.log('\nTest 2: Queued workflow...');
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
    if (result && result.output && result.output.toLowerCase().includes('queued')) {
      console.log('✓ Queued workflow test passed');
    } else {
      console.log('✗ Queued workflow test failed');
    }
  } catch (e) {
    console.log('✗ Queued workflow test failed:', e);
  }
  
  // Test 3: Scheduled workflow creation
  try {
    console.log('\nTest 3: Scheduled workflow creation...');
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
    
    createScheduledAgentWorkflow(
      '0 * * * *',  // Every hour
      agent,
      inputItems,
      session,
      false
    );
    
    console.log('✓ Scheduled workflow creation test passed');
  } catch (e) {
    console.log('✗ Scheduled workflow creation test failed:', e);
  }
  
  console.log('\nTests completed.');
}

if (import.meta.main) {
  runTests().catch(console.error);
}

