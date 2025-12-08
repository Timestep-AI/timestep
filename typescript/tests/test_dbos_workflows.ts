/** Tests for DBOS workflow functionality. */

import { test, beforeAll, afterAll } from 'vitest';
import {
  configureDBOS,
  ensureDBOSLaunched,
  cleanupDBOS,
  runAgentWorkflow,
  queueAgentWorkflow,
  createScheduledAgentWorkflow,
  registerGenericWorkflows,
} from '../timestep/index.ts';
import {
  Agent,
  OpenAIConversationsSession,
  ModelSettings,
} from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';
import { saveAgent } from '../timestep/stores/agent_store/store.ts';
import { saveSession } from '../timestep/stores/session_store/store.ts';

let dbosAvailable = false;

beforeAll(async () => {
  try {
    const { DBOS } = await import('@dbos-inc/dbos-sdk');
    
    // Step 1: Destroy DBOS (per DBOS testing best practices)
    // Note: Don't destroy registry - workflows are registered via decorators at import time
    try {
      DBOS.destroy({ destroyRegistry: false });
    } catch (error) {
      // Ignore if DBOS doesn't exist yet
    }
    
    // Step 2: Configure DBOS
    console.log('Configuring DBOS...');
    await configureDBOS({ name: 'timestep-test' });
    console.log('DBOS configured');
    
    // Step 3: Drop system database (per DBOS testing best practices)
    // This ensures a clean state for migrations
    // First, manually drop the dbos schema to ensure complete cleanup
    // Then call dropSystemDB which should handle the rest
    try {
      const env = typeof process !== 'undefined' ? process.env : {};
      const dbUrl = env['PG_CONNECTION_URI'];
      if (dbUrl) {
        console.log('Manually dropping dbos schema...');
        const pg = await import('pg');
        const { Pool } = pg;
        const pool = new Pool({ connectionString: dbUrl });
        try {
          // Drop the dbos schema if it exists (CASCADE to drop all objects)
          await pool.query('DROP SCHEMA IF EXISTS dbos CASCADE');
          console.log('dbos schema dropped');
        } catch (schemaError: any) {
          // Ignore errors - schema might not exist or might be in use
          console.log('Schema drop skipped:', schemaError.message);
        } finally {
          await pool.end();
        }
      }
      
      // Now call dropSystemDB to ensure complete cleanup
      console.log('Dropping DBOS system database...');
      await DBOS.dropSystemDB();
      console.log('DBOS system database dropped');
    } catch (error) {
      // Ignore if drop fails (e.g., database in use or already clean)
      console.log('System database drop skipped (may already be clean)');
    }
    
    // Step 4: Launch DBOS (workflows are already registered via decorators)
    // DBOS.launch() will automatically run migrations to ensure schema is up-to-date
    // This allows Python and TypeScript to share the same database schema
    // The migrations are idempotent and will create missing tables if needed
    console.log('Launching DBOS...');
    await ensureDBOSLaunched();
    console.log('DBOS launched');
    
    dbosAvailable = true;
  } catch (error: any) {
    console.error('DBOS setup failed:', error);
    console.error('Exiting test suite due to DBOS setup failure');
    // Force immediate exit to stop all tests
    process.exit(1);
  }
}, 120000); // 120 second timeout

afterAll(async () => {
  console.log('Cleaning up DBOS resources after all tests...');
  await cleanupDBOS();
  console.log('DBOS resources cleaned up.');
});

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

  // Create agent and session
  const agentName = `test-assistant-basic-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  const agent = new Agent({
    instructions: 'You are a helpful assistant. Answer concisely.',
    model: 'gpt-4.1',
    modelSettings: { temperature: 0 },
    name: agentName,
  });
  const session = new OpenAIConversationsSession();
  
  // Save agent and session to database (stores manage connections internally)
  let agentId: string;
  let sessionId: string;
  agentId = await saveAgent(agent);
  sessionId = await saveSession(session);
  
  const inputItems: AgentInputItem[] = [
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: "Say 'hello' and nothing else." }] }
  ];
  
  const result = await runAgentWorkflow(
    agentId,
    inputItems,
    sessionId,
    false,
    'test-workflow-1'
  );
  
  if (!result) {
    throw new Error('Basic workflow test failed: result is null or undefined');
  }
  
  // Extract text output from result
  // result.output is an array of response items, not a string
  let outputText: string | undefined;
  
  if (result.output) {
    if (Array.isArray(result.output)) {
      // Extract text from output array (find message items with output_text content)
      const textParts: string[] = [];
      for (const item of result.output) {
        if (item.type === 'message' && item.role === 'assistant' && item.content) {
          for (const block of item.content) {
            if (block.type === 'output_text' && block.text) {
              textParts.push(block.text);
            }
          }
        }
      }
      outputText = textParts.join(' ');
    } else if (typeof result.output === 'string') {
      outputText = result.output;
    }
  }
  
  if (!outputText) {
    console.log('Result keys:', Object.keys(result));
    console.log('Result.output type:', typeof result.output);
    console.log('Result.output is array?', Array.isArray(result.output));
    if (Array.isArray(result.output)) {
      console.log('Result.output length:', result.output.length);
      console.log('Result.output[0]:', result.output[0]);
    }
    throw new Error(`Basic workflow test failed: could not extract text from result.output. Result keys: ${Object.keys(result).join(', ')}`);
  }
  
  if (!outputText.toLowerCase().includes('hello')) {
    throw new Error(`Basic workflow test failed: output text "${outputText.substring(0, 100)}" does not contain "hello"`);
  }
});

test('test_queue_agent_workflow', async () => {
  const env = typeof process !== 'undefined' ? process.env : {};
  if (!env['OPENAI_API_KEY']) {
    return; // Skip if no API key
  }

  // Create agent and session
  const agentName = `test-assistant-queue-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  const agent = new Agent({
    instructions: 'You are a helpful assistant. Answer concisely.',
    model: 'gpt-4.1',
    modelSettings: { temperature: 0 },
    name: agentName,
  });
  const session = new OpenAIConversationsSession();
  
  // Save agent and session to database (stores manage connections internally)
  let agentId: string;
  let sessionId: string;
  agentId = await saveAgent(agent);
  sessionId = await saveSession(session);
  
  const inputItems: AgentInputItem[] = [
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: "Say 'queued' and nothing else." }] }
  ];
  
  const handle = await queueAgentWorkflow(
    agentId,
    inputItems,
    sessionId,
    false,
    undefined,
    undefined,
    undefined,
    undefined,
    `test-queue-${Date.now()}-${Math.random().toString(36).substring(7)}`
  );
  
  // Poll for workflow completion before getting result
  // Use same pattern as Python - check every second
  const maxWait = 90;
  const startTime = Date.now();
  let status: string | undefined;
  let lastStatus: string | undefined;
  
  while (Date.now() - startTime < maxWait * 1000) {
    let statusObj: any;
    try {
      statusObj = handle.getStatus();
      // Handle if getStatus returns a Promise
      if (statusObj instanceof Promise) {
        statusObj = await statusObj;
      }
      // Extract status from status object
      if (statusObj && typeof statusObj === 'object' && 'status' in statusObj) {
        status = String(statusObj.status);
      } else {
        status = String(statusObj);
      }
    } catch (e) {
      console.error('Error getting status:', e);
      status = 'ERROR';
      break;
    }
    
    // Log status changes for debugging
    if (status !== lastStatus) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`[${elapsed}s] Status: ${lastStatus} -> ${status}`);
      lastStatus = status;
    }
    
    if (status === 'SUCCESS' || status === 'FAILED' || status === 'ERROR') {
      break;
    }
    await new Promise(resolve => setTimeout(resolve, 1000)); // Check every second
  }
  
  if (!status || (status !== 'SUCCESS' && status !== 'FAILED' && status !== 'ERROR')) {
    throw new Error(`Workflow did not complete after ${maxWait} seconds. Status: ${status}`);
  }
  
  // getResult() is ASYNC in TypeScript DBOS - must await it!
  let result: any;
  try {
    result = await handle.getResult();
  } catch (e: any) {
    throw new Error(`Failed to get result from workflow handle: ${e.message}`);
  }
  
  console.log('Workflow handle result:', JSON.stringify(result, null, 2));
  console.log('Result type:', typeof result);
  console.log('Result keys:', result ? Object.keys(result) : 'null');
  
  if (!result) {
    throw new Error(`Queued workflow test failed: result is null or undefined`);
  }
  
  if (!result.output && !result.final_output) {
    throw new Error(`Queued workflow test failed: no output in result. Result: ${JSON.stringify(result)}`);
  }
  
  // Extract text output from result (output is an array)
  let outputText: string | undefined;
  if (result.output) {
    if (Array.isArray(result.output)) {
      const textParts: string[] = [];
      for (const item of result.output) {
        if (item.type === 'message' && item.role === 'assistant' && item.content) {
          for (const block of item.content) {
            if (block.type === 'output_text' && block.text) {
              textParts.push(block.text);
            }
          }
        }
      }
      outputText = textParts.join(' ');
    } else if (typeof result.output === 'string') {
      outputText = result.output;
    }
  }
  
  if (!outputText || !outputText.toLowerCase().includes('queued')) {
    throw new Error(`Queued workflow test failed: output text "${outputText || 'missing'}" does not contain "queued". Full result: ${JSON.stringify(result, null, 2)}`);
  }
});

test('test_create_scheduled_workflow', async () => {
  // This test verifies that scheduled workflows must be created before DBOS launch.
  // Since DBOS is launched in beforeAll, this test should fail with an appropriate error.
  const agentName = `test-assistant-scheduled-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  const agent = new Agent({
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4.1',
    modelSettings: { temperature: 0 },
    name: agentName,
  });
  const session = new OpenAIConversationsSession();
  
  // Save agent and session to database (stores manage connections internally)
  let agentId: string;
  let sessionId: string;
  agentId = await saveAgent(agent);
  sessionId = await saveSession(session);
  
  const inputItems: AgentInputItem[] = [
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Hello' }] }
  ];
  
  // This should raise an error because DBOS is already launched
  try {
    await createScheduledAgentWorkflow(
      '0 * * * *',  // Every hour
      agentId,
      inputItems,
      sessionId,
      false
    );
    throw new Error('Expected createScheduledAgentWorkflow to throw an error when DBOS is already launched');
  } catch (error: any) {
    // Verify that the error message is appropriate
    if (!error.message.includes('Cannot create scheduled workflow after DBOS launch')) {
      throw new Error(`Unexpected error message: ${error.message}`);
    }
    // This is the expected behavior - scheduled workflows must be registered before launch
  }
});

