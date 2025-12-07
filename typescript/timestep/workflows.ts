/** DBOS workflows for durable agent execution. */

import { DBOS, WorkflowQueue, WorkflowHandle } from '@dbos-inc/dbos-sdk';
import { configureDBOS, ensureDBOSLaunched, getDBOSConnectionString, isDBOSLaunched } from './dbos_config.ts';
import { RunStateStore } from './run_state_store.ts';
import { runAgent, defaultResultProcessor } from './index.ts';
import { Agent, Session, RunState } from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';
import { loadAgent } from './agent_store.ts';
import { loadSession } from './session_store.ts';
import { DatabaseConnection } from './db_connection.ts';

// Default queue for agent workflows with rate limiting
let defaultQueue: WorkflowQueue | null = null;

function getDefaultQueue(): WorkflowQueue {
  if (!defaultQueue) {
    // Rate limit: 50 requests per 60 seconds (conservative for LLM APIs)
    defaultQueue = new WorkflowQueue('timestep_agent_queue', {
      rateLimit: { limitPerPeriod: 50, periodSec: 60 }
    });
  }
  return defaultQueue;
}

async function loadAgentStep(agentId: string): Promise<Agent> {
  /**
   * Step that loads an agent from the database.
   */
  const connectionString = getDBOSConnectionString();
  if (!connectionString) {
    throw new Error('DBOS connection string not available');
  }

  const db = new DatabaseConnection({ connectionString });
  await db.connect();
  try {
    const agent = await loadAgent(agentId, db);
    return agent;
  } finally {
    await db.disconnect();
  }
}

async function loadSessionDataStep(sessionId: string): Promise<any> {
  /**
   * Step that loads session data from the database.
   * 
   * Returns serializable session data object, not the Session object itself.
   */
  const connectionString = getDBOSConnectionString();
  if (!connectionString) {
    throw new Error('DBOS connection string not available');
  }

  const db = new DatabaseConnection({ connectionString });
  await db.connect();
  try {
    const sessionData = await loadSession(sessionId, db);
    if (!sessionData) {
      throw new Error(`Session with id ${sessionId} not found`);
    }
    return sessionData;
  } finally {
    await db.disconnect();
  }
}

const runAgentStep = DBOS.registerStep(async (
  agent: Agent,
  runInput: AgentInputItem[] | RunState<any, any>,
  sessionData: any,
  stream: boolean,
  resultProcessor?: (result: any) => Promise<any>
): Promise<any> => {
  /** Step that runs an agent. Returns serializable dict. */
  // Reconstruct Session object from sessionData
  const sessionType = sessionData.session_type || '';
  const internalSessionId = sessionData.session_id;

  let session: Session;
  if (sessionType.includes('OpenAIConversationsSession')) {
    const { OpenAIConversationsSession } = await import('@openai/agents');
    session = new OpenAIConversationsSession({ conversationId: internalSessionId });
  } else {
    throw new Error(`Unsupported session type: ${sessionType}`);
  }

  const processor = resultProcessor || defaultResultProcessor;

  // Run agent - returns RunResult
  const runResult = await runAgent(agent, runInput, session, stream, processor);

  // Extract output - RunResult might have output in different formats
  // Try multiple methods to extract the output
  let output: any = null;
  
  // Method 1: Check final_output directly
  if (runResult.final_output !== undefined && runResult.final_output !== null) {
    output = runResult.final_output;
  }
  // Method 2: Use to_input_list() to get all items, then filter for assistant messages
  else if (runResult.to_input_list && typeof runResult.to_input_list === 'function') {
    try {
      const allItems = runResult.to_input_list();
      output = allItems.filter(
        (item: any) => item.type === 'message' && item.role === 'assistant'
      );
    } catch (e) {
      // Fall through to next method
    }
  }
  // Method 3: Check new_items
  if (!output && runResult.new_items) {
    const outputItems: any[] = [];
    for (const item of runResult.new_items) {
      try {
        const inputItem = (item as any).to_input_item();
        if (inputItem.type === 'message' && inputItem.role === 'assistant') {
          outputItems.push(inputItem);
        }
      } catch (e) {
        // Skip items that can't be converted
      }
    }
    if (outputItems.length > 0) {
      output = outputItems;
    }
  }
  // Method 4: Check output property directly
  if (!output && runResult.output !== undefined) {
    output = runResult.output;
  }

  // Extract only serializable data - RunResult has non-serializable objects
  return {
    final_output: output,
    interruptions: runResult.interruptions || []
  };
});

async function saveStateStep(
  result: any,
  agentId: string,
  sessionId: string | undefined
): Promise<void> {
  /**
   * Step that saves agent state. This must be a step because it accesses the database.
   */
  await DBOS.runStep(
    async () => {
      const connectionString = getDBOSConnectionString();
      if (!connectionString) {
        throw new Error('DBOS connection string not available');
      }

      const db = new DatabaseConnection({ connectionString });
      await db.connect();
      try {
        const agent = await loadAgent(agentId, db);
        const stateStore = new RunStateStore({ agent, sessionId });

        if (result.state) {
          await stateStore.save(result.state);
        } else if (result.toState) {
          const state = result.toState();
          await stateStore.save(state);
        } else {
          throw new Error('Result does not have state or toState() method');
        }
      } finally {
        await db.disconnect();
      }
    },
    { name: 'saveState' }
  );
}

async function executeAgentWithStateHandling(
  agentId: string,
  inputItems: AgentInputItem[] | RunState<any, any>,
  sessionId: string,
  stream: boolean,
  resultProcessor: ((result: any) => Promise<any>) | undefined,
  timeoutSeconds: number | undefined
): Promise<any> {
  /** Execute agent. Returns output. State persistence handled by RunStateStore outside workflow. */
  // Step 1: Load agent from database
  const agent = await loadAgentStep(agentId);

  // Step 2: Load session data from database
  const sessionData = await loadSessionDataStep(sessionId);

  // Step 3: Run agent - returns dict with output and interruptions
  const resultDict = await runAgentStep(agent, inputItems, sessionData, stream, resultProcessor);

  // Return output - state saving happens outside workflow via RunStateStore
  return {
    output: resultDict.final_output,
    interruptions: resultDict.interruptions
  };
}

/**
 * Register the generic workflows before DBOS launch.
 * This must be called before ensureDBOSLaunched().
 */
export async function registerGenericWorkflows(): Promise<void> {
  // The workflow is already registered via DBOS.registerWorkflow() decorator
  // This function is kept for compatibility but does nothing
  return;
}

const agentWorkflow = DBOS.registerWorkflow(async function agentWorkflow(
  agentId: string,
  inputItemsJson: string,  // Serialized input items
  sessionId: string,
  stream: boolean = false,
  timeoutSeconds?: number
) {
  /**
   * Workflow that runs an agent using IDs stored in the database.
   */
  // Deserialize input items
  const inputItemsData = JSON.parse(inputItemsJson);
  const inputItems = inputItemsData as AgentInputItem[] | RunState<any, any>;

  // Execute and return result
  const result = await executeAgentWithStateHandling(
    agentId,
    inputItems,
    sessionId,
    stream,
    undefined,
    timeoutSeconds
  );
  
  // Ensure we return a proper result object with output
  if (!result) {
    throw new Error('Workflow returned null/undefined result');
  }
  
  // Extract output - handle both output and final_output keys
  const output = result.output || result.final_output || null;
  const interruptions = result.interruptions || [];
  
  // Ensure output is serializable - deep clone to remove any non-serializable references
  let serializableOutput: any = null;
  if (output) {
    try {
      serializableOutput = JSON.parse(JSON.stringify(output));
    } catch (e) {
      // If serialization fails, try to extract just the text
      if (Array.isArray(output)) {
        serializableOutput = output.map((item: any) => {
          if (item && typeof item === 'object') {
            return JSON.parse(JSON.stringify(item));
          }
          return item;
        });
      } else {
        serializableOutput = output;
      }
    }
  }
  
  const serializableInterruptions = interruptions ? JSON.parse(JSON.stringify(interruptions)) : [];
  
  // Return a plain object that DBOS can serialize - use explicit keys
  // IMPORTANT: For queued workflows in DBOS TypeScript, the result must be a plain object
  // with no nested complex structures that might cause serialization issues
  const workflowResult = {
    output: serializableOutput,
    interruptions: serializableInterruptions
  };
  
  // Verify it's serializable before returning - this is critical for queued workflows
  try {
    const testSerialization = JSON.stringify(workflowResult);
    if (!testSerialization || testSerialization === '{}') {
      throw new Error('Workflow result serialization produced empty object');
    }
  } catch (e: any) {
    throw new Error(`Workflow result is not serializable: ${e.message}`);
  }
  
  // Log the result structure for debugging
  console.log('agentWorkflow returning result with keys:', Object.keys(workflowResult));
  console.log('agentWorkflow output type:', typeof workflowResult.output);
  console.log('agentWorkflow output is array?', Array.isArray(workflowResult.output));
  
  return workflowResult;
});

export async function runAgentWorkflow(
  agentId: string,
  inputItems: AgentInputItem[] | RunState<any, any>,
  sessionId: string,
  stream: boolean = false,
  workflowId?: string,
  timeoutSeconds?: number
): Promise<any> {
  /**
   * Run an agent in a durable DBOS workflow.
   * 
   * This workflow automatically saves state on interruptions and can be resumed
   * if the process crashes or restarts.
   */
  // Ensure DBOS is configured (but not launched yet - workflows must be registered before launch)
  if (!getDBOSConnectionString()) {
    await configureDBOS();
  }

  // Ensure generic workflow is registered
  await registerGenericWorkflows();

  // Launch DBOS if not already launched (after workflow registration)
  if (!isDBOSLaunched()) {
    await ensureDBOSLaunched();
  }

  // Serialize input items
  const inputItemsJson = JSON.stringify(inputItems);

  // Call the workflow with serializable parameters
  if (workflowId) {
    const handle = await DBOS.startWorkflow(agentWorkflow, { workflowID: workflowId })(
      agentId, inputItemsJson, sessionId, stream, timeoutSeconds
    );
    return await handle.getResult();
  } else {
    return await agentWorkflow(agentId, inputItemsJson, sessionId, stream, timeoutSeconds);
  }
}

export async function queueAgentWorkflow(
  agentId: string,
  inputItems: AgentInputItem[] | RunState<any, any>,
  sessionId: string,
  stream: boolean = false,
  queueName?: string,
  workflowId?: string,
  timeoutSeconds?: number,
  priority?: number,
  deduplicationId?: string
): Promise<WorkflowHandle<any>> {
  /**
   * Enqueue an agent run in a DBOS queue with rate limiting support.
   * 
   * This is useful for managing concurrent agent executions and respecting
   * LLM API rate limits.
   * 
   * @returns WorkflowHandle that can be used to get the result
   */
  // Ensure DBOS is configured (but not launched yet - workflows must be registered before launch)
  if (!getDBOSConnectionString()) {
    await configureDBOS();
  }

  // Ensure generic workflow is registered
  await registerGenericWorkflows();

  // Launch DBOS if not already launched (after workflow registration)
  if (!isDBOSLaunched()) {
    await ensureDBOSLaunched();
  }

  // Get queue
  const queue = queueName ? new WorkflowQueue(queueName) : getDefaultQueue();

  // Serialize input items
  const inputItemsJson = JSON.stringify(inputItems);

  // Enqueue options
  const enqueueOptions: any = {};
  if (priority !== undefined) {
    enqueueOptions.priority = priority;
  }
  if (deduplicationId) {
    enqueueOptions.deduplicationID = deduplicationId;
  }

  const startParams: any = {
    queueName: queue.name,
    ...enqueueOptions
  };

  if (workflowId) {
    startParams.workflowID = workflowId;
  }

  if (timeoutSeconds) {
    startParams.timeoutMS = timeoutSeconds * 1000;
  }

  const handle = await DBOS.startWorkflow(agentWorkflow, startParams)(
    agentId, inputItemsJson, sessionId, stream, timeoutSeconds
  );
  return handle;
}

export async function createScheduledAgentWorkflow(
  crontab: string,
  agentId: string,
  inputItems: AgentInputItem[] | RunState<any, any>,
  sessionId: string,
  stream: boolean = false
): Promise<void> {
  /**
   * Create a scheduled workflow that runs an agent periodically.
   * 
   * This function registers a scheduled workflow with DBOS. The workflow will
   * run automatically according to the crontab schedule.
   * 
   * Example:
   *   createScheduledAgentWorkflow(
   *     '0 0,6,12,18 * * *',  // Every 6 hours
   *     agentId,
   *     inputItems,
   *     sessionId
   *   );
   */
  // Check if DBOS is already launched - if so, we can't register new scheduled workflows
  if (isDBOSLaunched()) {
    throw new Error(
      'Cannot create scheduled workflow after DBOS launch. ' +
      'Scheduled workflows must be registered before DBOS.launch() is called. ' +
      'Call createScheduledAgentWorkflow() before ensureDBOSLaunched().'
    );
  }

  // Ensure DBOS is configured (but not launched yet)
  if (!getDBOSConnectionString()) {
    await configureDBOS();
  }

  // Serialize input items
  const inputItemsJson = JSON.stringify(inputItems);

  // Register a scheduled workflow
  const scheduledWorkflow = DBOS.registerScheduled(
    async (scheduledTime: Date, startTime: Date) => {
      return await agentWorkflow(agentId, inputItemsJson, sessionId, stream, undefined);
    },
    { crontab }
  );
}
