/** DBOS workflows for durable agent execution. */

import { DBOS, WorkflowQueue, WorkflowHandle } from '@dbos-inc/dbos-sdk';
import { configureDBOS, ensureDBOSLaunched } from './dbos_config.ts';
import { RunStateStore } from './run_state_store.ts';
import { runAgent, defaultResultProcessor } from './index.ts';
import { Agent, Session, RunState } from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';

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

async function runAgentStep(
  agent: Agent,
  runInput: AgentInputItem[] | RunState<any, any>,
  session: Session,
  stream: boolean,
  resultProcessor?: (result: any) => Promise<any>
): Promise<any> {
  /**
   * Step that runs an agent. This must be a step because it's non-deterministic.
   */
  const processor = resultProcessor || defaultResultProcessor;
  return await DBOS.runStep(
    async () => {
      return await runAgent(agent, runInput, session, stream, processor);
    },
    { name: 'runAgent' }
  );
}

async function saveStateStep(
  result: any,
  stateStore: RunStateStore
): Promise<void> {
  /**
   * Step that saves agent state. This must be a step because it accesses the database.
   */
  await DBOS.runStep(
    async () => {
      if (result.state) {
        await stateStore.save(result.state);
      } else if (result.toState) {
        const state = result.toState();
        await stateStore.save(state);
      } else {
        throw new Error('Result does not have state or toState() method');
      }
    },
    { name: 'saveState' }
  );
}

async function executeAgentWithStateHandling(
  agent: Agent,
  inputItems: AgentInputItem[] | RunState<any, any>,
  session: Session,
  stream: boolean,
  resultProcessor: ((result: any) => Promise<any>) | undefined,
  stateStore: RunStateStore
): Promise<any> {
  /** Execute agent and handle state persistence. */
  // Step 1: Run agent (non-deterministic, must be a step)
  const result = await runAgentStep(agent, inputItems, session, stream, resultProcessor);
  
  // Step 2: Handle interruptions and save state if needed
  if (result.interruptions?.length) {
    await saveStateStep(result, stateStore);
  }
  
  return result;
}

async function runAgentWorkflowImpl(
  agent: Agent,
  inputItems: AgentInputItem[] | RunState<any, any>,
  session: Session,
  stream: boolean,
  resultProcessor: ((result: any) => Promise<any>) | undefined,
  stateStore: RunStateStore | null,
  sessionId: string | undefined,
  timeoutSeconds: number | undefined
): Promise<any> {
  /** Internal implementation of run_agent_workflow. */
  // Create state store if not provided
  let store = stateStore;
  if (!store) {
    if (!sessionId) {
      // Try to get session ID from session
      if ('getSessionId' in session && typeof session.getSessionId === 'function') {
        sessionId = await session.getSessionId();
      }
    }
    store = new RunStateStore({ agent, sessionId });
  }
  
  // Execute with timeout if provided
  if (timeoutSeconds) {
    const workflowFn = async () => {
      return await executeAgentWithStateHandling(
        agent, inputItems, session, stream, resultProcessor, store!
      );
    };
    const workflow = DBOS.registerWorkflow(workflowFn);
    const handle = await DBOS.startWorkflow(workflow, { timeoutMS: timeoutSeconds * 1000 })();
    return await handle.getResult();
  } else {
    return await executeAgentWithStateHandling(
      agent, inputItems, session, stream, resultProcessor, store!
    );
  }
}

export async function runAgentWorkflow(
  agent: Agent,
  inputItems: AgentInputItem[] | RunState<any, any>,
  session: Session,
  stream: boolean = false,
  resultProcessor?: (result: any) => Promise<any>,
  stateStore?: RunStateStore,
  sessionId?: string,
  workflowId?: string,
  timeoutSeconds?: number
): Promise<any> {
  /**
   * Run an agent in a durable DBOS workflow.
   * 
   * This workflow automatically saves state on interruptions and can be resumed
   * if the process crashes or restarts.
   */
  await ensureDBOSLaunched();
  
  const workflowFn = async () => {
    return await runAgentWorkflowImpl(
      agent, inputItems, session, stream, resultProcessor,
      stateStore || null, sessionId, timeoutSeconds
    );
  };
  
  const workflow = DBOS.registerWorkflow(workflowFn);
  
  if (workflowId) {
    const handle = await DBOS.startWorkflow(workflow, { workflowID: workflowId })();
    return await handle.getResult();
  } else {
    return await workflow();
  }
}

export function queueAgentWorkflow(
  agent: Agent,
  inputItems: AgentInputItem[] | RunState<any, any>,
  session: Session,
  stream: boolean = false,
  resultProcessor?: (result: any) => Promise<any>,
  stateStore?: RunStateStore,
  sessionId?: string,
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
  return (async () => {
    await ensureDBOSLaunched();
    
    // Get queue
    const queue = queueName ? new WorkflowQueue(queueName) : getDefaultQueue();
    
    // Prepare workflow function
    const workflowFn = async () => {
      return await runAgentWorkflowImpl(
        agent, inputItems, session, stream, resultProcessor,
        stateStore || null, sessionId, timeoutSeconds
      );
    };
    
    const workflow = DBOS.registerWorkflow(workflowFn);
    
    // Enqueue with options
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
    
    const handle = await DBOS.startWorkflow(workflow, startParams)();
    return handle;
  })();
}

export function createScheduledAgentWorkflow(
  crontab: string,
  agent: Agent,
  inputItems: AgentInputItem[] | RunState<any, any>,
  session: Session,
  stream: boolean = false,
  resultProcessor?: (result: any) => Promise<any>,
  stateStore?: RunStateStore,
  sessionId?: string
): void {
  /**
   * Create a scheduled workflow that runs an agent periodically.
   * 
   * This function registers a scheduled workflow with DBOS. The workflow will
   * run automatically according to the crontab schedule.
   * 
   * Example:
   *   createScheduledAgentWorkflow(
   *     '0 0,6,12,18 * * *',  // Every 6 hours
   *     agent,
   *     inputItems,
   *     session
   *   );
   */
  (async () => {
    await ensureDBOSLaunched();
    
    const workflowFn = async (scheduledTime: Date, startTime: Date) => {
      return await runAgentWorkflowImpl(
        agent, inputItems, session, stream, resultProcessor,
        stateStore || null, sessionId, undefined
      );
    };
    
    const workflow = DBOS.registerWorkflow(workflowFn);
    DBOS.registerScheduled(workflow, { crontab });
  })();
}

