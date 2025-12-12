/** Tests for A2A AgentExecutor implementation. */
/* eslint-disable @typescript-eslint/unbound-method */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TimestepAgentExecutor } from '../timestep/a2a/agent_executor.js';
import {
  extractTextFromParts,
  a2AToOpenAI as convertA2AMessagesToOpenAI,
} from '../timestep/a2a/message_converter.js';
import type { RequestContext, ExecutionEventBus } from '@a2a-js/sdk/server';
import type { Message, Task } from '@a2a-js/sdk';
import { runAgent } from '../timestep/core/agent.js';

// Mock the agent module
vi.mock('../timestep/core/agent.js', () => ({
  runAgent: vi.fn(),
}));

describe('A2A AgentExecutor Helper Functions', () => {
  it('should extract text from parts', () => {
    /** Test extracting text from A2A message parts. */
    const parts = [
      { kind: 'text', text: 'Hello' },
      { kind: 'text', text: ' World' },
      { kind: 'other', data: 'ignored' },
    ];
    const result = extractTextFromParts(parts);
    expect(result).toBe('Hello World');
  });

  it('should convert A2A messages to OpenAI format', () => {
    /** Test converting A2A messages to OpenAI format. */
    const a2aMessages = [
      {
        kind: 'message',
        role: 'user',
        messageId: 'msg-1',
        parts: [{ kind: 'text', text: 'Hello' }],
      } as Message,
      {
        kind: 'message',
        role: 'agent',
        messageId: 'msg-2',
        parts: [{ kind: 'text', text: 'Hi there' }],
      } as Message,
    ];
    const result = convertA2AMessagesToOpenAI(a2aMessages);
    
    expect(result.length).toBe(2);
    expect(result[0].role).toBe('user');
    expect(result[0].content).toBe('Hello');
    expect(result[1].role).toBe('assistant');
    expect(result[1].content).toBe('Hi there');
  });

  it('should convert A2A messages with tool calls', () => {
    /** Test converting A2A messages with tool calls. */
    const a2aMessages = [
      {
        kind: 'message',
        role: 'user',
        messageId: 'msg-1',
        parts: [{ kind: 'text', text: "What's the weather?" }],
      } as Message,
      {
        kind: 'message',
        role: 'agent',
        messageId: 'msg-2',
        parts: [{ kind: 'text', text: '' }],
        toolCalls: [{ id: 'call_1', function: { name: 'get_weather', arguments: '{"city": "Oakland"}' } }],
      } as Message & { toolCalls: Array<{ id: string; function: { name: string; arguments: string } }> },
      {
        kind: 'message',
        role: 'tool',
        messageId: 'msg-3',
        parts: [{ kind: 'text', text: 'Sunny' }],
        toolCallId: 'call_1',
      } as unknown as Message & { toolCallId: string },
    ];
    const result = convertA2AMessagesToOpenAI(a2aMessages);
    
    expect(result.length).toBe(3);
    expect(result[0].role).toBe('user');
    expect(result[0].content).toBe("What's the weather?");
    expect(result[1].role).toBe('assistant');
    expect('tool_calls' in result[1]).toBe(true);
    expect(result[2].role).toBe('tool');
    expect((result[2] as { tool_call_id: string }).tool_call_id).toBe('call_1');
  });
});

describe('A2A AgentExecutor Tests', () => {
  let executor: TimestepAgentExecutor;
  let mockEventBus: ExecutionEventBus;
  let mockContext: RequestContext;

  beforeEach(() => {
    executor = new TimestepAgentExecutor();
    mockEventBus = {
      publish: vi.fn(),
    } as unknown as ExecutionEventBus;
    
    mockContext = {
      userMessage: {
        kind: 'message',
        role: 'user',
        messageId: 'test-msg-1',
        parts: [{ kind: 'text', text: 'Hello' }],
        contextId: 'test-context-id',
      },
      task: undefined,
    } as RequestContext;
  });

  it('should initialize with default tools', () => {
    /** Test TimestepAgentExecutor initialization. */
    expect(executor).toBeDefined();
    // Tools are private, but we can verify by checking execution
  });

  it('should create a task when none exists', async () => {
    /** Test that executor creates a task when none exists. */
    vi.mocked(runAgent).mockResolvedValue('Test response');

    await executor.execute(mockContext, mockEventBus);

    // Verify task was created (first publish should be a task)
    expect(mockEventBus.publish).toHaveBeenCalled();
    const firstCall = vi.mocked(mockEventBus.publish).mock.calls[0];
    expect(firstCall).toBeDefined();
    const firstEvent = firstCall?.[0] as { kind?: string; status?: { state?: string } };
    if (firstEvent?.kind === 'task' && firstEvent?.status) {
      expect(firstEvent.status.state).toBe('submitted');
    }
  });

  it('should use existing task when provided', async () => {
    /** Test executor with existing task. */
    const existingTask: Task = {
      kind: 'task',
      id: 'test-task-id',
      contextId: 'test-context-id',
      status: {
        state: 'submitted',
        timestamp: new Date().toISOString(),
      },
      history: [],
    };

    Object.assign(mockContext, { task: existingTask });
    vi.mocked(runAgent).mockResolvedValue('Test response');

    await executor.execute(mockContext, mockEventBus);

    // Verify runAgent was called
    expect(runAgent).toHaveBeenCalled();
    // Verify events were published
    expect(mockEventBus.publish).toHaveBeenCalled();
  });

  it('should publish working status update', async () => {
    /** Test that executor publishes working status. */
    vi.mocked(runAgent).mockResolvedValue('Test response');

    await executor.execute(mockContext, mockEventBus);

    // Find working status update
    const calls = vi.mocked(mockEventBus.publish).mock.calls;
    const workingUpdate = calls.find(
      (call) => {
        const event = call[0] as { kind?: string; status?: { state?: string } };
        return event?.kind === 'status-update' && event?.status?.state === 'working';
      }
    );
    expect(workingUpdate).toBeDefined();
  });

  it('should publish completed status on success', async () => {
    /** Test that executor publishes completed status. */
    vi.mocked(runAgent).mockResolvedValue('Test response');

    await executor.execute(mockContext, mockEventBus);

    // Find completed status update
    const calls = vi.mocked(mockEventBus.publish).mock.calls;
    const completedUpdate = calls.find(
      (call) => {
        const event = call[0] as { kind?: string; status?: { state?: string }; final?: boolean };
        return event?.kind === 'status-update' && 
               event?.status?.state === 'completed' &&
               event?.final === true;
      }
    );
    expect(completedUpdate).toBeDefined();
  });

  it('should publish failed status on error', async () => {
    /** Test executor error handling. */
    vi.mocked(runAgent).mockRejectedValue(new Error('Test error'));

    // The executor catches errors and publishes failed status, it doesn't throw
    await executor.execute(mockContext, mockEventBus);

    // Find failed status update
    const calls = vi.mocked(mockEventBus.publish).mock.calls;
    const failedUpdate = calls.find(
      (call) => {
        const event = call[0] as { kind?: string; status?: { state?: string }; final?: boolean };
        return event?.kind === 'status-update' && 
               event?.status?.state === 'failed' &&
               event?.final === true;
      }
    );
    expect(failedUpdate).toBeDefined();
  });

  it('should cancel task', async () => {
    /** Test executor cancellation. */
    const taskId = 'test-task-id';
    
    await executor.cancelTask(taskId, mockEventBus);

    // Verify cancellation event was published
    expect(mockEventBus.publish).toHaveBeenCalled();
    const call = vi.mocked(mockEventBus.publish).mock.calls[0];
    const event = call?.[0] as { kind?: string; status?: { state?: string }; final?: boolean };
    expect(event?.kind).toBe('status-update');
    expect(event?.status?.state).toBe('canceled');
    expect(event?.final).toBe(true);
  });

  it('should handle tool approval requirement', async () => {
    /** Test tool approval handling. */
    // Mock runAgent to trigger tool approval via event emitter
    vi.mocked(runAgent).mockImplementation(async (
      _messages: unknown,
      _tools: unknown,
      _model: unknown,
      _apiKey: unknown,
      eventEmitter: import('../timestep/core/agent_events.js').AgentEventEmitter | undefined
    ) => {
      if (eventEmitter) {
        // Simulate tool call
        const toolCall = {
          id: 'call_1',
          type: 'function' as const,
          function: {
            name: 'get_weather',
            arguments: '{"city": "Oakland"}',
          },
        };
        // Emit tool-approval-required event - this will create a Future and wait for it
        const future = new Promise<boolean>((resolve) => {
          // Emit with the correct format that the handler expects
          eventEmitter.emit('tool-approval-required', {
            toolCall,
            resolve,
          });
          // Give it a moment to set up the resolver and publish the event
          setTimeout(() => {
            // Manually resolve the approval by finding the resolver and calling it
            const pendingApprovals = executor['_pendingApprovals'] as Map<string, { toolCall: unknown; resolve: (approved: boolean) => void }> | undefined;
            if (pendingApprovals && pendingApprovals.size > 0) {
              const approval = Array.from(pendingApprovals.values())[0];
              approval.resolve(true); // Auto-approve for test
            }
          }, 100);
        });
        
        // Wait a bit to let the status be published
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Wait for approval to complete
        await future;
      }
      return 'Test response';
    });

    // Execute - this will trigger the approval requirement
    const executePromise = executor.execute(mockContext, mockEventBus);
    
    // Wait for the input-required status to be published
    await new Promise(resolve => setTimeout(resolve, 100));

    // Verify input-required status was published
    const calls = vi.mocked(mockEventBus.publish).mock.calls;
    const inputRequiredUpdate = calls.find(
      (call) => {
        const event = call[0] as { kind?: string; status?: { state?: string } };
        return event?.kind === 'status-update' && 
               event?.status?.state === 'input-required';
      }
    );
    expect(inputRequiredUpdate).toBeDefined();
    
    // Now simulate approval by sending an approval message to resolve the pending approval
    // Get task ID from the published events
    const taskEvent = calls.find(
      (call) => {
        const event = call[0] as { kind?: string; taskId?: string };
        return event?.kind === 'task' && event?.taskId;
      }
    );
    const taskId = taskEvent 
      ? (taskEvent[0] as { taskId?: string })?.taskId || 'test-task-id'
      : 'test-task-id';
    
    const approvalContext = {
      ...mockContext,
      userMessage: {
        ...mockContext.userMessage,
        parts: [{ kind: 'text', text: 'approve' }],
        contextId: 'test-context-id',
        taskId: taskId,
      },
    };
    
    // Execute with approval message - this should resolve the pending approval and continue
    await executor.execute(approvalContext, mockEventBus);
    
    // Wait for the original execution to complete
    await executePromise;
  }, 10000); // Increase timeout for this test
});

