/** Integration tests for A2A functionality. */
/* eslint-disable @typescript-eslint/unbound-method */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { RequestContext, ExecutionEventBus } from '@a2a-js/sdk/server';
import { runAgent } from '../timestep/core/agent.js';

// Mock the pg module (native module that vitest can't load)
vi.mock('pg', () => ({
  default: {
    Pool: vi.fn().mockImplementation(() => ({
      on: vi.fn(),
      connect: vi.fn(),
      query: vi.fn(),
      end: vi.fn(),
    })),
  },
  Pool: vi.fn().mockImplementation(() => ({
    on: vi.fn(),
    connect: vi.fn(),
    query: vi.fn(),
    end: vi.fn(),
  })),
}));

// Mock the agent module
vi.mock('../timestep/core/agent.js', () => ({
  runAgent: vi.fn(),
}));

import { TimestepAgentExecutor } from '../timestep/a2a/agent_executor.js';
import { createAgentCard, createServer } from '../timestep/a2a/server.js';
import type { Agent } from '../timestep/a2a/postgres_agent_store.js';

describe('A2A Integration Tests', () => {
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
        parts: [{ kind: 'text', text: "What's 2+2?" }],
        contextId: 'test-context-id',
      },
      task: undefined,
    } as RequestContext;
  });

  it('should execute full flow from message to response', async () => {
    /** Test the full execution flow. */
    vi.mocked(runAgent).mockResolvedValue('2 + 2 = 4');

    await executor.execute(mockContext, mockEventBus);

    // Verify runAgent was called
    expect(runAgent).toHaveBeenCalled();
    
    // Verify multiple events were published
    const callCount = vi.mocked(mockEventBus.publish).mock.calls.length;
    expect(callCount).toBeGreaterThanOrEqual(3);
    
    // Verify final event is completed
    const calls = vi.mocked(mockEventBus.publish).mock.calls;
    const lastCall = calls[calls.length - 1];
    const lastEvent = lastCall?.[0] as { kind?: string; status?: { state?: string }; final?: boolean };
    expect(lastEvent?.kind).toBe('status-update');
    expect(lastEvent?.status?.state).toBe('completed');
    expect(lastEvent?.final).toBe(true);
  });

  it('should create agent card with all required fields', () => {
    /** Test that agent card has all required fields. */
    const testAgent: Agent = {
      id: 'test-agent',
      name: 'Test Agent',
      description: 'Test agent for testing',
      tools: ['get_weather', 'web_search'],
      model: 'gpt-4.1',
      created_at: new Date(),
      updated_at: new Date(),
    };
    const card = createAgentCard(testAgent);
    
    expect(card.name).toBeDefined();
    expect(card.url).toBeDefined();
    expect(card.version).toBeDefined();
    expect(card.defaultInputModes).toBeDefined();
    expect(card.defaultOutputModes).toBeDefined();
    expect(card.capabilities).toBeDefined();
    expect(card.skills).toBeDefined();
    expect(card.skills?.length).toBeGreaterThan(0);
    
    // Each skill should have required fields
    card.skills?.forEach(skill => {
      expect(skill.id).toBeDefined();
      expect(skill.name).toBeDefined();
      expect(skill.description).toBeDefined();
    });
  });

  it('should create server without errors', () => {
    /** Test that server can be created without errors. */
    const app = createServer('127.0.0.1', 8080);
    
    expect(app).toBeDefined();
    // Express app should have listen method
    expect(typeof app.listen).toBe('function');
  });
});

