/** Tests for A2A server setup. */

import { describe, it, expect, vi } from 'vitest';

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

import { createAgentCard, createServer } from '../timestep/a2a/server.js';
import { GetWeatherParameters } from '../timestep/core/tools.js';
import type { Agent } from '../timestep/a2a/postgres_agent_store.js';

describe('A2A Server Tests', () => {
  it('should create agent card with correct properties', () => {
    /** Test creating AgentCard. */
    const url = 'http://localhost:8080/';
    const testAgent: Agent = {
      id: 'test-agent',
      name: 'Test Agent',
      description: 'Test agent',
      tools: ['get_weather', 'web_search'],
      model: 'gpt-4.1',
      created_at: new Date(),
      updated_at: new Date(),
    };
    const card = createAgentCard(testAgent, url);

    expect(card.name).toBe('Test Agent');
    expect(card.url).toBe(url);
    expect(card.version).toBe('2026.0.5');
    expect(card.defaultInputModes).toEqual(['text']);
    expect(card.defaultOutputModes).toContain('text');
    expect(card.defaultOutputModes).toContain('task-status');
    expect(card.capabilities?.streaming).toBe(true);
    expect(card.skills).toBeDefined();
    expect(card.skills?.length).toBe(2);
    // Examples may not be a direct attribute, check if it exists
    const cardWithExamples = card as unknown as { examples?: string[] };
    if (cardWithExamples.examples) {
      expect(cardWithExamples.examples.length).toBeGreaterThan(0);
    }
  });

  it('should create agent card with skills', () => {
    /** Test that agent card includes skills. */
    const testAgent: Agent = {
      id: 'test-agent',
      name: 'Test Agent',
      description: 'Test agent',
      tools: ['get_weather', 'web_search'],
      model: 'gpt-4.1',
      created_at: new Date(),
      updated_at: new Date(),
    };
    const card = createAgentCard(testAgent);
    
    expect(card.skills).toBeDefined();
    expect(card.skills?.length).toBe(2);
    
    const weatherSkill = card.skills?.find(s => s.id === 'get_weather');
    expect(weatherSkill).toBeDefined();
    expect(weatherSkill?.name).toBe('Get Weather');
    
    const searchSkill = card.skills?.find(s => s.id === 'web_search');
    expect(searchSkill).toBeDefined();
    expect(searchSkill?.name).toBe('Web Search');
  });

  it('should create server application', () => {
    /** Test creating A2A server. */
    const app = createServer('127.0.0.1', 8080);
    
    expect(app).toBeDefined();
    // Express app should have methods like use, listen, etc.
    expect(typeof app.listen).toBe('function');
  });

  it('should create server with custom tools', () => {
    /** Test creating server with custom tools. */
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const tools = [
      { name: 'get_weather', parameters: GetWeatherParameters },
    ];
    const app = createServer('127.0.0.1', 8080, tools);
    expect(app).toBeDefined();
  });

  it('should create server with custom model', () => {
    /** Test creating server with custom model. */
    const app = createServer('127.0.0.1', 8080, undefined, 'gpt-4');
    expect(app).toBeDefined();
  });

  it('should create agent card with default URL', () => {
    /** Test creating agent card with default URL. */
    const testAgent: Agent = {
      id: 'test-agent',
      name: 'Test Agent',
      description: 'Test agent',
      tools: ['get_weather'],
      model: 'gpt-4.1',
      created_at: new Date(),
      updated_at: new Date(),
    };
    const card = createAgentCard(testAgent);
    expect(card.url).toBe('http://localhost:8080/');
  });
});

