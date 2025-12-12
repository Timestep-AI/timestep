/** Tests for basic agent functionality. */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';

// Mock the MCP client imports to avoid actual network calls
vi.mock('@modelcontextprotocol/sdk/client/index.js', () => ({
  Client: vi.fn().mockImplementation(() => ({
    connect: vi.fn().mockResolvedValue(undefined),
    request: vi.fn().mockResolvedValue({
      content: [{ type: 'text', text: 'The weather in Oakland is sunny' }],
    }),
    close: vi.fn().mockResolvedValue(undefined),
  })),
}));

vi.mock('@modelcontextprotocol/sdk/client/streamableHttp.js', () => ({
  StreamableHTTPClientTransport: vi.fn().mockImplementation(() => ({})),
}));

import { runAgent, GetWeatherParameters } from '../timestep/index';

describe('Agent Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should respond to basic queries without tools', async () => {
    /** Test basic agent response without tools. */
    const messages: ChatCompletionMessageParam[] = [
      { role: 'system', content: 'You are a helpful AI assistant.' },
      { role: 'user', content: "What's 2+2?" },
    ];

    const response = await runAgent(messages);

    expect(response).not.toBeNull();
    expect(response.length).toBeGreaterThan(0);
    expect(response).toContain('4');
  });

  it('should use tools when provided', async () => {
    /** Test agent using a tool. */
    const messages: ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content: 'You are a helpful AI assistant that can answer questions about weather. When asked about weather, you MUST use the getWeather tool.',
      },
      { role: 'user', content: "What's the weather in Oakland?" },
    ];

    const response = await runAgent(
      messages, 
      [{ name: 'get_weather', parameters: GetWeatherParameters }],
    );

    expect(response).not.toBeNull();
    expect(response.length).toBeGreaterThan(0);
    expect(
      response.includes('Oakland') || response.toLowerCase().includes('weather')
    ).toBe(true);
  });

  it('should maintain conversation context', async () => {
    /** Test agent maintaining conversation context. */
    const messages: ChatCompletionMessageParam[] = [
      { role: 'system', content: 'You are a helpful AI assistant.' },
      { role: 'user', content: "What's 2+2?" },
    ];

    // First message (runAgent will append the assistant response to messages)
    const response1 = await runAgent(messages);
    expect(response1).toContain('4');

    // Follow-up message with history (should remember)
    // Note: runAgent already appended the assistant message, so we just add the user message
    messages.push({ role: 'user', content: "What's three times that number?" });
    const response2 = await runAgent(messages);

    expect(response2).not.toBeNull();
    expect(
      response2.includes('12') || response2.toLowerCase().includes('three')
    ).toBe(true);
  });
});
