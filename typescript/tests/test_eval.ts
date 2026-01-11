/** Tests for Timestep AI Agents SDK - core and eval modules. */

import { describe, it, expect } from 'vitest';
import {
  runEpisode,
  streamEpisode,
  agentBuiltinEcho,
  DEFAULT_TOOLS,
  toolCalc,
  toolEcho,
} from '../timestep/index';
import {
  FinalContains,
  FinalRegex,
  FinalJSON,
  TranscriptContains,
  TranscriptRegex,
  ForbiddenTools,
  MaxToolCalls,
  MinToolCalls,
  ToolCallSequence,
  ToolCallOrder,
  ToolResultJSON,
  OutcomeVerifier,
  parseGraderSpec,
  aggregateGrades,
} from '../timestep/index';
import type { ToolCallRecord } from '../timestep/core/tools';
import type { EpisodeInfo } from '../timestep/core/episode';
import { readJsonl, writeJsonl } from '../timestep/utils/jsonl';
import { ensureTaskId } from '../timestep/utils/messages';
import { mkdtempSync, unlinkSync, rmdirSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';

describe('Agent Harness', () => {
  it('should test builtin echo agent harness', () => {
    const messages = [
      { role: 'user', content: 'Hello' }
    ];
    const result = agentBuiltinEcho(messages, {});
    expect(result.role).toBe('assistant');
    expect(result.content).toContain('Hello');
  });
});

describe('Core Agent-Environment Loop', () => {
  it('should run a simple episode', async () => {
    const messages = [
      { role: 'user', content: 'Hello' }
    ];
    const [resultMessages, info] = await runEpisode(
      messages,
      agentBuiltinEcho,
      DEFAULT_TOOLS,
      undefined,
      { max_steps: 5 },
      { id: 'test' },
      0
    );
    expect(resultMessages.length).toBe(2); // user + assistant
    expect(info.terminated_reason).toBe('final_answer');
    expect(info.steps).toBe(1);
    expect(info.input_tokens).toBe(0); // Echo agent doesn't provide usage
    expect(info.output_tokens).toBe(0);
  });

  it('should track tokens when agent provides usage info', async () => {
    const agentWithUsage = async (messages: any[], context: any) => ({
      role: 'assistant',
      content: 'Hello',
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      }
    });

    const messages = [{ role: 'user', content: 'Hello' }];
    const [, info] = await runEpisode(
      messages,
      agentWithUsage,
      DEFAULT_TOOLS,
      undefined,
      { max_steps: 5 },
      { id: 'test' },
      0
    );
    expect(info.input_tokens).toBe(10);
    expect(info.output_tokens).toBe(5);
    expect(info.total_tokens).toBe(15);
  });
});

describe('Tools', () => {
  it('should execute calc tool', () => {
    const result = toolCalc({ expr: '2+2' });
    expect(result.value).toBe(4);
  });

  it('should execute echo tool', () => {
    const result = toolEcho({ test: 'value' });
    expect(result.echo.test).toBe('value');
  });
});

describe('Graders', () => {
  const createInfo = (): EpisodeInfo => ({
    task_id: 'test',
    trial: 1,
    seed: 0,
    steps: 1,
    tool_calls: 0,
    duration_s: 1.0,
    terminated_reason: 'final_answer',
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    cost_usd: 0.0,
  });

  it('should grade with FinalContains', () => {
    const grader = new FinalContains('Hello');
    const messages = [
      { role: 'user', content: 'Say Hello' },
      { role: 'assistant', content: 'Hello world' }
    ];
    const result = grader.grade(messages, [], {}, createInfo());
    expect(result.passed).toBe(true);
    expect(result.score).toBe(1.0);
  });

  it('should grade with FinalRegex', () => {
    const grader = new FinalRegex('^\\d+$');
    const messages = [
      { role: 'user', content: 'Say a number' },
      { role: 'assistant', content: '123' }
    ];
    const result = grader.grade(messages, [], {}, createInfo());
    expect(result.passed).toBe(true);
  });

  it('should grade with TranscriptContains', () => {
    const grader = new TranscriptContains('Hello');
    const messages = [
      { role: 'user', content: 'Say Hello' },
      { role: 'assistant', content: 'Hi' },
      { role: 'user', content: 'Say Hello again' },
      { role: 'assistant', content: 'Hello' }
    ];
    const result = grader.grade(messages, [], {}, createInfo());
    expect(result.passed).toBe(true);
  });

  it('should grade with TranscriptRegex', () => {
    const grader = new TranscriptRegex('\\d+');
    const messages = [
      { role: 'user', content: 'Say a number' },
      { role: 'assistant', content: 'The answer is 42' }
    ];
    const result = grader.grade(messages, [], {}, createInfo());
    expect(result.passed).toBe(true);
  });

  it('should grade with MinToolCalls', () => {
    const grader = new MinToolCalls(2);
    const messages = [
      { role: 'user', content: 'Use calc twice' },
      { role: 'assistant', content: '', tool_calls: [{ id: '1', function: { name: 'calc', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: '1', content: '{}' },
      { role: 'assistant', content: '', tool_calls: [{ id: '2', function: { name: 'calc', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: '2', content: '{}' },
      { role: 'assistant', content: 'Done' }
    ];
    const toolIndex: ToolCallRecord[] = [
      { tool_call_id: '1', name: 'calc', arguments_raw: '{}', arguments: {}, result_raw: '{}', result: {} },
      { tool_call_id: '2', name: 'calc', arguments_raw: '{}', arguments: {}, result_raw: '{}', result: {} },
    ];
    const info = { ...createInfo(), tool_calls: 2 };
    const result = grader.grade(messages, toolIndex, {}, info);
    expect(result.passed).toBe(true);
  });

  it('should grade with ToolCallOrder', () => {
    const grader = new ToolCallOrder(['calc', 'echo']);
    const messages = [
      { role: 'user', content: 'Use calc then echo' },
      { role: 'assistant', content: '', tool_calls: [{ id: '1', function: { name: 'calc', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: '1', content: '{}' },
      { role: 'assistant', content: '', tool_calls: [{ id: '2', function: { name: 'echo', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: '2', content: '{}' },
      { role: 'assistant', content: 'Done' }
    ];
    const toolIndex: ToolCallRecord[] = [
      { tool_call_id: '1', name: 'calc', arguments_raw: '{}', arguments: {}, result_raw: '{}', result: {} },
      { tool_call_id: '2', name: 'echo', arguments_raw: '{}', arguments: {}, result_raw: '{}', result: {} },
    ];
    const info = { ...createInfo(), tool_calls: 2 };
    const result = grader.grade(messages, toolIndex, {}, info);
    expect(result.passed).toBe(true);
  });

  it('should grade with OutcomeVerifier', () => {
    const verifier = (messages: any[], toolIndex: ToolCallRecord[], task: any) => {
      return toolIndex.some(r => r.name === 'calc');
    };
    const grader = new OutcomeVerifier(verifier);
    const messages = [
      { role: 'user', content: 'Use calc' },
      { role: 'assistant', content: '', tool_calls: [{ id: '1', function: { name: 'calc', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: '1', content: '{}' },
      { role: 'assistant', content: 'Done' }
    ];
    const toolIndex: ToolCallRecord[] = [
      { tool_call_id: '1', name: 'calc', arguments_raw: '{}', arguments: {}, result_raw: '{}', result: {} },
    ];
    const result = grader.grade(messages, toolIndex, {}, createInfo());
    expect(result.passed).toBe(true);
  });
});

describe('Grader Spec Parsing', () => {
  it('should parse grader specs', () => {
    const grader = parseGraderSpec('FinalContains:Hello');
    expect(grader).toBeInstanceOf(FinalContains);

    const grader2 = parseGraderSpec('MinToolCalls:2');
    expect(grader2).toBeInstanceOf(MinToolCalls);

    const grader3 = parseGraderSpec('ToolCallOrder:calc,echo');
    expect(grader3).toBeInstanceOf(ToolCallOrder);
  });
});

describe('Grade Aggregation', () => {
  it('should aggregate grades', () => {
    const grades = [
      { name: 'Test1', passed: true, score: 1.0 },
      { name: 'Test2', passed: false, score: 0.5 },
    ];
    const result = aggregateGrades(grades);
    expect(result.passed).toBe(false);
    expect(result.score).toBe(0.75);
  });
});

describe('JSONL I/O', () => {
  it('should read and write JSONL files', () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'timestep-test-'));
    const path = join(tmpDir, 'test.jsonl');
    
    try {
      const data = [{ id: 1, name: 'test' }, { id: 2, name: 'test2' }];
      writeJsonl(path, data);
      
      const readData = Array.from(readJsonl(path));
      expect(readData.length).toBe(2);
      expect(readData[0].id).toBe(1);
    } finally {
      try {
        unlinkSync(path);
        rmdirSync(tmpDir);
      } catch {
        // Ignore cleanup errors
      }
    }
  });
});

describe('Task ID Generation', () => {
  it('should generate stable task IDs', () => {
    const task: any = { messages: [{ role: 'user', content: 'test' }] };
    const taskId = ensureTaskId(task);
    expect('id' in task).toBe(true);
    expect(taskId).toBe(task.id);
    
    // Should be stable
    const task2: any = { messages: [{ role: 'user', content: 'test' }] };
    const taskId2 = ensureTaskId(task2);
    expect(taskId).toBe(taskId2);
  });
});

describe('Streaming Episode Runner', () => {
  it('should stream events with non-streaming agent', async () => {
    const messages = [
      { role: 'user', content: 'Hello' }
    ];
    
    const events: any[] = [];
    for await (const event of streamEpisode(
      messages,
      agentBuiltinEcho,
      DEFAULT_TOOLS,
      undefined,
      { max_steps: 5 },
      { id: 'test' },
      0
    )) {
      events.push(event);
    }
    
    // Should have RunStarted, StepStarted, TextMessageStart/Content/End, StepFinished, RunFinished
    expect(events.length).toBeGreaterThanOrEqual(5);
    expect(events[0].type).toBe('RunStarted');
    expect(events[events.length - 1].type).toBe('RunFinished');
    
    // Check RunFinished has correct structure
    const finalEvent = events[events.length - 1];
    expect(finalEvent.result).toBeDefined();
    expect(finalEvent.result.transcript).toBeDefined();
    expect(finalEvent.result.episodeInfo).toBeDefined();
    expect(finalEvent.result.episodeInfo.terminated_reason).toBe('final_answer');
  });

  it('should stream chunks with streaming agent', async () => {
    async function* simpleStreamingAgent(messages: any[], context: any) {
      const chunks = ['Hello', ' ', 'world', '!'];
      for (const chunk of chunks) {
        yield { type: 'content', delta: chunk };
      }
      yield { type: 'done' };
    }
    
    const messages = [
      { role: 'user', content: 'Say hello' }
    ];
    
    const events: any[] = [];
    const contentChunks: string[] = [];
    for await (const event of streamEpisode(
      messages,
      simpleStreamingAgent,
      DEFAULT_TOOLS,
      undefined,
      { max_steps: 5 },
      { id: 'test' },
      0
    )) {
      events.push(event);
      if (event.type === 'TextMessageContent') {
        contentChunks.push(event.delta);
      }
    }
    
    // Should have received content chunks
    expect(contentChunks.length).toBe(4);
    expect(contentChunks.join('')).toBe('Hello world!');
    
    // Should have TextMessageEnd
    const messageEnd = events.find((e: any) => e.type === 'TextMessageEnd');
    expect(messageEnd).toBeDefined();
    
    // Should end with RunFinished
    expect(events[events.length - 1].type).toBe('RunFinished');
  });

  it('should stream events with tool calls', async () => {
    const agentWithTool = async (messages: any[], context: any) => ({
      role: 'assistant',
      content: '',
      tool_calls: [{
        id: 'call_1',
        type: 'function',
        function: {
          name: 'calc',
          arguments: '{"expr": "2+2"}'
        }
      }]
    });
    
    const messages = [
      { role: 'user', content: 'Calculate 2+2' }
    ];
    
    const events: any[] = [];
    const toolCallEvents: any[] = [];
    for await (const event of streamEpisode(
      messages,
      agentWithTool,
      DEFAULT_TOOLS,
      ['calc'],
      { max_steps: 5 },
      { id: 'test' },
      0
    )) {
      events.push(event);
      if (event.type === 'ToolCallStart' || event.type === 'ToolCallResult') {
        toolCallEvents.push(event);
      }
    }
    
    // Should have tool call events
    expect(toolCallEvents.length).toBeGreaterThanOrEqual(2);
    expect(toolCallEvents[0].type).toBe('ToolCallStart');
    const toolCallResult = toolCallEvents.find((e: any) => e.type === 'ToolCallResult');
    expect(toolCallResult).toBeDefined();
    expect(toolCallResult!.result.value).toBe(4);
  });

  it('should stream tool call chunks from streaming agent', async () => {
    async function* streamingAgentWithTool(messages: any[], context: any) {
      // Yield tool call chunks
      yield { type: 'tool_call', delta: { id: 'call_1', function: { name: 'calc' } } };
      yield { type: 'tool_call', delta: { id: 'call_1', function: { arguments: '{"expr":' } } };
      yield { type: 'tool_call', delta: { id: 'call_1', function: { arguments: ' "3+3"' } } };
      yield { type: 'tool_call', delta: { id: 'call_1', function: { arguments: '}' } } };
      yield { type: 'done' };
    }
    
    const messages = [
      { role: 'user', content: 'Calculate 3+3' }
    ];
    
    const events: any[] = [];
    const toolCallDeltas: any[] = [];
    for await (const event of streamEpisode(
      messages,
      streamingAgentWithTool,
      DEFAULT_TOOLS,
      ['calc'],
      { max_steps: 5 },
      { id: 'test' },
      0
    )) {
      events.push(event);
      if (event.type === 'ToolCallChunk') {
        toolCallDeltas.push(event.chunk);
      }
    }
    
    // Should have received tool call chunks
    expect(toolCallDeltas.length).toBeGreaterThan(0);
    
    // Should have ToolCallStart
    const toolCallStart = events.find((e: any) => e.type === 'ToolCallStart');
    expect(toolCallStart).toBeDefined();
    expect(toolCallStart!.name).toBe('calc');
  });

  it('should handle streaming agent errors', async () => {
    async function* errorStreamingAgent(messages: any[], context: any) {
      yield { type: 'error', error: 'test_error' };
    }
    
    const messages = [
      { role: 'user', content: 'Test' }
    ];
    
    const events: any[] = [];
    for await (const event of streamEpisode(
      messages,
      errorStreamingAgent,
      DEFAULT_TOOLS,
      undefined,
      { max_steps: 5 },
      { id: 'test' },
      0
    )) {
      events.push(event);
    }
    
    // Should have RunError event
    const errorEvents = events.filter((e: any) => e.type === 'RunError');
    expect(errorEvents.length).toBeGreaterThan(0);
    expect(errorEvents[0].message).toBe('test_error');
    
    // Should end with RunFinished with error
    expect(events[events.length - 1].type).toBe('RunFinished');
    expect(events[events.length - 1].result.episodeInfo.error).toBeDefined();
  });
});
