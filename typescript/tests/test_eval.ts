/** Tests for Timestep AI Agents SDK - core and eval modules. */

import { describe, it, expect } from 'vitest';
import {
  runEpisode,
  agentBuiltinEcho,
  DEFAULT_TOOLS,
  toolCalc,
  toolEcho,
} from '../timestep/index.js';
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
} from '../timestep/index.js';
import type { ToolCallRecord, EpisodeInfo } from '../timestep/core/episode.js';

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
    expect((grader as FinalContains).substring).toBe('Hello');

    const grader2 = parseGraderSpec('MinToolCalls:2');
    expect(grader2).toBeInstanceOf(MinToolCalls);
    expect((grader2 as MinToolCalls).minCalls).toBe(2);

    const grader3 = parseGraderSpec('ToolCallOrder:calc,echo');
    expect(grader3).toBeInstanceOf(ToolCallOrder);
    expect((grader3 as ToolCallOrder).expectedSequence).toEqual(['calc', 'echo']);
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
