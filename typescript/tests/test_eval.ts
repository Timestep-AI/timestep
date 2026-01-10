/** Tests for eval framework. */

import { describe, it, expect } from 'vitest';
import { writeFileSync, unlinkSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  runEpisode,
  agentBuiltinEcho,
  DEFAULT_TOOLS,
  toolCalc,
  toolEcho,
} from '../timestep/eval/index.js';
import {
  FinalContains,
  FinalRegex,
  ForbiddenTools,
  MaxToolCalls,
  ToolCallSequence,
  parseGraderSpec,
  aggregateGrades,
} from '../timestep/eval/graders.js';
import { readJsonl, writeJsonl, ensureTaskId } from '../timestep/utils/index.js';

describe('Agent', () => {
  it('should echo user message', () => {
    const messages = [
      { role: 'user', content: 'Hello' }
    ];
    const result = agentBuiltinEcho(messages, {});
    expect(result.role).toBe('assistant');
    expect(result.content).toContain('Hello');
  });
});

describe('Tools', () => {
  it('should calculate expression', () => {
    const result = toolCalc({ expr: '2+2' });
    expect(result.value).toBe(4);
  });

  it('should echo arguments', () => {
    const result = toolEcho({ test: 'value' });
    expect(result.echo.test).toBe('value');
  });
});

describe('Episode', () => {
  it('should run simple episode', () => {
    const messages = [
      { role: 'user', content: 'Hello' }
    ];
    const [resultMessages, info] = runEpisode(
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
  });
});

describe('Graders', () => {
  it('should check final contains', () => {
    const grader = new FinalContains('Hello');
    const messages = [
      { role: 'user', content: 'Say Hello' },
      { role: 'assistant', content: 'Hello world' }
    ];
    const result = grader.grade(messages, [], {}, {} as any);
    expect(result.passed).toBe(true);
    expect(result.score).toBe(1.0);
  });

  it('should check final regex', () => {
    const grader = new FinalRegex('^\\d+$');
    const messages = [
      { role: 'user', content: 'Say a number' },
      { role: 'assistant', content: '123' }
    ];
    const result = grader.grade(messages, [], {}, {} as any);
    expect(result.passed).toBe(true);
  });

  it('should parse grader spec', () => {
    const grader = parseGraderSpec('FinalContains:Hello');
    expect(grader).toBeInstanceOf(FinalContains);
    expect((grader as FinalContains).substring).toBe('Hello');
  });

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

describe('Utils', () => {
  it('should read and write JSONL', () => {
    const path = join(tmpdir(), `test_${Date.now()}.jsonl`);
    const data = [
      { id: 1, name: 'test' },
      { id: 2, name: 'test2' }
    ];
    
    try {
      writeJsonl(path, data);
      const readData = Array.from(readJsonl(path));
      expect(readData.length).toBe(2);
      expect(readData[0].id).toBe(1);
    } finally {
      unlinkSync(path);
    }
  });

  it('should ensure task ID', () => {
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
