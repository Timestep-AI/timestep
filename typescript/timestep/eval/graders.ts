/** Built-in graders for evaluation. */

import type { Message } from '../utils/messages';
import { lastAssistantContent } from '../utils/messages';
import { clamp01 } from '../utils/io';
import type { EpisodeInfo } from '../core/episode';
import type { ToolCallRecord } from '../core/tools';
import type { JSON } from '../core/types';

export abstract class Grader {
  name: string = 'Grader';

  abstract grade(
    messages: Message[],
    toolIndex: ToolCallRecord[],
    task: JSON,
    info: EpisodeInfo
  ): JSON | Promise<JSON>;
}

// Code-based graders

export class FinalRegex extends Grader {
  name = 'FinalRegex';
  private pattern?: string;
  private fromExpectedKey: string;

  constructor(pattern?: string, fromExpectedKey: string = 'final_regex') {
    super();
    this.pattern = pattern;
    this.fromExpectedKey = fromExpectedKey;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const pat = this.pattern || (task.expected || {})[this.fromExpectedKey];
    if (!pat) {
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
    }
    const text = lastAssistantContent(messages);
    const regex = new RegExp(pat, 'm');
    const ok = regex.test(text);
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { pattern: pat } };
  }
}

export class FinalContains extends Grader {
  name = 'FinalContains';
  private substring?: string;
  private fromExpectedKey: string;

  constructor(substring?: string, fromExpectedKey: string = 'final_contains') {
    super();
    this.substring = substring;
    this.fromExpectedKey = fromExpectedKey;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const sub = this.substring || (task.expected || {})[this.fromExpectedKey];
    if (!sub) {
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
    }
    const text = lastAssistantContent(messages);
    const ok = text.includes(String(sub));
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { substring: sub } };
  }
}

export class FinalJSON extends Grader {
  name = 'FinalJSON';
  private requiredKeys?: string[];
  private fromExpectedKey: string;

  constructor(requiredKeys?: string[], fromExpectedKey: string = 'final_json_required_keys') {
    super();
    this.requiredKeys = requiredKeys;
    this.fromExpectedKey = fromExpectedKey;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const keys = this.requiredKeys || (task.expected || {})[this.fromExpectedKey];
    if (!keys) {
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
    }
    const text = lastAssistantContent(messages);
    let obj: any;
    try {
      obj = JSON.parse(text);
    } catch (e: any) {
      return { name: this.name, passed: false, score: 0.0, details: { error: 'invalid_json', exception: String(e) } };
    }
    const missing = (keys as string[]).filter(k => !(k in obj));
    const ok = missing.length === 0;
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { missing } };
  }
}

export class TranscriptContains extends Grader {
  name = 'TranscriptContains';
  private substring?: string;
  private fromExpectedKey: string;

  constructor(substring?: string, fromExpectedKey: string = 'transcript_contains') {
    super();
    this.substring = substring;
    this.fromExpectedKey = fromExpectedKey;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const sub = this.substring || (task.expected || {})[this.fromExpectedKey];
    if (!sub) {
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
    }
    // Check all messages, not just final
    const transcriptText = messages.map(m => String(m.content || '')).join(' ');
    const ok = transcriptText.includes(String(sub));
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { substring: sub } };
  }
}

export class TranscriptRegex extends Grader {
  name = 'TranscriptRegex';
  private pattern?: string;
  private fromExpectedKey: string;

  constructor(pattern?: string, fromExpectedKey: string = 'transcript_regex') {
    super();
    this.pattern = pattern;
    this.fromExpectedKey = fromExpectedKey;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const pat = this.pattern || (task.expected || {})[this.fromExpectedKey];
    if (!pat) {
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
    }
    // Check all messages, not just final
    const transcriptText = messages.map(m => String(m.content || '')).join(' ');
    const regex = new RegExp(pat, 'm');
    const ok = regex.test(transcriptText);
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { pattern: pat } };
  }
}

export class ForbiddenTools extends Grader {
  name = 'ForbiddenTools';

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const allowed = task.tools_allowed;
    if (allowed === undefined) {
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
    }
    const allowedSet = new Set(allowed as string[]);
    const used = toolIndex.map(r => r.name);
    const forbidden = used.filter(n => !allowedSet.has(n));
    const ok = forbidden.length === 0;
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { forbidden, used } };
  }
}

export class MaxToolCalls extends Grader {
  name = 'MaxToolCalls';
  private maxCalls: number;
  private fromLimitsKey: string;

  constructor(maxCalls: number = 999999, fromLimitsKey: string = 'max_tool_calls') {
    super();
    this.maxCalls = maxCalls;
    this.fromLimitsKey = fromLimitsKey;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const lim = (task.limits || {})[this.fromLimitsKey];
    const maxCalls = lim !== undefined ? Number(lim) : this.maxCalls;
    const calls = toolIndex.length;
    const ok = calls <= maxCalls;
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { calls, max_calls: maxCalls } };
  }
}

export class MinToolCalls extends Grader {
  name = 'MinToolCalls';
  private minCalls: number;
  private fromLimitsKey: string;

  constructor(minCalls: number = 0, fromLimitsKey: string = 'min_tool_calls') {
    super();
    this.minCalls = minCalls;
    this.fromLimitsKey = fromLimitsKey;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const lim = (task.limits || {})[this.fromLimitsKey];
    const minCalls = lim !== undefined ? Number(lim) : this.minCalls;
    const calls = toolIndex.length;
    const ok = calls >= minCalls;
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { calls, min_calls: minCalls } };
  }
}

export class ToolCallSequence extends Grader {
  name = 'ToolCallSequence';
  private mustCall?: string;
  private fromExpectedKey: string;

  constructor(mustCall?: string, fromExpectedKey: string = 'must_call_tool') {
    super();
    this.mustCall = mustCall;
    this.fromExpectedKey = fromExpectedKey;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const must = this.mustCall || (task.expected || {})[this.fromExpectedKey];
    if (!must) {
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
    }
    const used = toolIndex.map(r => r.name);
    const ok = used.includes(String(must));
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { must_call: must, used } };
  }
}

export class ToolCallOrder extends Grader {
  name = 'ToolCallOrder';
  private expectedSequence?: string[];
  private fromExpectedKey: string;

  constructor(expectedSequence?: string[], fromExpectedKey: string = 'tool_call_order') {
    super();
    this.expectedSequence = expectedSequence;
    this.fromExpectedKey = fromExpectedKey;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const expected = this.expectedSequence || (task.expected || {})[this.fromExpectedKey];
    if (!expected) {
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
    }
    const actual = toolIndex.map(r => r.name);
    // Check if expected sequence appears in actual sequence (allowing extra calls)
    let expectedIdx = 0;
    for (const toolName of actual) {
      if (expectedIdx < expected.length && toolName === expected[expectedIdx]) {
        expectedIdx++;
      }
    }
    const ok = expectedIdx === expected.length;
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { expected, actual } };
  }
}

export class ToolResultJSON extends Grader {
  name = 'ToolResultJSON';
  private toolName?: string;
  private requiredKeys?: string[];

  constructor(toolName?: string, requiredKeys?: string[]) {
    super();
    this.toolName = toolName;
    this.requiredKeys = requiredKeys;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    // Configure from task.expected if not provided:
    const exp = task.expected || {};
    const tool = this.toolName || exp.tool_result_name;
    const keys = this.requiredKeys || exp.tool_result_required_keys;

    if (!tool || !keys) {
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
    }

    // Find last record for that tool
    const recs = toolIndex.filter(r => r.name === tool);
    if (recs.length === 0) {
      return { name: this.name, passed: false, score: 0.0, details: { error: 'tool_not_called', tool } };
    }

    const last = recs[recs.length - 1];
    let obj: any;
    if (typeof last.result === 'string') {
      try {
        obj = JSON.parse(last.result);
      } catch {
        return { name: this.name, passed: false, score: 0.0, details: { error: 'tool_result_not_json' } };
      }
    } else {
      obj = last.result;
    }

    if (typeof obj !== 'object' || Array.isArray(obj)) {
      return { name: this.name, passed: false, score: 0.0, details: { error: 'tool_result_not_object' } };
    }

    const missing = (keys as string[]).filter(k => !(k in obj));
    const ok = missing.length === 0;
    return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: { tool, missing } };
  }
}

// Outcome verification grader

export class OutcomeVerifier extends Grader {
  name = 'OutcomeVerifier';
  private verifierFn?: (messages: Message[], toolIndex: ToolCallRecord[], task: JSON) => boolean;

  constructor(verifierFn?: (messages: Message[], toolIndex: ToolCallRecord[], task: JSON) => boolean) {
    super();
    this.verifierFn = verifierFn;
  }

  grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): JSON {
    const verifier = this.verifierFn;
    if (!verifier) {
      // Try to get from task.expected
      const verifierData = (task.expected || {}).outcome_verifier;
      if (verifierData === undefined) {
        return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
      }
      // If it's a string, try to import it
      if (typeof verifierData === 'string') {
        // For now, require verifierFn to be passed directly
        return { name: this.name, passed: true, score: 1.0, details: { skipped: true, note: 'outcome_verifier must be passed as function' } };
      }
      // Otherwise assume it's a callable (though JSON can't serialize functions)
      // This won't work in practice, but we'll handle it gracefully
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true, note: 'outcome_verifier must be passed as function' } };
    }

    try {
      const ok = verifier(messages, toolIndex, task);
      return { name: this.name, passed: ok, score: ok ? 1.0 : 0.0, details: {} };
    } catch (e: any) {
      return { name: this.name, passed: false, score: 0.0, details: { error: 'verifier_exception', exception: String(e) } };
    }
  }
}

// LLM-as-judge grader

export class LLMJudge extends Grader {
  name = 'LLMJudge';
  private rubric?: string;
  private model: string;
  private temperature: number;
  private gradeTranscript: boolean;
  private fromExpectedKey: string;

  constructor(
    rubric?: string,
    model: string = 'gpt-4o-mini',
    temperature: number = 0.0,
    gradeTranscript: boolean = false,
    fromExpectedKey: string = 'llm_judge_rubric',
  ) {
    super();
    this.rubric = rubric;
    this.model = model;
    this.temperature = temperature;
    this.gradeTranscript = gradeTranscript;
    this.fromExpectedKey = fromExpectedKey;
  }

  async grade(messages: Message[], toolIndex: ToolCallRecord[], task: JSON, info: EpisodeInfo): Promise<JSON> {
    const rubric = this.rubric || (task.expected || {})[this.fromExpectedKey];
    if (!rubric) {
      return { name: this.name, passed: true, score: 1.0, details: { skipped: true } };
    }

    let client: any;
    try {
      // Dynamic import to avoid requiring OpenAI at module load time
      const { OpenAI } = await import('openai');
      client = new OpenAI();
    } catch {
      return { name: this.name, passed: false, score: 0.0, details: { error: 'openai_not_installed' } };
    }

    // Prepare content to grade
    let contentToGrade: string;
    if (this.gradeTranscript) {
      // Grade full transcript
      const transcriptText = messages
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => `${m.role}: ${m.content || ''}`)
        .join('\n');
      contentToGrade = `Transcript:\n${transcriptText}`;
    } else {
      // Grade only final assistant message
      contentToGrade = `Final assistant message: ${lastAssistantContent(messages)}`;
    }

    // Create judge prompt
    const judgePrompt = `You are evaluating an AI agent's performance. Here is the rubric:

${rubric}

Here is what to evaluate:

${contentToGrade}

Respond with a JSON object with:
- "passed": boolean (true if the agent meets the criteria)
- "score": float between 0.0 and 1.0
- "reasoning": string explaining your judgment
`;

    try {
      const response = await client.chat.completions.create({
        model: this.model,
        messages: [{ role: 'user', content: judgePrompt }] as any,
        temperature: this.temperature,
        response_format: { type: 'json_object' },
      });
      const result = JSON.parse(response.choices[0].message.content || '{}');
      const passed = Boolean(result.passed || false);
      const score = Number(result.score || 0.0);
      const reasoning = result.reasoning || '';
      return {
        name: this.name,
        passed,
        score: clamp01(score),
        details: { reasoning, model: this.model },
      };
    } catch (e: any) {
      return { name: this.name, passed: false, score: 0.0, details: { error: 'llm_judge_failed', exception: String(e) } };
    }
  }
}

export const BUILTIN_GRADERS: Record<string, new (...args: any[]) => Grader> = {
  // Code-based
  FinalRegex,
  FinalContains,
  FinalJSON,
  TranscriptContains,
  TranscriptRegex,
  ForbiddenTools,
  MaxToolCalls,
  MinToolCalls,
  ToolCallSequence,
  ToolCallOrder,
  ToolResultJSON,
  // Outcome verification
  OutcomeVerifier,
  // LLM-as-judge
  LLMJudge,
};

export function parseGraderSpec(spec: string): Grader {
  /**
   * Simple CLI grader spec format:
   *   - "FinalRegex" (uses task.expected.final_regex)
   *   - "FinalRegex:^133$" (explicit regex)
   *   - "FinalContains:Mike"
   *   - "MaxToolCalls:5"
   *   - "MinToolCalls:2"
   *   - "ToolCallSequence:calc"
   *   - "ToolCallOrder:calc,echo" (expected sequence)
   *   - "ToolResultJSON:calc,value" (tool=calc, required_keys=[value])
   *   - "LLMJudge" (uses task.expected.llm_judge_rubric)
   *   - "LLMJudge:gpt-4o-mini:0.0" (model:temperature)
   */
  if (!spec.includes(':')) {
    const Cls = BUILTIN_GRADERS[spec];
    if (!Cls) {
      throw new Error(`Unknown grader '${spec}'. Available: ${Object.keys(BUILTIN_GRADERS).join(', ')}`);
    }
    return new Cls();
  }

  const [name, arg] = spec.split(':', 2);
  const Cls = BUILTIN_GRADERS[name];
  if (!Cls) {
    throw new Error(`Unknown grader '${name}'. Available: ${Object.keys(BUILTIN_GRADERS).join(', ')}`);
  }

  if (name === 'FinalRegex') {
    return new FinalRegex(arg);
  }
  if (name === 'FinalContains') {
    return new FinalContains(arg);
  }
  if (name === 'TranscriptContains') {
    return new TranscriptContains(arg);
  }
  if (name === 'TranscriptRegex') {
    return new TranscriptRegex(arg);
  }
  if (name === 'MaxToolCalls') {
    return new MaxToolCalls(Number(arg));
  }
  if (name === 'MinToolCalls') {
    return new MinToolCalls(Number(arg));
  }
  if (name === 'ToolCallSequence') {
    return new ToolCallSequence(arg);
  }
  if (name === 'ToolCallOrder') {
    const sequence = arg.split(',').map(s => s.trim()).filter(s => s);
    return new ToolCallOrder(sequence);
  }
  if (name === 'ToolResultJSON') {
    const parts = arg.split(',').map(p => p.trim()).filter(p => p);
    const tool = parts[0] || undefined;
    const keys = parts.length > 1 ? parts.slice(1) : undefined;
    return new ToolResultJSON(tool, keys);
  }
  if (name === 'FinalJSON') {
    const keys = arg.split(',').map(k => k.trim()).filter(k => k);
    return new FinalJSON(keys);
  }
  if (name === 'LLMJudge') {
    // Format: model:temperature or just model
    const parts = arg.split(':');
    const model = parts[0] || 'gpt-4o-mini';
    const temp = parts[1] ? Number(parts[1]) : 0.0;
    return new LLMJudge(undefined, model, temp);
  }

  // Default: no-arg init
  return new Cls();
}

export function aggregateGrades(grades: JSON[]): JSON {
  /** Aggregate multiple grade results into a single result. */
  if (grades.length === 0) {
    return { passed: false, score: 0.0 };
  }
  const mean = grades.reduce((sum, g) => sum + Number(g.score || 0), 0) / grades.length;
  const passed = grades.every(g => Boolean(g.passed));
  return { passed, score: clamp01(mean) };
}
