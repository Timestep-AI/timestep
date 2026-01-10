/** Built-in graders for evaluation. */

import type { Message } from '../utils/messages.js';
import { lastAssistantContent } from '../utils/messages.js';
import { clamp01 } from '../utils/io.js';
import type { ToolCallRecord, EpisodeInfo } from './episode.js';

export type JSON = Record<string, any>;

export abstract class Grader {
  name: string = 'Grader';

  abstract grade(
    messages: Message[],
    toolIndex: ToolCallRecord[],
    task: JSON,
    info: EpisodeInfo
  ): JSON;
}

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

export const BUILTIN_GRADERS: Record<string, new (...args: any[]) => Grader> = {
  FinalRegex,
  FinalContains,
  FinalJSON,
  ForbiddenTools,
  MaxToolCalls,
  ToolCallSequence,
  ToolResultJSON,
};

export function parseGraderSpec(spec: string): Grader {
  /**
   * Simple CLI grader spec format:
   *   - "FinalRegex" (uses task.expected.final_regex)
   *   - "FinalRegex:^133$" (explicit regex)
   *   - "FinalContains:Mike"
   *   - "MaxToolCalls:5"
   *   - "ToolCallSequence:calc"
   *   - "ToolResultJSON:calc,value" (tool=calc, required_keys=[value])
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
  if (name === 'MaxToolCalls') {
    return new MaxToolCalls(Number(arg));
  }
  if (name === 'ToolCallSequence') {
    return new ToolCallSequence(arg);
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
