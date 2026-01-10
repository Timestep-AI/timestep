/** Tool execution and indexing. */

import { JSON, Message, ToolFn } from './types';

export function toolCalc(args: JSON): any {
  /**
   * Demo tool: calculates arithmetic expression in args["expr"].
   * WARNING: For production, do NOT use eval. Use a safe expression parser.
   */
  const expr = String(args.expr || '');
  // Extremely restricted eval (still not perfect for production)
  // eslint-disable-next-line no-eval
  const val = eval(expr);
  return { expr, value: val };
}

export function toolEcho(args: JSON): any {
  /** Demo tool: echoes back arguments. */
  return { echo: args };
}

export const DEFAULT_TOOLS: Record<string, ToolFn> = {
  calc: toolCalc,
  echo: toolEcho,
};

export function buildToolsSchema(tools: Record<string, ToolFn>, allowed?: string[]): JSON[] {
  /**
   * Builds a minimal OpenAI-style tools schema list.
   * 
   * NOTE: This is a minimal schema for interoperability; you can extend it.
   */
  let names = Object.keys(tools).sort();
  if (allowed) {
    const allowedSet = new Set(allowed);
    names = names.filter(n => allowedSet.has(n));
  }

  // Minimal function schema; arguments left open (free-form JSON) by default.
  const schema: JSON[] = [];
  for (const name of names) {
    schema.push({
      type: 'function',
      function: {
        name,
        description: `Tool '${name}'`,
        parameters: {
          type: 'object',
          properties: {},
          additionalProperties: true,
        },
      },
    });
  }
  return schema;
}

export interface ToolCallRecord {
  tool_call_id: string;
  name: string;
  arguments_raw: any;
  arguments: JSON;
  result_raw: string;
  result: any;
  error?: string;
}

export function indexToolCalls(messages: Message[]): ToolCallRecord[] {
  /**
   * Pairs assistant tool calls with subsequent tool messages by tool_call_id.
   * 
   * Returns a list of ToolCallRecord in chronological order of the tool calls.
   */
  // Map tool_call_id -> (name, arguments_raw)
  const calls: Record<string, [string, any]> = {};
  const orderedIds: string[] = [];

  for (const m of messages) {
    if (m.role === 'assistant') {
      for (const tc of (m.tool_calls || [])) {
        const tcId = String(tc.id || '');
        const fn = tc.function || {};
        const name = String(fn.name || '');
        const argsRaw = fn.arguments || '{}';
        if (tcId) {
          calls[tcId] = [name, argsRaw];
          orderedIds.push(tcId);
        }
      }
    }
  }

  // Pair with tool results
  const results: Record<string, [string, any]> = {}; // id -> (raw_content, parsed)
  for (const m of messages) {
    if (m.role === 'tool') {
      const tcId = String(m.tool_call_id || '');
      const raw = String(m.content || '');
      let parsed: any = null;
      try {
        parsed = JSON.parse(raw);
      } catch {
        parsed = raw;
      }
      if (tcId) {
        results[tcId] = [raw, parsed];
      }
    }
  }

  const out: ToolCallRecord[] = [];
  for (const tcId of orderedIds) {
    const [name, argsRaw] = calls[tcId] || ['', '{}'];
    let argsParsed: JSON = {};
    let err: string | undefined = undefined;
    try {
      argsParsed = typeof argsRaw === 'string' ? JSON.parse(argsRaw) : (argsRaw || {});
      if (typeof argsParsed !== 'object' || Array.isArray(argsParsed)) {
        argsParsed = { _non_dict_args: argsParsed };
      }
    } catch {
      err = 'invalid_tool_arguments_json';
      argsParsed = {};
    }

    const [resRaw, resParsed] = results[tcId] || ['', null];
    out.push({
      tool_call_id: tcId,
      name,
      arguments_raw: argsRaw,
      arguments: argsParsed,
      result_raw: resRaw,
      result: resParsed,
      error: err,
    });
  }
  return out;
}
