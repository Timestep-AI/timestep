/** Episode runner - the core agent-environment loop. */

import type { AgentFn } from './agent.js';
import type { ToolFn } from './tools.js';
import { buildToolsSchema } from './tools.js';
import { isAssistantMessage } from '../utils/messages.js';
import { now } from '../utils/io.js';

export type JSON = Record<string, any>;
export type Message = Record<string, any>;

export interface EpisodeInfo {
  task_id: string;
  trial: number;
  seed: number;
  steps: number;
  tool_calls: number;
  duration_s: number;
  terminated_reason: string; // final_answer | max_steps | time_limit | error
  error?: string;
}

export function runEpisode(
  initialMessages: Message[],
  agent: AgentFn,
  tools: Record<string, ToolFn>,
  toolsAllowed: string[] | undefined,
  limits: JSON,
  taskMeta: JSON,
  seed: number,
): [Message[], EpisodeInfo] {
  /**
   * Runs the canonical loop:
   *   - agent returns assistant message
   *   - if assistant has tool_calls: env executes them and appends tool messages
   *   - else: assistant is final; done
   */
  const taskId = String(taskMeta.id || 'unknown');
  const maxSteps = Number((limits || {}).max_steps || 30);
  const timeLimitS = Number((limits || {}).time_limit_s || 120);

  const messages = [...initialMessages];
  const toolAllow = toolsAllowed ? new Set(toolsAllowed) : null;

  const t0 = now();
  let steps = 0;
  let toolCalls = 0;
  let terminatedReason = 'error';
  let err: string | undefined = undefined;

  // Provide schema to agent via context (optional)
  const toolsSchema = buildToolsSchema(tools, toolsAllowed);

  for (let step = 0; step < maxSteps; step++) {
    if (now() - t0 > timeLimitS) {
      terminatedReason = 'time_limit';
      break;
    }

    const context = {
      tools_schema: toolsSchema,
      task: taskMeta,
      seed,
      limits,
    };

    const assistantMsg = await agent(messages, context);
    if (!isAssistantMessage(assistantMsg)) {
      terminatedReason = 'error';
      err = 'agent_returned_non_assistant_message';
      // Append an error assistant message for observability
      messages.push({ role: 'assistant', content: '', _error: err });
      break;
    }

    // Normalize basic fields
    if (!assistantMsg.content) assistantMsg.content = '';
    messages.push(assistantMsg);
    steps++;

    const tcs = assistantMsg.tool_calls || [];
    if (tcs.length > 0) {
      // Execute tool calls and append tool messages
      for (const tc of tcs) {
        toolCalls++;
        const tcId = String(tc.id || '');
        const fn = tc.function || {};
        const name = String(fn.name || '');
        const argStr = fn.arguments || '{}';

        // Enforce allowlist
        if (toolAllow && !toolAllow.has(name)) {
          const result = { error: `forbidden_tool:${name}` };
          messages.push({
            role: 'tool',
            tool_call_id: tcId,
            content: JSON.stringify(result),
          });
          continue;
        }

        // Unknown tool
        if (!(name in tools)) {
          const result = { error: `unknown_tool:${name}` };
          messages.push({
            role: 'tool',
            tool_call_id: tcId,
            content: JSON.stringify(result),
          });
          continue;
        }

        // Parse arguments
        let args: JSON = {};
        try {
          args = typeof argStr === 'string' ? JSON.parse(argStr) : (argStr || {});
          if (typeof args !== 'object' || Array.isArray(args)) {
            args = { _non_dict_args: args };
          }
        } catch {
          const result = { error: 'invalid_tool_arguments_json' };
          messages.push({
            role: 'tool',
            tool_call_id: tcId,
            content: JSON.stringify(result),
          });
          continue;
        }

        // Invoke tool
        let res: any;
        try {
          res = tools[name](args);
        } catch (e: any) {
          res = { error: String(e) };
        }

        messages.push({
          role: 'tool',
          tool_call_id: tcId,
          content: JSON.stringify(res),
        });
      }

      // Continue loop (not done)
      continue;
    }

    // No tool calls => final answer => done
    terminatedReason = 'final_answer';
    break;
  }
  
  if (terminatedReason === 'error' && !err) {
    terminatedReason = 'max_steps';
  }

  const duration = now() - t0;
  const info: EpisodeInfo = {
    task_id: taskId,
    trial: Number(taskMeta._trial || 0),
    seed,
    steps,
    tool_calls: toolCalls,
    duration_s: Math.round(duration * 1000) / 1000,
    terminated_reason: terminatedReason,
    error: err,
  };
  
  return [messages, info];
}
