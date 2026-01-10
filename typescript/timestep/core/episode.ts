/** Episode runner - the core agent-environment loop. */

import { AgentFn, JSON, Message, ToolFn } from './types';
import { buildToolsSchema } from './tools';
import { isAssistantMessage } from '../../utils/messages';
import { now } from '../../utils/io';

export interface EpisodeInfo {
  task_id: string;
  trial: number;
  seed: number;
  steps: number;
  tool_calls: number;
  duration_s: number;
  terminated_reason: string; // final_answer | max_steps | time_limit | error
  error?: string;
  // Token and cost tracking (optional, populated if agent provides usage info)
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
}

function extractUsageFromMessage(msg: Message): { input_tokens: number; output_tokens: number; total_tokens: number } {
  /** Extract token usage from agent message if present. */
  const usage = msg.usage || {};
  return {
    input_tokens: Number(usage.prompt_tokens || 0) || 0,
    output_tokens: Number(usage.completion_tokens || 0) || 0,
    total_tokens: Number(usage.total_tokens || 0) || 0,
  };
}

export async function runEpisode(
  initialMessages: Message[],
  agent: AgentFn,
  tools: Record<string, ToolFn>,
  toolsAllowed?: string[],
  limits?: JSON,
  taskMeta?: JSON,
  seed: number = 0,
): Promise<[Message[], EpisodeInfo]> {
  /**
   * Orchestrates the agent harness in the canonical agent-environment loop:
   *   - Loop calls agent harness with messages and context
   *   - Agent harness returns assistant message
   *   - If assistant has tool_calls: environment executes them and appends tool messages, loop continues
   *   - Else: assistant is final; done
   * 
   * This is the core execution pattern that orchestrates the agent harness. The evaluation harness builds on top of this.
   */
  const taskId = String((taskMeta || {}).id || 'unknown');
  const maxSteps = Number((limits || {}).max_steps || 30);
  const timeLimitS = Number((limits || {}).time_limit_s || 120);

  const messages = [...initialMessages];
  const toolAllow = toolsAllowed ? new Set(toolsAllowed) : null;

  const t0 = now();
  let steps = 0;
  let toolCalls = 0;
  let terminatedReason = 'error';
  let err: string | undefined = undefined;
  
  // Token tracking
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let totalTokens = 0;

  // Provide schema to agent via context (optional)
  const toolsSchema = buildToolsSchema(tools, toolsAllowed);

  for (let step = 0; step < maxSteps; step++) {
    if (now() - t0 > timeLimitS) {
      terminatedReason = 'time_limit';
      break;
    }

    const context = {
      tools_schema: toolsSchema,
      task: taskMeta || {},
      seed,
      limits: limits || {},
    };

    const assistantMsg = await agent(messages, context);
    if (!isAssistantMessage(assistantMsg)) {
      terminatedReason = 'error';
      err = 'agent_returned_non_assistant_message';
      // Append an error assistant message for observability
      messages.push({ role: 'assistant', content: '', _error: err });
      break;
    }

    // Extract token usage if available
    const usage = extractUsageFromMessage(assistantMsg);
    totalInputTokens += usage.input_tokens;
    totalOutputTokens += usage.output_tokens;
    totalTokens += usage.total_tokens;

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

  if (steps >= maxSteps) {
    terminatedReason = 'max_steps';
  }

  const duration = now() - t0;
  const info: EpisodeInfo = {
    task_id: taskId,
    trial: Number((taskMeta || {})._trial || 0),
    seed,
    steps,
    tool_calls: toolCalls,
    duration_s: Math.round(duration * 1000) / 1000,
    terminated_reason: terminatedReason,
    error: err,
    input_tokens: totalInputTokens,
    output_tokens: totalOutputTokens,
    total_tokens: totalTokens,
    cost_usd: 0.0, // Can be calculated from tokens if pricing is known
  };
  return [messages, info];
}
