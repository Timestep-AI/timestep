/** Episode runner - the core agent-environment loop. */

import { AgentFn, JSON as JSONType, Message, StreamingAgentFn, ToolFn } from './types';
import { buildToolsSchema } from './tools';
import { isAssistantMessage } from '../utils/messages';
import { now } from '../utils/io';

function generateMessageId(): string {
  return `msg_${Math.random().toString(36).substring(2, 14)}`;
}

function generateRunId(taskId: string, trial?: number): string {
  if (trial !== undefined) {
    return `run_${taskId}_trial_${trial}`;
  }
  return `run_${taskId}`;
}

function generateThreadId(taskId: string): string {
  return `thread_${taskId}`;
}

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

async function* _runEpisodeStream(
  initialMessages: Message[],
  agent: AgentFn | StreamingAgentFn,
  tools: Record<string, ToolFn>,
  toolsAllowed?: string[],
  limits?: JSONType,
  taskMeta?: JSONType,
  seed: number = 0,
): AsyncGenerator<JSONType> {
  /**
   * Internal implementation that yields AG-UI protocol events as the episode progresses.
   * Supports both streaming and non-streaming agents.
   */
  const taskId = String((taskMeta || {}).id || 'unknown');
  const trial = (taskMeta || {})._trial as number | undefined;
  const threadId = generateThreadId(taskId);
  const runId = generateRunId(taskId, trial);
  
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
  
  // Emit RunStarted
  yield {
    type: 'RunStarted',
    threadId,
    runId,
    input: {
      messages: initialMessages,
      tools_allowed: toolsAllowed,
      limits,
      task_meta: taskMeta,
      seed,
    }
  };
  
  let currentMessageId: string | undefined = undefined;
  let currentStepName: string | undefined = undefined;

  for (let step = 0; step < maxSteps; step++) {
    if (now() - t0 > timeLimitS) {
      terminatedReason = 'time_limit';
      break;
    }

    currentStepName = `step_${step + 1}`;
    yield {
      type: 'StepStarted',
      stepName: currentStepName,
    };

    const context = {
      tools_schema: toolsSchema,
      task: taskMeta || {},
      seed,
      limits: limits || {},
    };

    // Handle streaming vs non-streaming agents
    let assistantMsg: Message = { role: 'assistant', content: '' };
    
    // Call agent - check if it returns an async generator (streaming) or a Message (non-streaming)
    const result = agent(messages, context);
    
    // Check if result is an async generator (streaming agent)
    if (result && typeof (result as any)[Symbol.asyncIterator] === 'function') {
      // It's a streaming agent - consume the stream
      let accumulatedContent = '';
      const accumulatedToolCalls: Record<string, any> = {};
      const toolCallIds: string[] = [];
      
      for await (const chunk of result as AsyncGenerator<JSONType>) {
        const chunkType = chunk.type || '';
        
        if (chunkType === 'content') {
          const delta = String(chunk.delta || '');
          accumulatedContent += delta;
          // Generate message ID on first content chunk
          if (!currentMessageId) {
            currentMessageId = generateMessageId();
            yield {
              type: 'TextMessageStart',
              messageId: currentMessageId,
              role: 'assistant',
            };
          }
          yield {
            type: 'TextMessageContent',
            messageId: currentMessageId,
            delta,
          };
        } else if (chunkType === 'tool_call') {
          const delta = chunk.delta || {};
          const tcId = String(delta.id || '');
          if (tcId && !accumulatedToolCalls[tcId]) {
            accumulatedToolCalls[tcId] = {
              id: tcId,
              type: 'function',
              function: { name: '', arguments: '' },
            };
            toolCallIds.push(tcId);
          }
          
          if (accumulatedToolCalls[tcId]) {
            const tc = accumulatedToolCalls[tcId];
            const fnDelta = delta.function || {};
            if (fnDelta.name) {
              tc.function.name = fnDelta.name;
            }
            if (fnDelta.arguments) {
              tc.function.arguments += fnDelta.arguments;
            }
          }
          
          // AG-UI ToolCallChunk
          const chunkFnDelta = delta.function || {};
          const chunkData: JSONType = {};
          if (chunkFnDelta.arguments) {
            chunkData._partial = chunkFnDelta.arguments;
          }
          yield {
            type: 'ToolCallChunk',
            toolCallId: tcId,
            chunk: chunkData,
          };
        } else if (chunkType === 'done') {
          break;
        } else if (chunkType === 'error') {
          err = String(chunk.error || 'unknown_error');
          terminatedReason = 'error';
          yield {
            type: 'RunError',
            message: err,
            code: 'AGENT_ERROR',
          };
          break;
        }
      }
      
      // Build complete message from accumulated state
      assistantMsg.content = accumulatedContent;
      if (Object.keys(accumulatedToolCalls).length > 0) {
        assistantMsg.tool_calls = toolCallIds.map(tcId => accumulatedToolCalls[tcId]);
      }
    } else {
      // Non-streaming agent - result is a Message (or Promise<Message>)
      if (result instanceof Promise) {
        assistantMsg = await result;
      } else {
        assistantMsg = result as Message;
      }
    }

    if (!isAssistantMessage(assistantMsg)) {
      terminatedReason = 'error';
      err = 'agent_returned_non_assistant_message';
      messages.push({ role: 'assistant', content: '', _error: err });
      yield {
        type: 'RunError',
        message: err,
        code: 'AGENT_ERROR',
      };
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

    // Handle message lifecycle for AG-UI
    if (assistantMsg.content && !currentMessageId) {
      currentMessageId = generateMessageId();
      yield {
        type: 'TextMessageStart',
        messageId: currentMessageId,
        role: assistantMsg.role || 'assistant',
      };
      // Emit content as single chunk if not already streamed
      if (assistantMsg.content) {
        yield {
          type: 'TextMessageContent',
          messageId: currentMessageId,
          delta: String(assistantMsg.content || ''),
        };
      }
    }
    
    // End the message
    if (currentMessageId) {
      yield {
        type: 'TextMessageEnd',
        messageId: currentMessageId,
      };
      currentMessageId = undefined;
    }

    const tcs = assistantMsg.tool_calls || [];
    if (tcs.length > 0) {
      // Execute tool calls and append tool messages
      for (const tc of tcs) {
        toolCalls++;
        const tcId = String(tc.id || '');
        const fn = tc.function || {};
        const name = String(fn.name || '');
        const argStr = fn.arguments || '{}';

        // AG-UI ToolCallStart
        yield {
          type: 'ToolCallStart',
          toolCallId: tcId,
          name,
        };
        
        // Parse arguments for ToolCallArgs
        let args: JSONType = {};
        try {
          args = typeof argStr === 'string' ? JSON.parse(argStr) : (argStr || {});
          if (typeof args !== 'object' || Array.isArray(args)) {
            args = { _non_dict_args: args };
          }
        } catch {
          args = {};
        }
        
        // Emit ToolCallArgs if we have arguments
        if (Object.keys(args).length > 0) {
          yield {
            type: 'ToolCallArgs',
            toolCallId: tcId,
            args,
          };
        }
        
        yield {
          type: 'ToolCallEnd',
          toolCallId: tcId,
        };

        // Enforce allowlist
        if (toolAllow && !toolAllow.has(name)) {
          const result = { error: `forbidden_tool:${name}` };
          messages.push({
            role: 'tool',
            tool_call_id: tcId,
            content: JSON.stringify(result),
          });
          yield {
            type: 'ToolCallResult',
            toolCallId: tcId,
            result,
          };
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
          yield {
            type: 'ToolCallResult',
            toolCallId: tcId,
            result,
          };
          continue;
        }

        // Parse arguments (reuse args from ToolCallArgs parsing above)
        // If args wasn't set above (shouldn't happen), parse again
        if (!args || Object.keys(args).length === 0) {
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
            yield {
              type: 'ToolCallResult',
              toolCallId: tcId,
              result,
            };
            continue;
          }
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
        yield {
          type: 'ToolCallResult',
          toolCallId: tcId,
          result: res,
        };
      }

      // Continue loop (not done)
      // Continue loop (not done)
      if (currentStepName) {
        yield {
          type: 'StepFinished',
          stepName: currentStepName,
        };
        currentStepName = undefined;
      }
      continue;
    }

    // No tool calls => final answer => done
    terminatedReason = 'final_answer';
    if (currentStepName) {
      yield {
        type: 'StepFinished',
        stepName: currentStepName,
      };
      currentStepName = undefined;
    }
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
    cost_usd: 0.0,
  };
  yield {
    type: 'RunFinished',
    threadId,
    runId,
    result: {
      transcript: messages,
      episodeInfo: {
        task_id: info.task_id,
        trial: info.trial,
        seed: info.seed,
        steps: info.steps,
        tool_calls: info.tool_calls,
        duration_s: info.duration_s,
        terminated_reason: info.terminated_reason,
        error: info.error,
        input_tokens: info.input_tokens,
        output_tokens: info.output_tokens,
        total_tokens: info.total_tokens,
        cost_usd: info.cost_usd,
      }
    }
  };
}

export async function runEpisode(
  initialMessages: Message[],
  agent: AgentFn,
  tools: Record<string, ToolFn>,
  toolsAllowed?: string[],
  limits?: JSONType,
  taskMeta?: JSONType,
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
        let args: JSONType = {};
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

export async function* streamEpisode(
  initialMessages: Message[],
  agent: AgentFn | StreamingAgentFn,
  tools: Record<string, ToolFn>,
  toolsAllowed?: string[],
  limits?: JSONType,
  taskMeta?: JSONType,
  seed: number = 0,
): AsyncGenerator<JSONType> {
  /**
   * Streaming version of `runEpisode()` that yields events and chunks in real-time.
   * 
   * Supports both streaming agents (StreamingAgentFn) and non-streaming agents (AgentFn).
   * 
   * Yields:
   * Emits AG-UI protocol events:
   *   - `RunStarted`, `StepStarted`, `StepFinished`, `RunFinished`
   *   - `TextMessageStart`, `TextMessageContent`, `TextMessageEnd`
   *   - `ToolCallStart`, `ToolCallArgs`, `ToolCallEnd`, `ToolCallChunk`, `ToolCallResult`
   *   - `RunError`
   */
  yield* _runEpisodeStream(
    initialMessages, agent, tools, toolsAllowed, limits, taskMeta, seed
  );
}
