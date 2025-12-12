/** Streaming agent core using OpenAI streaming API. */

import OpenAI from 'openai';
// Note: In TypeScript, OpenAI client is already async (all methods return promises)
// This is equivalent to AsyncOpenAI in Python
import { zodFunction } from 'openai/helpers/zod';
import type { 
  ChatCompletionMessageParam, 
  ChatCompletionMessageToolCall,
  ChatCompletionMessageFunctionToolCall 
} from 'openai/resources/chat/completions';
import type { z } from 'zod';
import { callFunction } from './tools';
import type { AgentEventEmitter } from './agent_events';

function mergeToolCalls(
  existingToolCalls: ChatCompletionMessageToolCall[],
  deltaToolCalls: Array<{
    index: number;
    id?: string;
    type?: string;
    function?: {
      name?: string;
      arguments?: string;
    };
  }>
): ChatCompletionMessageToolCall[] {
  /** Merge incremental tool call deltas into complete tool calls. */
  const toolCallsMap = new Map<number, ChatCompletionMessageToolCall>();
  
  // Initialize map with existing tool calls
  existingToolCalls.forEach((tc, idx) => {
    toolCallsMap.set(idx, { ...tc });
  });
  
  // Merge deltas
  for (const deltaTc of deltaToolCalls) {
    const index = deltaTc.index;
    
    if (!toolCallsMap.has(index)) {
      // New tool call (we only support function tool calls)
      toolCallsMap.set(index, {
        id: deltaTc.id || '',
        type: 'function' as const,
        function: {
          name: '',
          arguments: '',
        },
      } as ChatCompletionMessageFunctionToolCall);
    }
    
    const toolCall = toolCallsMap.get(index)! as ChatCompletionMessageFunctionToolCall;
    
    // Merge function name
    if (deltaTc.function?.name) {
      toolCall.function.name = deltaTc.function.name;
    }
    
    // Merge arguments (they come as incremental strings)
    if (deltaTc.function?.arguments) {
      toolCall.function.arguments += deltaTc.function.arguments;
    }
    
    // Update ID if provided
    if (deltaTc.id) {
      toolCall.id = deltaTc.id;
    }
  }
  
  // Convert back to array, sorted by index
  return Array.from(toolCallsMap.entries())
    .sort(([a], [b]) => a - b)
    .map(([, tc]) => tc);
}

function serialize(result: unknown): string {
  /** Serialize tool execution result to JSON string. */
  try {
    if (typeof result === 'string') {
      // Try to parse as JSON first, if it fails return as-is
      try {
        JSON.parse(result);
        return result;
      } catch {
        // Not valid JSON, wrap in JSON string
        return JSON.stringify(result);
      }
    } else {
      return JSON.stringify(result);
    }
  } catch {
    // Fallback: convert to string and wrap
    return JSON.stringify(String(result));
  }
}

export async function runAgent(
  messages: ChatCompletionMessageParam[],
  tools?: Array<{ name: string; parameters: z.ZodTypeAny }>,
  model: string = 'gpt-4.1',
  apiKey?: string,
  eventEmitter?: AgentEventEmitter,
  contextId?: string,
): Promise<string> {
  /** Run agent with streaming OpenAI API and tool support.
   *
   * @param messages - List of message objects with 'role' and 'content'
   * @param tools - Optional list of tool objects with schema (from zodFunction) and execute function
   * @param model - OpenAI model name (default: "gpt-4.1")
   * @param apiKey - OpenAI API key (defaults to OPENAI_API_KEY env var)
   * @param eventEmitter - Optional event emitter for agent events
   * @param contextId - Optional context ID for handoff tool
   * @returns Final assistant response as string
   */
  const apiKeyToUse = apiKey || process.env.OPENAI_API_KEY;
  
  if (!apiKeyToUse) {
    throw new Error('OpenAI API key is required. Set OPENAI_API_KEY env var or pass apiKey parameter.');
  }
  
  // Initialize OpenAI client (already async in TypeScript)
  const client = new OpenAI({ apiKey: apiKeyToUse });
  
  // eslint-disable-next-line no-constant-condition
  while (true) {
    // 1. STREAM ASSISTANT OUTPUT
    const stream = await client.chat.completions.create({
      model,
      messages,
      tools: tools ? tools.map(tool => zodFunction(tool)) : undefined,
      stream: true,
    });
    
    const assistantMsg: ChatCompletionMessageParam = {
      role: 'assistant',
      content: '',
    };
    let toolCalls: ChatCompletionMessageToolCall[] = [];
    
    for await (const event of stream) {
      // Check if this is an assistant message delta event
      const choice = event.choices[0];
      if (choice?.delta) {
        const delta = choice.delta;
        
        // Handle content delta
        if (delta.content) {
          const content = typeof assistantMsg.content === 'string' ? assistantMsg.content : '';
          assistantMsg.content = content + delta.content;
          if (eventEmitter) {
            eventEmitter.emit('delta', { content: delta.content });
          }
        }
        
        // Handle tool_calls delta
        if (delta.tool_calls && delta.tool_calls.length > 0) {
          toolCalls = mergeToolCalls(
            toolCalls,
            delta.tool_calls
          );
          if (eventEmitter) {
            eventEmitter.emit('delta', { tool_calls: delta.tool_calls });
          }
        }
      }
    }
    
    // Only include tool_calls if there are any (OpenAI doesn't allow empty arrays)
    if (toolCalls.length > 0) {
      assistantMsg.tool_calls = toolCalls;
    }
    
    // Append assistant message to conversation
    messages.push(assistantMsg);
    
    // Emit assistant message event (before waiting for approval)
    if (eventEmitter) {
      try {
        await eventEmitter.emit('assistant-message', { message: assistantMsg });
      } catch (error) {
        console.error('[runAgent] Error in assistant-message event', { error });
        // Don't fail - continue execution
      }
    }
    
    // 2. MULTIPLE TOOL CALLS
    if (toolCalls.length > 0) {
      const toolMessages: ChatCompletionMessageParam[] = [];
      
      for (const toolCall of toolCalls) {
        // --- HUMAN APPROVAL ---
        let approved = false;
        if (eventEmitter) {
          console.log('[runAgent] Requesting tool approval', { toolCallId: toolCall.id, toolName: 'function' in toolCall ? toolCall.function?.name : 'unknown' });
          // Create a promise that will be resolved by the event handler
          approved = await new Promise<boolean>((resolve) => {
            eventEmitter.emit('tool-approval-required', {
              toolCall,
              resolve,
            });
          });
          console.log('[runAgent] Tool approval received', { approved, toolCallId: toolCall.id });
        } else {
          // Default: auto-approve if no event emitter provided
          approved = true;
        }
        
        if (approved) {
          // EXECUTE TOOL
          // We only support function tool calls
          if (toolCall.type !== 'function') {
            throw new Error(`Unsupported tool call type: ${toolCall.type}`);
          }
          const toolName = toolCall.function.name;
          const toolArgs = toolCall.function.arguments;
          console.log('[runAgent] Executing tool', { toolName, toolCallId: toolCall.id });
          const args: Record<string, unknown> = typeof toolArgs === 'string' ? JSON.parse(toolArgs) as Record<string, unknown> : toolArgs as Record<string, unknown>;
          
          // For handoff tool, create the tool result message BEFORE initiating the handoff
          // This ensures it appears in the messages array before any child messages
          if (toolName === 'handoff') {
            const handoffResult = JSON.stringify({ message: args.message, agentId: args.agentId });
            // Add tool result message to messages array immediately
            const handoffToolMessage: ChatCompletionMessageParam = {
              role: 'tool',
              tool_call_id: toolCall.id,
              content: handoffResult,
            };
            messages.push(handoffToolMessage);
            // Save immediately to ensure it's persisted before child messages arrive
            if (eventEmitter) {
              try {
                // Emit assistant-message event to save (it saves all messages up to this point)
                await eventEmitter.emit('assistant-message', { message: handoffToolMessage });
              } catch (error) {
                console.error('[runAgent] Error saving handoff tool result', { error });
                // Don't fail - continue execution
              }
            }
            // Also publish via tool-result event so the UI can see it during streaming
            if (eventEmitter) {
              console.log('[runAgent] Publishing handoff tool result via tool-result event', { toolCallId: toolCall.id });
              eventEmitter.emit('tool-result', {
                toolCallId: toolCall.id,
                toolName,
                result: handoffResult,
              });
            }
            console.log('[runAgent] Created and saved handoff tool result before initiating handoff', { toolCallId: toolCall.id });
          }
          
          // For handoff tool, create a child message callback that emits events
          const childMessageCallback = toolName === 'handoff' && eventEmitter
            ? (message: { kind: string; role: string; messageId: string; parts: Array<{ kind: string; text?: string }>; contextId: string; taskId?: string; toolName?: string; tool_calls?: unknown[] }) => {
                eventEmitter.emit('child-message', { message });
              }
            : undefined;
          
          // For handoff, create a tool result callback that emits events
          const toolResultCallback = toolName === 'handoff' ? undefined : eventEmitter
            ? (toolCallId: string, toolName: string, result: string) => {
                eventEmitter.emit('tool-result', { toolCallId, toolName, result });
              }
            : undefined;
          
          // For approval, create a callback that uses the event emitter
          const approvalCallback = eventEmitter
            ? async (toolCall: ChatCompletionMessageToolCall): Promise<boolean> => {
                return new Promise<boolean>((resolve) => {
                  eventEmitter.emit('tool-approval-required', {
                    toolCall,
                    resolve,
                  });
                });
              }
            : undefined;
          
          const result = await callFunction(toolName, args, approvalCallback, contextId, toolResultCallback, toolCall.id, childMessageCallback);
          const serializedResult = serialize(result);
          console.log('[runAgent] Tool executed successfully', { toolName, toolCallId: toolCall.id, resultLength: typeof result === 'string' ? result.length : JSON.stringify(result).length, result: serializedResult.substring(0, 100) });
          
          // Publish tool result - handoff already created its result, so skip here
          if (eventEmitter && toolName !== 'handoff') {
            console.log('[runAgent] Publishing tool result', { toolName, toolCallId: toolCall.id });
            eventEmitter.emit('tool-result', {
              toolCallId: toolCall.id,
              toolName,
              result: serializedResult,
            });
            console.log('[runAgent] Tool result published', { toolName });
          } else if (toolName === 'handoff') {
            console.log('[runAgent] Skipping tool result publish for handoff (already created)', { toolName });
          }
          
          // Only add tool message if it wasn't already added (for handoff)
          if (toolName !== 'handoff') {
            toolMessages.push({
              role: 'tool',
              tool_call_id: toolCall.id,
              content: serializedResult,
            });
          }
        } else {
          // SYNTHETIC ERROR TOOL MSG
          console.log('[runAgent] Tool call rejected', { toolCallId: toolCall.id });
          toolMessages.push({
            role: 'tool',
            tool_call_id: toolCall.id,
            content: serialize({ error: 'Human rejected tool call' }),
          });
        }
      }
      
      // Append ALL tool messages in order
      messages.push(...toolMessages);
      continue;
    }
    
    // 3. NO TOOLS → FINAL ANSWER
    // Content can be string, array, or null - we ensure it's a string
    const content = assistantMsg.content;
    if (typeof content === 'string') {
      return content;
    } else if (Array.isArray(content)) {
      // If content is an array, extract text from text parts
      return content
        .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
        .map(part => part.text)
        .join('');
    } else {
      return '';
    }
  }
}

