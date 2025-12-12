/** A2A AgentExecutor implementation for Timestep agent. */

import type {
  AgentExecutor,
  ExecutionEventBus,
  RequestContext,
} from '@a2a-js/sdk/server';
import type { Message, Task, TaskStatusUpdateEvent } from '@a2a-js/sdk';
import { randomUUID } from 'crypto';
import type { z } from 'zod';
import { runAgent } from '../core/agent.js';
import { GetWeatherParameters, WebSearchParameters } from '../core/tools.js';
import type { 
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
} from 'openai/resources/chat/completions';
import type { PostgresTaskStore } from './postgres_task_store.js';
import { AgentEventEmitter } from '../core/agent_events.js';
import { a2AToOpenAI, extractTextFromParts } from './message_converter.js';

// Re-export for convenience
export { extractTextFromParts } from './message_converter.js';

export class TimestepAgentExecutor implements AgentExecutor {
  /** AgentExecutor that wraps Timestep's runAgent function. */
  
  private tools: Array<{ name: string; parameters: z.ZodTypeAny }>;
  private model: string;
  private pendingApprovals: Map<string, {
    toolCall: ChatCompletionMessageToolCall;
    resolve: (approved: boolean) => void;
  }> = new Map();
  private taskStore?: PostgresTaskStore;

  constructor(
    tools?: Array<{ name: string; parameters: z.ZodTypeAny }>,
    model: string = 'gpt-4.1',
    taskStore?: PostgresTaskStore
  ) {
    /** Initialize the TimestepAgentExecutor.
     *
     * @param tools - List of tool objects to use. If undefined, uses default tools.
     * @param model - OpenAI model name to use.
     * @param taskStore - Optional PostgresTaskStore for saving OpenAI messages.
     */
    this.tools = tools || [
      { name: 'get_weather', parameters: GetWeatherParameters },
      { name: 'web_search', parameters: WebSearchParameters },
    ];
    this.model = model;
    this.taskStore = taskStore;
  }

  async execute(
    requestContext: RequestContext,
    eventBus: ExecutionEventBus
  ): Promise<void> {
    /** Execute the agent task.
     *
     * @param requestContext - Request context containing message and task information.
     * @param eventBus - Event bus for publishing A2A events.
     */
    const userMessage = requestContext.userMessage;
    const existingTask = requestContext.task;

    // Extract user input first to check if it's an approval response
    const userInput = extractTextFromParts(userMessage.parts as Array<{ kind: string; text?: string }>);
    
    // Check if this is an approval response (BEFORE determining taskId/contextId)
    // Approval responses are simple "approve" or "reject" messages
    const isApprovalResponse = userInput && (
      userInput.toLowerCase().trim() === 'approve' || 
      userInput.toLowerCase().trim() === 'reject'
    );
    
    // If it's an approval response, try to find a pending approval
    if (isApprovalResponse) {
      const messageContextId = userMessage.contextId;
      const messageTaskId = userMessage.taskId;
      const approved = userInput.toLowerCase().trim() === 'approve';
      
      console.log('[AgentExecutor] Received approval response', {
        messageContextId,
        messageTaskId,
        approved,
        pendingKeys: Array.from(this.pendingApprovals.keys())
      });
      
      // Try to find pending approval - first exact match, then by contextId
      let approvalKey: string | undefined;
      if (messageContextId && messageTaskId) {
        approvalKey = `${messageContextId}:${messageTaskId}`;
        if (!this.pendingApprovals.has(approvalKey)) {
          approvalKey = undefined;
        }
      }
      
      // If exact match failed, try to find any pending approval for this context
      if (!approvalKey && messageContextId) {
        for (const key of this.pendingApprovals.keys()) {
          if (key.startsWith(`${messageContextId}:`)) {
            approvalKey = key;
            break;
          }
        }
      }
      
      if (approvalKey) {
        const pendingApproval = this.pendingApprovals.get(approvalKey);
        if (pendingApproval) {
          console.log('Resolving approval', { approvalKey, approved });
          // Delete BEFORE resolving to avoid race conditions
          this.pendingApprovals.delete(approvalKey);
          try {
            pendingApproval.resolve(approved);
            console.log('Approval resolved successfully');
          } catch (error) {
            console.error('Error resolving approval:', error);
          }
          return; // Don't process as a new message - original agent execution will continue
        }
      }
      
      // If no approval found, log warning and return
      console.warn('Approval response received but no pending approval found', { 
        messageContextId, 
        messageTaskId,
        pendingKeys: Array.from(this.pendingApprovals.keys())
      });
      return; // Don't process as a new message
    }

    // 1. Task Creation
    // Use taskId from message if provided (for approval responses), otherwise use existing task or create new
    const taskId = userMessage.taskId || existingTask?.id || randomUUID();
    const contextId = userMessage.contextId || existingTask?.contextId || randomUUID();

    if (!userInput) {
      const failureUpdate: TaskStatusUpdateEvent = {
        kind: 'status-update',
        taskId: taskId,
        contextId: contextId,
        status: {
          state: 'failed',
          message: {
            kind: 'message',
            role: 'agent',
            messageId: randomUUID(),
            parts: [{ kind: 'text', text: 'No user input found in context.' }],
            taskId: taskId,
            contextId: contextId,
          },
          timestamp: new Date().toISOString(),
        },
        final: true,
      };
      eventBus.publish(failureUpdate);
      return;
    }

    // Only create new task if this is not an approval response
    if (!existingTask) {
      const initialTask: Task = {
        kind: 'task',
        id: taskId,
        contextId: contextId,
        status: {
          state: 'submitted',
          timestamp: new Date().toISOString(),
        },
        history: [userMessage],
        metadata: userMessage.metadata,
      };
      eventBus.publish(initialTask);
    }

    // 2. Working State
    const workingStatusUpdate: TaskStatusUpdateEvent = {
      kind: 'status-update',
      taskId: taskId,
      contextId: contextId,
      status: {
        state: 'working',
        message: {
          kind: 'message',
          role: 'agent',
          messageId: randomUUID(),
          parts: [{ kind: 'text', text: 'Processing your request...' }],
          taskId: taskId,
          contextId: contextId,
        },
        timestamp: new Date().toISOString(),
      },
      final: false,
    };
    eventBus.publish(workingStatusUpdate);

    // 4. History Conversion
    // Load full conversation history from context (all previous tasks)
    let openaiMessages: ChatCompletionMessageParam[] = [];
    
    if (this.taskStore) {
      try {
        // Get all OpenAI messages from previous tasks in this context
        const contextMessages = await (this.taskStore as PostgresTaskStore).getOpenAIMessagesByContextId(contextId);
        openaiMessages = [...contextMessages];
        console.log('Loaded context history', { contextId, messageCount: openaiMessages.length });
      } catch (error) {
        console.error('Error loading context history, falling back to task history', { error, contextId });
        // Fallback to task history if loading context messages fails
        const taskHistory: Message[] = existingTask?.history || [];
        openaiMessages = a2AToOpenAI(taskHistory);
      }
    } else {
      // Fallback: use task history if no taskStore available
      const taskHistory: Message[] = existingTask?.history || [];
      openaiMessages = a2AToOpenAI(taskHistory);
    }
    
    // Add the current user message if it's not an approval response
    if (userMessage) {
      const messageText = extractTextFromParts(userMessage.parts as Array<{ kind: string; text?: string }>);
      const isApproval = messageText && (
        messageText.toLowerCase().trim() === 'approve' || 
        messageText.toLowerCase().trim() === 'reject'
      );
      
      // Only add to history if it's not an approval response
      if (!isApproval) {
        const userText = extractTextFromParts(userMessage.parts as Array<{ kind: string; text?: string }>);
        openaiMessages.push({
          role: 'user',
          content: userText,
        });
        console.log('Added current user message to history', { messageText: userText.substring(0, 50) });
        
        // Save user message immediately
        if (this.taskStore) {
          try {
            await (this.taskStore as PostgresTaskStore).saveOpenAIMessages(taskId, openaiMessages);
            console.log('Saved user message immediately', { taskId, contextId });
          } catch (error) {
            console.error('Error saving user message', { error, taskId });
            // Don't fail - continue execution
          }
        }
      }
    }
    
    // Add system message if not present
    if (!openaiMessages.some(msg => msg.role === 'system')) {
      openaiMessages.unshift({
        role: 'system',
        content: 'You are a helpful AI assistant.',
      });
    }

    // 5. Agent Execution with streaming and tool approval
    let accumulatedContent = '';
    const streamingEvents: TaskStatusUpdateEvent[] = [];
    // Generate a single messageId for this streaming response - reuse for all deltas
    const streamingMessageId = randomUUID();

    // Create event emitter for agent events
    const eventEmitter = new AgentEventEmitter();

    // Handle delta events (streaming updates)
    eventEmitter.on('delta', (delta) => {
      if (delta.content) {
        accumulatedContent += delta.content;
        // Create event for this chunk (will be published later)
        // Use the same messageId for all deltas of this streaming message
        const event: TaskStatusUpdateEvent = {
          kind: 'status-update',
          taskId: taskId,
          contextId: contextId,
          status: {
            state: 'working',
            message: {
              kind: 'message',
              role: 'agent',
              messageId: streamingMessageId,
              parts: [{ kind: 'text', text: accumulatedContent }],
              taskId: taskId,
              contextId: contextId,
            },
            timestamp: new Date().toISOString(),
          },
          final: false,
        };
        streamingEvents.push(event);
      }
    });

    // Handle tool approval required events
    eventEmitter.on('tool-approval-required', async (event) => {
      const { toolCall, resolve } = event;
      // ChatCompletionMessageToolCall can be function or custom, check type
      const toolName = 'function' in toolCall ? (toolCall.function?.name || 'unknown') : 'unknown';
      const toolArgs = 'function' in toolCall ? (toolCall.function?.arguments || '{}') : '{}';
      
      let toolArgsDict: Record<string, unknown> = {};
      try {
        toolArgsDict = typeof toolArgs === 'string' ? JSON.parse(toolArgs) as Record<string, unknown> : toolArgs as Record<string, unknown>;
      } catch {
        toolArgsDict = {};
      }
      
      const approvalMessage = `Tool call requires approval:\nTool: ${toolName}\nArguments: ${JSON.stringify(toolArgsDict, null, 2)}`;
      
      const approvalUpdate: TaskStatusUpdateEvent = {
        kind: 'status-update',
        taskId: taskId,
        contextId: contextId,
        status: {
          state: 'input-required',
          message: {
            kind: 'message',
            role: 'agent',
            messageId: randomUUID(),
            parts: [{ kind: 'text', text: approvalMessage }],
            taskId: taskId,
            contextId: contextId,
          },
          timestamp: new Date().toISOString(),
        },
        final: false, // Don't close the stream - we need to continue after approval
      };
      await Promise.resolve(eventBus.publish(approvalUpdate));
      
      // Store approval request with resolver - use simple key that frontend can match
      const approvalKey = `${contextId}:${taskId}`;
      console.log('Tool approval required', { approvalKey, toolName, taskId, contextId });
      
      // Store the resolver - will be resolved when we get approval response
      this.pendingApprovals.set(approvalKey, {
        toolCall,
        resolve: (approved: boolean) => {
          console.log('Promise resolved with approval:', { approved, approvalKey });
          resolve(approved);
        },
      });
      console.log('Waiting for approval', { approvalKey, pendingCount: this.pendingApprovals.size });
    });

    // Handle tool result events
    eventEmitter.on('tool-result', (event) => {
      const { toolCallId, toolName, result } = event;
      /** Publish tool execution result as an event. */
      console.log('[onToolResult] Called', { toolName, toolCallId, resultLength: result.length, taskId, contextId, isChild: toolName.startsWith('child:') });
      
      // Check if this is a child message (format: "child:agent" or "child:tool" or "child:tool:toolName")
      if (toolName.startsWith('child:')) {
        console.log('[onToolResult] Handling child message', { toolName });
        const parts = toolName.split(':');
        const childRole = parts[1]; // 'agent' or 'tool'
        const childToolName = parts.length > 2 ? parts[2] : undefined; // toolName if present
        
        const childMessage: Message & { toolName?: string } = {
          kind: 'message',
          role: childRole === 'agent' ? 'agent' : 'tool',
          messageId: toolCallId,
          parts: [{ 
            kind: 'text', 
            text: result 
          }],
          taskId: taskId,
          contextId: contextId,
          timestamp: new Date().toISOString(),
        };
        
        // Preserve toolName for tool messages
        if (childRole === 'tool' && childToolName) {
          childMessage.toolName = childToolName;
        }
        
        // Publish child message event (no need to save A2A format - we use OpenAI format as source of truth)
        const childMessageEvent: TaskStatusUpdateEvent = {
          kind: 'status-update',
          taskId: taskId,
          contextId: contextId,
          status: {
            state: 'working',
            message: childMessage,
            timestamp: new Date().toISOString(),
          },
          final: false,
        };
        console.log('Publishing child message event to parent context', { taskId, contextId, role: childRole, toolName: childToolName });
        eventBus.publish(childMessageEvent);
        return;
      }
      
      // Regular tool result - include tool name in message for UI to identify
      console.log('[onToolResult] Creating tool result message', { toolName, result: result.substring(0, 100) });
      const toolResultMessage: Message & { toolName?: string } = {
        kind: 'message',
        role: 'tool',
        messageId: randomUUID(),
        parts: [{ 
          kind: 'text', 
          text: result
        }],
        taskId: taskId,
        contextId: contextId,
        toolName: toolName,
        timestamp: new Date().toISOString(),
      };
      
      // Publish tool result event (no need to save A2A format - we use OpenAI format as source of truth)
      const toolResultEvent: TaskStatusUpdateEvent = {
        kind: 'status-update',
        taskId: taskId,
        contextId: contextId,
        status: {
          state: 'working',
          message: toolResultMessage,
          timestamp: new Date().toISOString(),
        },
        final: false,
      };
      console.log('[onToolResult] Publishing tool result event', { taskId, contextId, toolName, messageToolName: toolResultMessage.toolName, eventMessageToolName: (toolResultEvent.status.message as Message & { toolName?: string }).toolName });
      eventBus.publish(toolResultEvent);
      console.log('[tool-result] Tool result event published successfully', { toolName });
    });

    // Handle assistant message events
    eventEmitter.on('assistant-message', async (event) => {
      const { message } = event;
      // Don't save "Processing your request..." messages - they're not real messages
      const content = typeof message.content === 'string' ? message.content : '';
      if (content === 'Processing your request...') {
        console.log('Skipping save for "Processing your request..." message');
        return;
      }
      
      // Check if this is an assistant message with tool calls
      const assistantMsg = message as Extract<ChatCompletionMessageParam, { role: 'assistant' }>;
      const hasToolCalls = !!(assistantMsg.tool_calls && assistantMsg.tool_calls.length > 0);
      
      if (this.taskStore) {
        try {
          // Save all messages up to and including this assistant message
          await (this.taskStore as PostgresTaskStore).saveOpenAIMessages(taskId, openaiMessages);
          console.log('Saved assistant message immediately', { taskId, messageCount: openaiMessages.length, hasToolCalls });
        } catch (error) {
          console.error('Error saving assistant message', { error, taskId });
          // Don't fail - continue execution
        }
      }
      
      // If this is an assistant message with tool calls, publish it as a status-update event
      // This ensures child agents forward it via onChildMessage callback
      if (hasToolCalls) {
        // Convert to A2A format for publishing
        const textContent = typeof assistantMsg.content === 'string' 
          ? assistantMsg.content 
          : assistantMsg.content
            ? assistantMsg.content
                .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
                .map(part => part.text)
                .join('')
            : '';
        
        const assistantA2AMessage: Message & { tool_calls?: unknown[] } = {
          kind: 'message',
          role: 'agent',
          messageId: randomUUID(),
          parts: textContent ? [{ kind: 'text', text: textContent }] : [],
          taskId: taskId,
          contextId: contextId,
          timestamp: new Date().toISOString(),
          tool_calls: assistantMsg.tool_calls,
        };
        
        const assistantEvent: TaskStatusUpdateEvent = {
          kind: 'status-update',
          taskId: taskId,
          contextId: contextId,
          status: {
            state: 'working',
            message: assistantA2AMessage,
            timestamp: new Date().toISOString(),
          },
          final: false,
        };
        
        console.log('Publishing assistant message with tool calls as event', { taskId, contextId, toolCallsCount: assistantMsg.tool_calls.length });
        eventBus.publish(assistantEvent);
      }
    });

    // Handle child message events from handoff
    eventEmitter.on('child-message', (event) => {
      const { message } = event;
      // Convert child A2A message to OpenAI format and add to parent's messages
      const textParts = message.parts
        ?.filter((p) => p.kind === 'text' && p.text)
        .map((p) => p.text)
        .join('') || '';
      
      if (message.role === 'agent') {
        // Don't save "Processing your request..." messages
        if (textParts === 'Processing your request...') {
          console.log('[onChildMessage] Skipping "Processing your request..." message');
          return;
        }
        
        // Child agent message - check if it has tool_calls
        const msgWithToolCalls = message as typeof message & { tool_calls?: unknown[] };
        if (msgWithToolCalls.tool_calls && Array.isArray(msgWithToolCalls.tool_calls) && msgWithToolCalls.tool_calls.length > 0) {
          // Verify handoff tool result is in the array before adding child message
          // The handoff tool result should have been added before this callback is called
          const hasHandoffToolResult = openaiMessages.some(msg => 
            msg.role === 'tool' && 
            typeof (msg as Extract<ChatCompletionMessageParam, { role: 'tool' }>).content === 'string' &&
            (msg as Extract<ChatCompletionMessageParam, { role: 'tool' }>).content.includes('"agentId":"weather-assistant"')
          );
          if (!hasHandoffToolResult) {
            console.warn('[onChildMessage] Handoff tool result not found in messages array before adding child message!', { 
              messageCount: openaiMessages.length,
              lastMessageRole: openaiMessages[openaiMessages.length - 1]?.role 
            });
          }
          
          // Convert to OpenAI format and add to parent's messages
          const openaiMessage: ChatCompletionMessageParam = {
            role: 'assistant',
            content: textParts || null,
            tool_calls: msgWithToolCalls.tool_calls as Extract<ChatCompletionMessageParam, { role: 'assistant' }>['tool_calls'],
          };
          openaiMessages.push(openaiMessage);
          console.log('[onChildMessage] Added child assistant message with tool calls to parent messages', { 
            toolCallsCount: msgWithToolCalls.tool_calls.length, 
            toolCalls: msgWithToolCalls.tool_calls,
            hasHandoffToolResult,
            totalMessages: openaiMessages.length
          });
          
          // Save immediately
          if (this.taskStore) {
            (this.taskStore as PostgresTaskStore).saveOpenAIMessages(taskId, openaiMessages).catch(error => {
              console.error('Error saving child assistant message', { error, taskId });
            });
          }
        } else if (textParts && textParts.trim()) {
          // Regular assistant message without tool calls (only if it has content)
          const openaiMessage: ChatCompletionMessageParam = {
            role: 'assistant',
            content: textParts,
          };
          openaiMessages.push(openaiMessage);
          console.log('[onChildMessage] Added child assistant message to parent messages');
          
          // Save immediately
          if (this.taskStore) {
            (this.taskStore as PostgresTaskStore).saveOpenAIMessages(taskId, openaiMessages).catch(error => {
              console.error('Error saving child assistant message', { error, taskId });
            });
          }
        }
      } else if (message.role === 'tool' && message.toolName) {
        // Child tool message - find the corresponding tool_call_id from the previous assistant message
        let toolCallId: string | undefined;
        for (let i = openaiMessages.length - 1; i >= 0; i--) {
          const prevMsg = openaiMessages[i];
          if (prevMsg.role === 'assistant') {
            const assistantMsg = prevMsg as Extract<ChatCompletionMessageParam, { role: 'assistant' }>;
            if (assistantMsg.tool_calls) {
              const matchingToolCall = assistantMsg.tool_calls.find(
                tc => tc.function?.name === message.toolName
              );
              if (matchingToolCall) {
                toolCallId = matchingToolCall.id;
                break;
              }
            }
          }
        }
        
        if (toolCallId) {
          // Parse content - it might be a JSON string
          let content: string = textParts;
          try {
            if (content.startsWith('"') && content.endsWith('"')) {
              content = JSON.parse(content);
            } else if (content.startsWith('{') || content.startsWith('[')) {
              content = JSON.parse(content);
            }
          } catch (e) {
            // Keep as string if parsing fails
          }
          
          const openaiMessage: ChatCompletionMessageParam = {
            role: 'tool',
            tool_call_id: toolCallId,
            content: content,
          };
          openaiMessages.push(openaiMessage);
          console.log('[onChildMessage] Added child tool message to parent messages', { toolName: message.toolName, toolCallId });
          
          // Save immediately
          if (this.taskStore) {
            (this.taskStore as PostgresTaskStore).saveOpenAIMessages(taskId, openaiMessages).catch(error => {
              console.error('Error saving child tool message', { error, taskId });
            });
          }
        } else {
          console.warn('[onChildMessage] Could not find tool_call_id for child tool message', { toolName: message.toolName });
        }
      }
      
      // Also forward via tool-result event for UI display
      if (textParts) {
        const childToolName = message.role === 'tool' && message.toolName
          ? `child:tool:${message.toolName}`
          : `child:${message.role}`;
        eventEmitter.emit('tool-result', {
          toolCallId: message.messageId,
          toolName: childToolName,
          result: textParts,
        });
      }
    });

    try {
      console.log('Starting runAgent', { taskId, contextId, messageCount: openaiMessages.length });
      // Run the agent with event emitter
      const finalResponse = await runAgent(
        openaiMessages,
        this.tools,
        this.model,
        undefined, // apiKey
        eventEmitter, // Pass event emitter instead of callbacks
        contextId // Pass contextId for handoff tool
      );
      console.log('runAgent completed', { taskId, contextId, responseLength: finalResponse?.length });
      
      // Save messages after runAgent completes (tool results may have been added)
      // The openaiMessages array is modified in place by runAgent (tool messages are appended)
      if (this.taskStore) {
        try {
          await (this.taskStore as PostgresTaskStore).saveOpenAIMessages(taskId, openaiMessages);
          console.log('Saved OpenAI messages after runAgent', { taskId, messageCount: openaiMessages.length });
        } catch (error) {
          console.error('Error saving messages after runAgent', { error, taskId });
          // Don't fail - continue execution
        }
      }
      
      // Publish streaming events (publish every 5th event to avoid too many updates)
      // But also publish the last event if there are any
      if (streamingEvents.length > 0) {
        for (let i = 0; i < streamingEvents.length; i++) {
          if (i % 5 === 0 || i === streamingEvents.length - 1) {
            eventBus.publish(streamingEvents[i]);
          }
        }
      }

      // 8. Completion - Publish final response
      // If we have accumulated content from streaming, use that; otherwise use finalResponse
      const finalText = accumulatedContent || finalResponse || '';
      console.log('Preparing final response', { taskId, contextId, textLength: finalText.length, accumulatedLength: accumulatedContent.length, finalResponseLength: finalResponse?.length });
      // Create final message for event (no need to save A2A format - we use OpenAI format as source of truth)
      const finalMessage: Message = {
        kind: 'message',
        role: 'agent',
        messageId: streamingMessageId, // Use same messageId as streaming deltas
        parts: [{ kind: 'text', text: finalText }],
        taskId: taskId,
        contextId: contextId,
        timestamp: new Date().toISOString(),
      };
      
      const finalUpdate: TaskStatusUpdateEvent = {
        kind: 'status-update',
        taskId: taskId,
        contextId: contextId,
        status: {
          state: 'completed',
          message: finalMessage,
          timestamp: new Date().toISOString(),
        },
        final: true,
      };
      console.log('Publishing final response', { taskId, contextId, textLength: finalText.length, eventKind: finalUpdate.kind });
      eventBus.publish(finalUpdate);
      console.log('Final response event published');

      // Add final assistant message to openaiMessages if it has content
      if (finalText && finalText !== 'Processing your request...') {
        openaiMessages.push({
          role: 'assistant',
          content: finalText,
        });
        
        // Save final assistant message immediately
        if (this.taskStore) {
          try {
            await (this.taskStore as PostgresTaskStore).saveOpenAIMessages(taskId, openaiMessages);
            console.log('Saved final assistant message', { taskId, messageCount: openaiMessages.length });
          } catch (error) {
            console.error('Error saving final assistant message', { error, taskId });
            // Don't fail - continue execution
          }
        }
      }
      // Note: We no longer save A2A format messages to task history.
      // OpenAI format messages are the single source of truth, converted to A2A format on-demand.

    } catch (error: unknown) {
      // Publish error status
      console.error('Error in agent execution', { error, taskId, contextId });
      const errorUpdate: TaskStatusUpdateEvent = {
        kind: 'status-update',
        taskId: taskId,
        contextId: contextId,
        status: {
          state: 'failed',
          message: {
            kind: 'message',
            role: 'agent',
            messageId: randomUUID(),
            parts: [{ kind: 'text', text: `Error: ${error instanceof Error ? error.message : String(error)}` }],
            taskId: taskId,
            contextId: contextId,
          },
          timestamp: new Date().toISOString(),
        },
        final: true,
      };
      eventBus.publish(errorUpdate);
      // Don't rethrow - let the execution complete gracefully
    }
  }

  async cancelTask(
    taskId: string,
    eventBus: ExecutionEventBus
  ): Promise<void> {
    /** Cancel the current task.
     *
     * @param taskId - Task ID to cancel.
     * @param eventBus - Event bus for publishing cancellation event.
     */
    const cancelledUpdate: TaskStatusUpdateEvent = {
      kind: 'status-update',
      taskId: taskId,
      contextId: '', // Will be set by the framework
      status: {
        state: 'canceled',
        timestamp: new Date().toISOString(),
      },
      final: true,
    };
    await Promise.resolve(eventBus.publish(cancelledUpdate));
  }
}

