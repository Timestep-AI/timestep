/** Handoff tool for agent-to-agent communication via A2A protocol. */

import { randomUUID } from 'crypto';
import type { ChatCompletionMessageToolCall } from 'openai/resources/chat/completions';

export interface HandoffArgs {
  message: string;
  agentId: string;
  sourceContextId?: string;
}

export interface HandoffCallbacks {
  onApprovalRequired?: (toolCall: ChatCompletionMessageToolCall) => Promise<boolean>;
  onChildMessage?: (message: {
    kind: string;
    role: string;
    messageId: string;
    parts: Array<{ kind: string; text?: string }>;
    contextId: string;
    taskId?: string;
    toolName?: string;
    tool_calls?: unknown[];
  }) => void;
}

/**
 * Parse SSE line and extract JSON data.
 */
function parseSSELine(line: string): unknown | null {
  if (line.startsWith('data: ')) {
    try {
      return JSON.parse(line.slice(6));
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Extract task ID and context ID from SSE event.
 */
function extractTaskInfo(result: unknown): { taskId?: string; contextId?: string } {
  if (result && typeof result === 'object') {
    const obj = result as Record<string, unknown>;
    if (obj.kind === 'task' && obj.id) {
      return {
        taskId: obj.id as string,
        contextId: (obj.contextId as string) || undefined,
      };
    } else if (obj.id && obj.contextId) {
      return {
        taskId: obj.id as string,
        contextId: obj.contextId as string,
      };
    }
  }
  return {};
}

/**
 * Handle approval request from child agent.
 */
async function handleChildApproval(
  approvalMessage: { parts?: Array<{ kind: string; text?: string }> },
  messageEndpoint: string,
  childContextId: string,
  taskId: string,
  onApprovalRequired: (toolCall: ChatCompletionMessageToolCall) => Promise<boolean>
): Promise<void> {
  if (!approvalMessage.parts) {
    return;
  }

  const approvalText = approvalMessage.parts
    .filter((p) => p.kind === 'text' && p.text)
    .map((p) => p.text)
    .join('');

  const toolMatch = approvalText.match(/Tool: (.+)/);
  const argsMatch = approvalText.match(/Arguments: (.+)/s);

  const toolName = toolMatch ? toolMatch[1].trim() : 'unknown';
  const toolArgs = argsMatch ? argsMatch[1].trim() : '{}';

  const syntheticToolCall: ChatCompletionMessageToolCall = {
    id: `handoff-approval-${taskId}`,
    type: 'function',
    function: {
      name: toolName,
      arguments: toolArgs,
    },
  };

  const approved = await onApprovalRequired(syntheticToolCall);
  const approvalResponseText = approved ? 'approve' : 'reject';
  const approvalMessageId = randomUUID();
  const approvalRequestId = Date.now().toString();

  try {
    await fetch(messageEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'message/stream',
        params: {
          message: {
            messageId: approvalMessageId,
            role: 'user',
            parts: [{ kind: 'text', text: approvalResponseText }],
            contextId: childContextId,
            taskId: taskId,
          },
        },
        id: approvalRequestId,
      }),
    });
  } catch (error) {
    console.error('[Handoff] Error sending approval:', error);
  }
}

/**
 * Establish parent-child relationship between contexts.
 */
async function establishParentRelationship(
  parentContextId: string,
  childContextId: string
): Promise<void> {
  try {
    const a2aServerUrl = process.env.A2A_SERVER_URL || 'http://localhost:8080';
    const updateUrl = `${a2aServerUrl}/contexts/${childContextId}`;

    console.log(`[Handoff] Establishing parent relationship: ${parentContextId} -> ${childContextId}`);

    const updateResponse = await fetch(updateUrl, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        parent_context_id: parentContextId,
      }),
    });

    if (!updateResponse.ok) {
      console.error(
        `[Handoff] Failed to set parent relationship: ${updateResponse.status} ${updateResponse.statusText}`
      );
    } else {
      console.log('[Handoff] Parent relationship established successfully');
    }
  } catch (error) {
    console.error('[Handoff] Error establishing parent relationship:', error);
  }
}

/**
 * Process SSE stream from child agent and forward messages to parent.
 */
async function processSSEStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  args: HandoffArgs,
  callbacks: HandoffCallbacks,
  messageEndpoint: string
): Promise<{ taskId: string; childContextId: string }> {
  const decoder = new TextDecoder();
  let buffer = '';
  let taskId: string | undefined;
  let childContextId: string | undefined;
  const childContextIdFromArgs = randomUUID(); // Fallback if not found in stream

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('[Handoff] SSE stream ended');
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const data = parseSSELine(line);
        if (!data || typeof data !== 'object') {
          continue;
        }

        const dataObj = data as Record<string, unknown>;
        if (!dataObj.result) {
          continue;
        }

        const result = dataObj.result;

        // Extract task ID and context ID from first event
        if (!taskId) {
          const taskInfo = extractTaskInfo(result);
          if (taskInfo.taskId) {
            taskId = taskInfo.taskId;
            childContextId = taskInfo.contextId || childContextIdFromArgs;
            console.log(`[Handoff] Found task ID from SSE: ${taskId}, context: ${childContextId}`);
          }
        }

        // Handle status updates
        if (
          result &&
          typeof result === 'object' &&
          (result as Record<string, unknown>).kind === 'status-update' &&
          (result as Record<string, unknown>).status
        ) {
          const statusResult = result as {
            kind: string;
            status: {
              state: string;
              message?: {
                role: string;
                parts?: Array<{ kind: string; text?: string }>;
              } & Record<string, unknown>;
            };
          };

          // Handle approvals from child agent
          if (
            statusResult.status.state === 'input-required' &&
            callbacks.onApprovalRequired &&
            messageEndpoint &&
            childContextId &&
            taskId
          ) {
            const approvalMessage = statusResult.status.message;
            if (approvalMessage) {
              await handleChildApproval(
                approvalMessage,
                messageEndpoint,
                childContextId,
                taskId,
                callbacks.onApprovalRequired
              );
            }
          }

          // Forward child messages to parent context
          if (
            statusResult.status.message &&
            args.sourceContextId &&
            callbacks.onChildMessage &&
            statusResult.status.state !== 'input-required'
          ) {
            const childMessage = statusResult.status.message;
            if (childMessage.role === 'agent' || childMessage.role === 'tool') {
              const childMessageWithExtras = childMessage as typeof childMessage & {
                toolName?: string;
                tool_calls?: unknown[];
              };
              console.log(`[Handoff] Forwarding child message to parent: ${childMessage.role}`, {
                toolName: childMessageWithExtras.toolName,
                hasToolCalls:
                  !!(
                    childMessageWithExtras.tool_calls &&
                    Array.isArray(childMessageWithExtras.tool_calls) &&
                    childMessageWithExtras.tool_calls.length > 0
                  ),
                toolCallsCount: childMessageWithExtras.tool_calls?.length || 0,
              });
              callbacks.onChildMessage({
                kind: 'message',
                ...childMessage,
                contextId: args.sourceContextId,
                toolName: childMessageWithExtras.toolName,
                tool_calls: childMessageWithExtras.tool_calls,
              });
            }
          }

          // Establish parent relationship when child completes
          if (
            statusResult.status.state === 'completed' &&
            args.sourceContextId &&
            childContextId
          ) {
            await establishParentRelationship(args.sourceContextId, childContextId);
          }

          // Stop when child completes or fails
          if (statusResult.status.state === 'completed' || statusResult.status.state === 'failed') {
            console.log(`[Handoff] Child agent ${statusResult.status.state}, stopping`);
            if (!taskId || !childContextId) {
              throw new Error('Task ID or context ID missing when child completed');
            }
            return { taskId, childContextId };
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (!taskId || !childContextId) {
    throw new Error('Failed to get task ID from target agent - task not found in SSE stream');
  }

  return { taskId, childContextId };
}

/**
 * Hand off a message to another agent via A2A protocol.
 */
export async function handoff(
  args: HandoffArgs,
  callbacks: HandoffCallbacks = {}
): Promise<string> {
  console.log(`[Handoff] Starting handoff to agent: ${args.agentId}`, {
    hasApprovalCallback: !!callbacks.onApprovalRequired,
  });

  try {
    // Get A2A server URL from environment, default to localhost:8080
    const a2aServerUrl = process.env.A2A_SERVER_URL || 'http://localhost:8080';
    const agentCardUrl = `${a2aServerUrl}/agents/${args.agentId}/.well-known/agent-card.json`;

    console.log(`[Handoff] Connecting to agent card: ${agentCardUrl}`);

    // Try to fetch the agent card first to verify connectivity
    try {
      const cardResponse = await fetch(agentCardUrl);
      if (!cardResponse.ok) {
        throw new Error(
          `Failed to fetch agent card: ${cardResponse.status} ${cardResponse.statusText}`
        );
      }
      const agentCard = (await cardResponse.json()) as { name?: string };
      console.log(`[Handoff] Agent card fetched successfully: ${agentCard.name || 'Unknown'}`);
    } catch (fetchError) {
      const fetchErrorMessage =
        fetchError instanceof Error ? fetchError.message : String(fetchError);
      console.error(`[Handoff] Failed to fetch agent card: ${fetchErrorMessage}`);
      throw new Error(`Failed to fetch agent card from ${agentCardUrl}: ${fetchErrorMessage}`);
    }

    // Create new context for child agent (allows parallel handoffs)
    const childContextId = randomUUID();
    const messageId = randomUUID();

    // Return the same JSON that was passed in as arguments
    const handoffResult = JSON.stringify({
      message: args.message,
      agentId: args.agentId,
    });

    // Send message via direct HTTP POST to child agent
    const agentBaseUrl = agentCardUrl.replace('/.well-known/agent-card.json', '');
    const messageEndpoint = `${agentBaseUrl}`;
    const requestId = Date.now().toString();

    console.log('[Handoff] Sending message to target agent via HTTP POST...', {
      endpoint: messageEndpoint,
      childContextId,
    });

    const response = await fetch(messageEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'message/stream',
        params: {
          message: {
            messageId: messageId,
            role: 'user',
            parts: [{ kind: 'text', text: args.message }],
            contextId: childContextId,
          },
        },
        id: requestId,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Failed to send message to target agent: ${response.status} ${response.statusText}`
      );
    }

    // Read SSE stream to forward child messages to parent and establish relationship
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('text/event-stream')) {
      throw new Error('Expected SSE stream response, got: ' + contentType);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    // Process SSE stream
    await processSSEStream(reader, args, callbacks, messageEndpoint);

    console.log('[Handoff] Handoff completed successfully');
    return handoffResult;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorStack = error instanceof Error ? error.stack : undefined;
    console.error(`[Handoff] Error during handoff to ${args.agentId}:`, errorMessage);
    if (errorStack) {
      console.error('[Handoff] Stack trace:', errorStack);
    }
    throw new Error(`Error during handoff: ${errorMessage}`);
  }
}

