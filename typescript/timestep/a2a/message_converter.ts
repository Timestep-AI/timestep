/** Unified message format conversion between OpenAI and A2A formats. */

import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import type { Message } from '@a2a-js/sdk';
import { randomUUID } from 'crypto';

/**
 * Extract text content from A2A message parts.
 */
export function extractTextFromParts(parts: Array<{ kind: string; text?: string }>): string {
  const textParts: string[] = [];
  for (const part of parts) {
    if (part.kind === 'text' && part.text) {
      textParts.push(part.text);
    }
  }
  return textParts.join('');
}

/**
 * Convert OpenAI messages to A2A format for display.
 *
 * Tool approvals are detected by checking if an assistant message with tool_calls
 * is followed by a tool message. If not, the tool call is pending approval.
 */
export function openaiToA2A(
  messages: ChatCompletionMessageParam[],
  contextId: string,
  taskId?: string
): Message[] {
  const a2aMessages: Message[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg.role === 'system') {
      // Skip system messages in display
      continue;
    }

    if (msg.role === 'user') {
      // User message
      a2aMessages.push({
        kind: 'message',
        role: 'user',
        messageId: randomUUID(),
        parts: [
          {
            kind: 'text',
            text: typeof msg.content === 'string' ? msg.content : '',
          },
        ],
        contextId: contextId,
        taskId: taskId,
        timestamp: new Date().toISOString(),
      });
    } else if (msg.role === 'assistant') {
      // Assistant message - check for tool calls
      const assistantMsg = msg as Extract<ChatCompletionMessageParam, { role: 'assistant' }>;

      if (assistantMsg.tool_calls && assistantMsg.tool_calls.length > 0) {
        // This assistant message has tool calls
        // Create an assistant message with tool calls (but no text content)
        // The tool results will be inserted after this message
        a2aMessages.push({
          kind: 'message',
          role: 'agent',
          messageId: randomUUID(),
          parts: assistantMsg.content
            ? typeof assistantMsg.content === 'string'
              ? [{ kind: 'text', text: assistantMsg.content }]
              : assistantMsg.content
                  .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
                  .map((part) => ({ kind: 'text' as const, text: part.text }))
            : [],
          contextId: contextId,
          taskId: taskId,
          timestamp: new Date().toISOString(),
          tool_calls: assistantMsg.tool_calls,
        });
      } else {
        // Regular assistant message
        const textContent =
          typeof assistantMsg.content === 'string'
            ? assistantMsg.content
            : assistantMsg.content
              ? assistantMsg.content
                  .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
                  .map((part) => part.text)
                  .join('')
              : '';

        if (textContent) {
          a2aMessages.push({
            kind: 'message',
            role: 'agent',
            messageId: randomUUID(),
            parts: [{ kind: 'text', text: textContent }],
            contextId: contextId,
            taskId: taskId,
            timestamp: new Date().toISOString(),
          });
        }
      }
    } else if (msg.role === 'tool') {
      // Tool message - include it in the output
      const toolMsg = msg as Extract<ChatCompletionMessageParam, { role: 'tool' }>;
      const toolContent =
        typeof toolMsg.content === 'string' ? toolMsg.content : JSON.stringify(toolMsg.content);
      const toolMessage: Message & { tool_call_id?: string } = {
        kind: 'message',
        role: 'tool',
        messageId: randomUUID(),
        parts: [{ kind: 'text', text: toolContent }],
        contextId: contextId,
        taskId: taskId,
        timestamp: new Date().toISOString(),
      };
      // Store tool_call_id for matching
      toolMessage.tool_call_id = toolMsg.tool_call_id;
      a2aMessages.push(toolMessage);
    }
  }

  return a2aMessages;
}

/**
 * Convert A2A message history to OpenAI message format.
 */
export function a2AToOpenAI(taskHistory: Message[] | undefined): ChatCompletionMessageParam[] {
  const messages: ChatCompletionMessageParam[] = [];

  if (!taskHistory || taskHistory.length === 0) {
    return messages;
  }

  for (const msg of taskHistory) {
    const role = msg.role;
    const parts = msg.parts || [];
    const text = extractTextFromParts(parts as Array<{ kind: string; text?: string }>);

    // Map A2A roles to OpenAI roles
    if (role === 'agent') {
      // Handle tool calls in assistant messages (text can be empty)
      const msgWithToolCalls = msg as Message & {
        toolCalls?: unknown[];
        tool_calls?: unknown[];
      };
      if (msgWithToolCalls.toolCalls || msgWithToolCalls.tool_calls) {
        const toolCalls = (msgWithToolCalls.toolCalls ||
          msgWithToolCalls.tool_calls) as unknown as Extract<
          ChatCompletionMessageParam,
          { role: 'assistant' }
        >['tool_calls'];
        messages.push({
          role: 'assistant',
          content: text || null,
          tool_calls: toolCalls,
        });
        continue;
      }
    } else if (role === 'user') {
      // User messages
    } else if (role === 'tool') {
      // Tool messages
      const msgWithToolCallId = msg as Message & {
        tool_call_id?: string;
        toolCallId?: string;
      };
      messages.push({
        role: 'tool',
        tool_call_id: msgWithToolCallId.tool_call_id || msgWithToolCallId.toolCallId || '',
        content: text,
      });
      continue;
    } else {
      continue;
    }

    // Skip messages with no text content (unless they have tool calls, handled above)
    if (!text) {
      continue;
    }

    // Skip approval responses - they should not be sent to the model
    const isApprovalResponse =
      text.toLowerCase().trim() === 'approve' || text.toLowerCase().trim() === 'reject';
    if (isApprovalResponse) {
      continue;
    }

    // Regular user or assistant messages
    if (role === 'user') {
      messages.push({
        role: 'user',
        content: text,
      });
    } else if (role === 'agent') {
      messages.push({
        role: 'assistant',
        content: text,
      });
    }
  }

  return messages;
}

// Legacy exports for backward compatibility
export const convertOpenAIToA2A = openaiToA2A;
export const convertA2AMessagesToOpenAI = a2AToOpenAI;

