/** Convert OpenAI messages to A2A format for display. */

import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import type { Message } from '@a2a-js/sdk';
import { randomUUID } from 'crypto';

export function convertOpenAIToA2A(
  messages: ChatCompletionMessageParam[],
  contextId: string,
  taskId?: string
): Message[] {
  /** Convert OpenAI messages to A2A format for UI display.
   *
   * Tool approvals are detected by checking if an assistant message with tool_calls
   * is followed by a tool message. If not, the tool call is pending approval.
   */
  const a2aMessages: Message[] = [];
  
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const nextMsg = messages[i + 1];
    
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
            ? (typeof assistantMsg.content === 'string' 
                ? [{ kind: 'text', text: assistantMsg.content }]
                : assistantMsg.content
                    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
                    .map(part => ({ kind: 'text' as const, text: part.text }))
              )
            : [],
          contextId: contextId,
          taskId: taskId,
          timestamp: new Date().toISOString(),
          tool_calls: assistantMsg.tool_calls,
        });
      } else {
        // Regular assistant message
        const textContent = typeof assistantMsg.content === 'string'
          ? assistantMsg.content
          : assistantMsg.content
            ? assistantMsg.content
                .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
                .map(part => part.text)
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
      const toolContent = typeof toolMsg.content === 'string' ? toolMsg.content : JSON.stringify(toolMsg.content);
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

