/** Message utilities for OpenAI chat protocol. */

import { stableHash } from './hashing';
import type { JSON, Message } from '../core/types';

export type { Message, JSON };

export function isAssistantMessage(msg: Message): boolean {
  return msg.role === 'assistant';
}

export function isToolMessage(msg: Message): boolean {
  return msg.role === 'tool';
}

export function lastAssistantContent(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === 'assistant') {
      return String(m.content || '');
    }
  }
  return '';
}

export function ensureTaskId(task: JSON): string {
  if (!task.id || !String(task.id).trim()) {
    task.id = stableHash(task);
  }
  return String(task.id);
}
