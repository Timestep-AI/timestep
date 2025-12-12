/** Event emitter for agent execution events. */

import { EventEmitter } from 'events';
import type { ChatCompletionMessageParam, ChatCompletionMessageToolCall } from 'openai/resources/chat/completions';

export interface AgentDeltaEvent {
  content?: string;
  tool_calls?: Array<{
    index: number;
    id?: string;
    type?: string;
    function?: {
      name?: string;
      arguments?: string;
    };
  }>;
}

export interface ToolApprovalRequiredEvent {
  toolCall: ChatCompletionMessageToolCall;
  resolve: (approved: boolean) => void;
}

export interface ToolResultEvent {
  toolCallId: string;
  toolName: string;
  result: string;
}

export interface AssistantMessageEvent {
  message: ChatCompletionMessageParam;
}

export interface ChildMessageEvent {
  message: {
    kind: string;
    role: string;
    messageId: string;
    parts: Array<{ kind: string; text?: string }>;
    contextId: string;
    taskId?: string;
    toolName?: string;
    tool_calls?: unknown[];
  };
}

export interface AgentEvents {
  delta: (event: AgentDeltaEvent) => void;
  'tool-approval-required': (event: ToolApprovalRequiredEvent) => void;
  'tool-result': (event: ToolResultEvent) => void;
  'assistant-message': (event: AssistantMessageEvent) => Promise<void>;
  'child-message': (event: ChildMessageEvent) => void;
}

export class AgentEventEmitter extends EventEmitter {
  on<K extends keyof AgentEvents>(event: K, listener: AgentEvents[K]): this {
    return super.on(event, listener);
  }

  emit<K extends keyof AgentEvents>(event: K, ...args: Parameters<AgentEvents[K]>): boolean {
    return super.emit(event, ...args);
  }

  once<K extends keyof AgentEvents>(event: K, listener: AgentEvents[K]): this {
    return super.once(event, listener);
  }

  off<K extends keyof AgentEvents>(event: K, listener: AgentEvents[K]): this {
    return super.off(event, listener);
  }
}

