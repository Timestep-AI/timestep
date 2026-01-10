/** Core types for the agent-environment loop. */

export type JSON = Record<string, any>;
export type Message = Record<string, any>;

// Agent harness: function that takes messages and context, returns assistant message
export type AgentFn = (messages: Message[], context: JSON) => Message | Promise<Message>;

// Tool function: deterministic function that takes args and returns result
export type ToolFn = (args: JSON) => any;
