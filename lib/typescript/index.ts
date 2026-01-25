/**
 * Timestep TypeScript Library
 * 
 * This is a placeholder for the future TypeScript implementation.
 * See README.md for more information.
 */

// TODO: Implement TypeScript version of Timestep library
// - Agent: A2A Server containing Loop
// - Environment: MCP Server (FastMCP) with system prompt and tools
// - Loop: AgentExecutor using MCP client
// - Human-in-the-loop: A2A input-required, MCP Elicitation, MCP Sampling
// - Trace-based evals: Verify handoffs and tool calls
// - All async/streaming-first

export interface Agent {
  agentId: string;
  name: string;
  // TODO: Implement
}

export interface Environment {
  environmentId: string;
  contextId: string;
  agentId: string;
  // TODO: Implement
}

export interface Loop {
  agentId: string;
  // TODO: Implement
}

// Placeholder exports
export const Agent = null;
export const Environment = null;
export const Loop = null;
