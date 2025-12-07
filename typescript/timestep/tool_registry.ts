/** Tool registry for mapping tool identifiers to Tool objects. */

import type { Tool } from '@openai/agents';

// Global registry mapping tool identifiers to Tool objects
const toolRegistry = new Map<string, Tool<any>>();

/**
 * Register a tool in the registry.
 *
 * @param identifier - Unique identifier for the tool (typically tool.name)
 * @param tool - The Tool object to register
 */
export function registerTool(identifier: string, tool: Tool<any>): void {
  toolRegistry.set(identifier, tool);
}

/**
 * Get a tool from the registry by identifier.
 *
 * @param identifier - The tool identifier
 * @returns The Tool object if found, undefined otherwise
 */
export function getTool(identifier: string): Tool<any> | undefined {
  return toolRegistry.get(identifier);
}

/**
 * Clear the tool registry (useful for testing).
 */
export function clearRegistry(): void {
  toolRegistry.clear();
}


