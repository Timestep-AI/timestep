/**
 * Monoidal category structures for agent composition.
 */

import { AgentCategory } from './categories';

export interface MonoidalCategory<T> {
  tensor(obj1: T, obj2: T): T;
  unit(): T;
  compose(f: T, g: T): T | null;
}

/**
 * Monoidal structure for agent composition.
 * 
 * - Tensor product: Parallel agent execution
 * - Unit: Empty agent (identity)
 * - Composition: Sequential agent workflows
 */
export class AgentComposition implements MonoidalCategory<string> {
  private parallelGroups: Map<string, string[]> = new Map();
  private readonly unitAgentId = '__unit__';

  constructor(private agentCategory: AgentCategory) {}

  tensor(agent1Id: string, agent2Id: string): string {
    const groupId = `parallel_${agent1Id}_${agent2Id}`;
    this.parallelGroups.set(groupId, [agent1Id, agent2Id]);
    return groupId;
  }

  unit(): string {
    return this.unitAgentId;
  }

  compose(f: string, g: string): string | null {
    // Use agent category composition for sequential handoffs
    return this.agentCategory.compose(f, g);
  }

  getParallelAgents(groupId: string): string[] {
    return this.parallelGroups.get(groupId) || [];
  }

  addParallelGroup(groupId: string, agentIds: string[]): void {
    this.parallelGroups.set(groupId, agentIds);
  }
}

