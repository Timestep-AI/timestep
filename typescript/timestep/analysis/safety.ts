/**
 * Runtime safety checks for agent systems.
 */

import { loadAgent } from '../stores/agent_store/store';
import { DatabaseConnection } from '../stores/shared/db_connection';

export class CircularDependencyChecker {
  async checkCircularHandoffs(agentId: string): Promise<string[] | null> {
    const visited: Set<string> = new Set();
    const recStack: Set<string> = new Set();
    const path: string[] = [];

    const hasCycle = async (aid: string): Promise<string[] | null> => {
      visited.add(aid);
      recStack.add(aid);
      path.push(aid);

      try {
        await loadAgent(aid);
      } catch (e) {
        // Can't load agent, skip cycle check
        recStack.delete(aid);
        path.pop();
        return null;
      }

      // Get handoff agent IDs from database
      const connectionString = process.env.PG_CONNECTION_URI;
      if (!connectionString) {
        recStack.delete(aid);
        path.pop();
        return null; // Can't check without DB
      }

      const db = new DatabaseConnection({ connectionString });
      await db.connect();
      try {
        const result = await db.query(
          `
          SELECT handoff_agent_id FROM agent_handoffs
          WHERE agent_id = $1 AND handoff_agent_id IS NOT NULL
        `,
          [aid]
        );

        for (const row of result.rows) {
          const handoffAgentId = row.handoff_agent_id;
          if (!visited.has(handoffAgentId)) {
            const cycle = await hasCycle(handoffAgentId);
            if (cycle) {
              return cycle;
            }
          } else if (recStack.has(handoffAgentId)) {
            // Found cycle - return the cycle path
            const cycleStart = path.indexOf(handoffAgentId);
            return [...path.slice(cycleStart), handoffAgentId];
          }
        }
      } finally {
        await db.disconnect();
      }

      recStack.delete(aid);
      path.pop();
      return null;
    };

    return await hasCycle(agentId);
  }
}

export class ToolCompatibilityChecker {
  async checkCompatibility(agentId: string): Promise<string[]> {
    let agent;
    try {
      agent = await loadAgent(agentId);
    } catch (e) {
      return [`Could not load agent ${agentId}`];
    }

    const warnings: string[] = [];

    // Check if tool outputs match handoff input requirements
    // Note: This is a simplified check - actual type checking would be more complex
    for (const tool of agent.tools || []) {
      const toolName = (tool as any).name || String(tool);

      for (const handoff of agent.handoffs || []) {
        // Handoff can be Agent or Handoff object
        let handoffAgent: any;
        if ((handoff as any).agent) {
          handoffAgent = (handoff as any).agent;
        } else if ((handoff as any).name) {
          handoffAgent = handoff;
        } else {
          continue;
        }

        const handoffName =
          handoffAgent.name || String(handoffAgent);
        // This is a placeholder - actual implementation would need type system
        warnings.push(
          `Tool ${toolName} may not be compatible with handoff to ${handoffName}`
        );
      }
    }

    return warnings;
  }
}

export interface StateInvariant<T> {
  check(state: T): boolean;
  description(): string;
}

export class StateVerifier {
  constructor(private invariants: StateInvariant<any>[]) {}

  async verifyTransition(
    stateStore: any, // RunStateStore
    beforeStateId: string,
    afterStateId: string
  ): Promise<string[]> {
    let beforeState, afterState;
    try {
      beforeState = await stateStore.loadById(beforeStateId);
      afterState = await stateStore.loadById(afterStateId);
    } catch (e: any) {
      return [`Could not load states: ${e.message || e}`];
    }

    const violations: string[] = [];

    for (const invariant of this.invariants) {
      if (!invariant.check(beforeState)) {
        violations.push(
          `Invariant violated before: ${invariant.description()}`
        );
      }
      if (!invariant.check(afterState)) {
        violations.push(
          `Invariant violated after: ${invariant.description()}`
        );
      }
    }

    return violations;
  }
}

