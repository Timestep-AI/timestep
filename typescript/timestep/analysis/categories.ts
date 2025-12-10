/**
 * Category definitions for agent systems.
 */

export interface Category<T> {
  compose(f: T, g: T): T | null;
  identity(obj: T): T;
  objects(): T[];
  morphisms(source: T, target: T): T[][];
}

/**
 * Category of agents.
 * 
 * Objects: Individual agents
 * Morphisms: Handoffs between agents
 */
export class AgentCategory implements Category<string> {
  private agents: Map<string, any> = new Map();
  private handoffs: Map<string, string[]> = new Map();

  addAgent(agentId: string, agent: any): void {
    this.agents.set(agentId, agent);
    if (!this.handoffs.has(agentId)) {
      this.handoffs.set(agentId, []);
    }
  }

  addHandoff(sourceId: string, targetId: string): void {
    const targets = this.handoffs.get(sourceId) || [];
    if (!targets.includes(targetId)) {
      targets.push(targetId);
      this.handoffs.set(sourceId, targets);
    }
  }

  compose(f: string, g: string): string | null {
    // Check if f can handoff to g
    const fHandoffs = this.handoffs.get(f);
    if (fHandoffs && fHandoffs.includes(g)) {
      return g;
    }
    // Check if there's a path f -> intermediate -> g
    if (fHandoffs) {
      for (const intermediate of fHandoffs) {
        const intermediateHandoffs = this.handoffs.get(intermediate);
        if (intermediateHandoffs && intermediateHandoffs.includes(g)) {
          return g;
        }
      }
    }
    return null;
  }

  identity(obj: string): string {
    return obj;
  }

  objects(): string[] {
    return Array.from(this.agents.keys());
  }

  morphisms(source: string, target: string): string[][] {
    const paths: string[][] = [];

    const findPaths = (
      current: string,
      target: string,
      visited: Set<string>,
      path: string[]
    ): void => {
      if (current === target) {
        paths.push([...path]);
        return;
      }
      if (visited.has(current)) {
        return;
      }
      visited.add(current);
      const handoffs = this.handoffs.get(current) || [];
      for (const nextAgent of handoffs) {
        path.push(nextAgent);
        findPaths(nextAgent, target, visited, path);
        path.pop();
      }
      visited.delete(current);
    };

    findPaths(source, target, new Set(), []);
    return paths;
  }
}

/**
 * Category of tools.
 * 
 * Objects: Individual tools
 * Morphisms: Tool invocations (Tool -> Result)
 */
export class ToolCategory implements Category<string> {
  private tools: Map<string, any> = new Map();
  private invocations: Map<string, string[]> = new Map();

  addTool(toolId: string, tool: any): void {
    this.tools.set(toolId, tool);
  }

  addInvocation(toolId: string, resultType: string): void {
    const results = this.invocations.get(toolId) || [];
    if (!results.includes(resultType)) {
      results.push(resultType);
      this.invocations.set(toolId, results);
    }
  }

  compose(f: string, g: string): string | null {
    // Simplified: if f's output type matches g's input type, they compose
    // This is a placeholder - actual implementation would need type checking
    const fResults = this.invocations.get(f);
    if (fResults && fResults.length > 0 && this.tools.has(g)) {
      return fResults[0];
    }
    return null;
  }

  identity(obj: string): string {
    return obj;
  }

  objects(): string[] {
    return Array.from(this.tools.keys());
  }

  morphisms(source: string, target: string): string[][] {
    // Simplified implementation
    const sourceResults = this.invocations.get(source);
    if (sourceResults && sourceResults.includes(target)) {
      return [[target]];
    }
    return [];
  }
}

