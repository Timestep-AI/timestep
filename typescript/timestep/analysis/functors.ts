/**
 * Functor implementations for agent systems.
 */

import { AgentCategory, ToolCategory } from './categories';

export interface Functor<T, U> {
  mapObject(obj: T): U;
  mapMorphism(morphism: T, source: T, target: T): U | null;
}

/**
 * Functor from Agent category to Tool category.
 * 
 * Maps agents to their available tools.
 */
export class AgentToolFunctor implements Functor<string, string[]> {
  private agentToTools: Map<string, string[]> = new Map();

  constructor(
    private agentCategory: AgentCategory,
    private toolCategory: ToolCategory
  ) {}

  mapObject(agentId: string): string[] {
    return this.agentToTools.get(agentId) || [];
  }

  addAgentTools(agentId: string, toolIds: string[]): void {
    this.agentToTools.set(agentId, toolIds);
  }

  mapMorphism(
    handoff: string,
    sourceAgent: string,
    targetAgent: string
  ): string[] | null {
    const sourceTools = this.mapObject(sourceAgent);
    const targetTools = this.mapObject(targetAgent);
    // Composition of tools: tools from both agents
    return [...sourceTools, ...targetTools];
  }
}

/**
 * Functor from Agent category to Agent category.
 * 
 * Maps agents to their handoff targets (delegation structure).
 */
export class HandoffFunctor implements Functor<string, string[]> {
  constructor(private agentCategory: AgentCategory) {}

  mapObject(agentId: string): string[] {
    // Get handoffs from agent category
    const handoffs: string[] = [];
    // Access private handoffs map - in real implementation would use public API
    return handoffs;
  }

  mapMorphism(
    handoff: string,
    sourceAgent: string,
    targetAgent: string
  ): string | null {
    const handoffs = this.mapObject(sourceAgent);
    if (handoffs.includes(targetAgent)) {
      return targetAgent;
    }
    return null;
  }
}

/**
 * Functor from Agent × Session category to RunState.
 * 
 * Maps agent-session pairs to execution states.
 */
export class StateFunctor implements Functor<[string, string], string | null> {
  private agentSessionStates: Map<string, string> = new Map();

  private key(agentId: string, sessionId: string): string {
    return `${agentId}:${sessionId}`;
  }

  mapObject(agentSession: [string, string]): string | null {
    const [agentId, sessionId] = agentSession;
    return this.agentSessionStates.get(this.key(agentId, sessionId)) || null;
  }

  addState(agentId: string, sessionId: string, stateId: string): void {
    this.agentSessionStates.set(this.key(agentId, sessionId), stateId);
  }

  mapMorphism(
    stateTransition: string,
    source: [string, string],
    target: [string, string]
  ): string | null {
    // State transitions preserve the state mapping
    return stateTransition;
  }
}

