/**
 * Olog (Ontology Log) builder for agent systems.
 */

import { loadAgent } from '../stores/agent_store/store';

export enum OlogRelationType {
  HAS = 'has',
  IS = 'is',
  USES = 'uses',
  DELEGATES_TO = 'delegates_to',
  FILTERS_WITH = 'filters_with',
  EXECUTES_IN = 'executes_in',
  REQUIRES = 'requires',
  PRODUCES = 'produces',
  MAINTAINS = 'maintains',
}

export interface OlogType {
  name: string;
  description?: string;
  examples?: string[];
}

export interface OlogAspect {
  source: string;
  target: string;
  relation: OlogRelationType;
  description?: string;
}

export class Olog {
  private types: Map<string, OlogType> = new Map();
  private aspects: OlogAspect[] = [];

  addType(ologType: OlogType): void {
    this.types.set(ologType.name, ologType);
  }

  addAspect(aspect: OlogAspect): void {
    this.aspects.push(aspect);
  }

  toMarkdown(): string {
    const lines: string[] = ['# Agent System Ontology', ''];

    // Document types
    lines.push('## Types', '');
    for (const [typeName, ologType] of this.types) {
      lines.push(`### ${typeName}`);
      if (ologType.description) {
        lines.push(ologType.description);
      }
      if (ologType.examples && ologType.examples.length > 0) {
        lines.push('');
        lines.push('**Examples:**');
        for (const example of ologType.examples) {
          lines.push(`- ${example}`);
        }
      }
      lines.push('');
    }

    // Document relationships
    lines.push('## Relationships', '');
    for (const aspect of this.aspects) {
      lines.push(
        `- **${aspect.source}** ${aspect.relation} **${aspect.target}**`
      );
      if (aspect.description) {
        lines.push(`  - ${aspect.description}`);
      }
    }
    lines.push('');

    return lines.join('\n');
  }

  toMermaid(): string {
    const lines: string[] = ['graph TD'];

    // Add type nodes
    for (const typeName of this.types.keys()) {
      lines.push(`  ${typeName}["${typeName}"]`);
    }

    // Add aspect edges
    for (const aspect of this.aspects) {
      lines.push(
        `  ${aspect.source} -->|${aspect.relation}| ${aspect.target}`
      );
    }

    return lines.join('\n');
  }
}

export class OlogBuilder {
  static async fromAgentSystem(agentIds: string[]): Promise<Olog> {
    const olog = new Olog();

    // Add Agent type
    olog.addType({
      name: 'Agent',
      description: 'An autonomous agent with instructions, tools, and handoffs',
      examples: ['WeatherAgent', 'Assistant', 'ResearchAgent'],
    });

    // Add Tool type
    olog.addType({
      name: 'Tool',
      description: 'A capability that an agent can use',
      examples: ['get_weather', 'WebSearchTool', 'CodeInterpreterTool'],
    });

    // Add Guardrail type
    olog.addType({
      name: 'Guardrail',
      description: 'Input or output filter for agents',
      examples: ['InputGuardrail', 'OutputGuardrail'],
    });

    // Add Workflow type
    olog.addType({
      name: 'Workflow',
      description: 'DBOS workflow for durable agent execution',
      examples: ['AgentWorkflow', 'ScheduledWorkflow'],
    });

    // Add Session type
    olog.addType({
      name: 'Session',
      description: 'Conversation state management',
      examples: ['OpenAIConversationsSession', 'SQLiteSession'],
    });

    // Process each agent
    for (const agentId of agentIds) {
      try {
        const agent = await loadAgent(agentId);

        // Agent has name
        olog.addAspect({
          source: 'Agent',
          target: 'Name',
          relation: OlogRelationType.HAS,
          description: `Every agent has a unique name (e.g., ${agent.name})`,
        });

        // Agent uses tools
        for (const tool of agent.tools || []) {
          const toolName = (tool as any).name || String(tool);
          olog.addAspect({
            source: 'Agent',
            target: 'Tool',
            relation: OlogRelationType.USES,
            description: `Agent ${agent.name} uses tool ${toolName}`,
          });
        }

        // Agent delegates to other agents
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
          olog.addAspect({
            source: 'Agent',
            target: 'Agent',
            relation: OlogRelationType.DELEGATES_TO,
            description: `Agent ${agent.name} can delegate to ${handoffName}`,
          });
        }

        // Agent filters with guardrails
        if ((agent as any).inputGuardrails && (agent as any).inputGuardrails.length > 0) {
          olog.addAspect({
            source: 'Agent',
            target: 'Guardrail',
            relation: OlogRelationType.FILTERS_WITH,
            description: `Agent ${agent.name} uses input guardrails`,
          });
        }

        if ((agent as any).outputGuardrails && (agent as any).outputGuardrails.length > 0) {
          olog.addAspect({
            source: 'Agent',
            target: 'Guardrail',
            relation: OlogRelationType.FILTERS_WITH,
            description: `Agent ${agent.name} uses output guardrails`,
          });
        }

        // Agent executes in workflow
        olog.addAspect({
          source: 'Agent',
          target: 'Workflow',
          relation: OlogRelationType.EXECUTES_IN,
          description: `Agent ${agent.name} can execute in DBOS workflows`,
        });

        // Agent maintains session
        olog.addAspect({
          source: 'Agent',
          target: 'Session',
          relation: OlogRelationType.MAINTAINS,
          description: `Agent ${agent.name} maintains conversation state`,
        });
      } catch (e) {
        // Skip agents that can't be loaded
        continue;
      }
    }

    return olog;
  }

  static async fromDatabaseSchema(): Promise<Olog> {
    const olog = new Olog();

    // Extract types from schema
    olog.addType({
      name: 'Agent',
      description: 'Stored agent definition with configuration',
    });
    olog.addType({
      name: 'Tool',
      description: 'Stored tool definition',
    });
    olog.addType({
      name: 'Guardrail',
      description: 'Input or output filter',
    });
    olog.addType({
      name: 'Workflow',
      description: 'DBOS workflow definition',
    });
    olog.addType({
      name: 'Session',
      description: 'Session state storage',
    });

    // Extract relationships from schema
    olog.addAspect({
      source: 'Agent',
      target: 'Tool',
      relation: OlogRelationType.USES,
      description: 'Many-to-many relationship via agent_tools table',
    });

    olog.addAspect({
      source: 'Agent',
      target: 'Guardrail',
      relation: OlogRelationType.FILTERS_WITH,
      description: 'Many-to-many relationship via agent_guardrails table',
    });

    olog.addAspect({
      source: 'Agent',
      target: 'Agent',
      relation: OlogRelationType.DELEGATES_TO,
      description: 'One-to-many relationship via agent_handoffs table',
    });

    return olog;
  }
}

export class OlogValidator {
  validate(olog: Olog): string[] {
    const issues: string[] = [];

    // Access private properties - in TypeScript we need to cast to access private members
    const ologAny = olog as any;
    const aspects = ologAny.aspects as OlogAspect[];
    const types = ologAny.types as Map<string, OlogType>;

    // Check all aspects reference valid types
    for (const aspect of aspects) {
      if (!types.has(aspect.source)) {
        issues.push(`Aspect references unknown type: ${aspect.source}`);
      }
      if (!types.has(aspect.target)) {
        issues.push(`Aspect references unknown type: ${aspect.target}`);
      }
    }

    // Check for isolated types (no relationships)
    for (const typeName of types.keys()) {
      const hasAspect = aspects.some(
        (a) => a.source === typeName || a.target === typeName
      );
      if (!hasAspect) {
        issues.push(`Type ${typeName} has no relationships`);
      }
    }

    return issues;
  }
}

