/**
 * String diagram representation for agent workflows.
 */

import { loadAgent } from '../stores/agent_store/store';

export enum DiagramNodeType {
  AGENT = 'agent',
  TOOL = 'tool',
  GUARDRAIL = 'guardrail',
  HANDOFF = 'handoff',
  STATE = 'state',
  INPUT = 'input',
  OUTPUT = 'output',
}

export interface DiagramNode {
  id: string;
  label: string;
  nodeType: DiagramNodeType;
  metadata?: Record<string, any>;
}

export interface DiagramEdge {
  source: string;
  target: string;
  label?: string;
  edgeType?: string; // data, handoff, state, approval
}

export class StringDiagram {
  nodes: DiagramNode[] = [];
  edges: DiagramEdge[] = [];

  addNode(node: DiagramNode): void {
    this.nodes.push(node);
  }

  addEdge(edge: DiagramEdge): void {
    this.edges.push(edge);
  }

  toMermaid(): string {
    const lines: string[] = ['graph LR'];

    // Add nodes
    for (const node of this.nodes) {
      lines.push(`  ${node.id}["${node.label}"]`);
    }

    // Add edges
    for (const edge of this.edges) {
      const label = edge.label ? `|"${edge.label}"|` : '';
      lines.push(`  ${edge.source} ${label}--> ${edge.target}`);
    }

    return lines.join('\n');
  }

  toSvg(): string {
    // Placeholder for actual SVG rendering
    // Would use graph layout algorithm (e.g., force-directed, hierarchical)
    return '<svg>...</svg>'; // Simplified for now
  }

  private getMermaidShape(nodeType: DiagramNodeType): string {
    const shapes: Record<DiagramNodeType, string> = {
      [DiagramNodeType.AGENT]: 'rect',
      [DiagramNodeType.TOOL]: 'round',
      [DiagramNodeType.GUARDRAIL]: 'diamond',
      [DiagramNodeType.HANDOFF]: 'hexagon',
      [DiagramNodeType.STATE]: 'cylinder',
      [DiagramNodeType.INPUT]: 'parallelogram',
      [DiagramNodeType.OUTPUT]: 'parallelogram',
    };
    return shapes[nodeType] || 'rect';
  }
}

export class DiagramBuilder {
  static async fromAgent(agentId: string): Promise<StringDiagram> {
    try {
      const agent = await loadAgent(agentId);
    } catch (e) {
      // Return empty diagram if agent can't be loaded
      return new StringDiagram();
    }

    const agent = await loadAgent(agentId);
    const diagram = new StringDiagram();

    // Add agent node
    const agentNode: DiagramNode = {
      id: `agent_${agentId}`,
      label: agent.name || agentId,
      nodeType: DiagramNodeType.AGENT,
      metadata: { agentId },
    };
    diagram.addNode(agentNode);

    // Add tool nodes
    for (let i = 0; i < (agent.tools || []).length; i++) {
      const tool = agent.tools![i];
      const toolName = (tool as any).name || String(tool);
      const toolNode: DiagramNode = {
        id: `tool_${agentId}_${i}`,
        label: toolName,
        nodeType: DiagramNodeType.TOOL,
        metadata: { toolIndex: i },
      };
      diagram.addNode(toolNode);
      diagram.addEdge({
        source: agentNode.id,
        target: toolNode.id,
        label: 'uses',
        edgeType: 'tool',
      });
    }

    // Add handoff nodes
    for (let i = 0; i < (agent.handoffs || []).length; i++) {
      const handoff = agent.handoffs![i];
      // Handoff can be Agent or Handoff object
      let handoffAgent: any;
      if ((handoff as any).agent) {
        handoffAgent = (handoff as any).agent;
      } else if ((handoff as any).name) {
        handoffAgent = handoff;
      } else {
        continue;
      }

      const handoffName = handoffAgent.name || String(handoffAgent);
      const handoffNode: DiagramNode = {
        id: `handoff_${agentId}_${i}`,
        label: handoffName,
        nodeType: DiagramNodeType.HANDOFF,
        metadata: { handoffIndex: i },
      };
      diagram.addNode(handoffNode);
      diagram.addEdge({
        source: agentNode.id,
        target: handoffNode.id,
        label: 'handoff',
        edgeType: 'handoff',
      });
    }

    // Add guardrail nodes
    const agentAny = agent as any;
    if (agentAny.inputGuardrails && agentAny.inputGuardrails.length > 0) {
      for (let i = 0; i < agentAny.inputGuardrails.length; i++) {
        const guardrail = agentAny.inputGuardrails[i];
        const guardrailName =
          guardrail.name || `InputGuardrail_${i}`;
        const guardrailNode: DiagramNode = {
          id: `input_guardrail_${agentId}_${i}`,
          label: guardrailName,
          nodeType: DiagramNodeType.GUARDRAIL,
          metadata: { guardrailType: 'input', index: i },
        };
        diagram.addNode(guardrailNode);
        diagram.addEdge({
          source: guardrailNode.id,
          target: agentNode.id,
          label: 'filters',
          edgeType: 'guardrail',
        });
      }
    }

    if (agentAny.outputGuardrails && agentAny.outputGuardrails.length > 0) {
      for (let i = 0; i < agentAny.outputGuardrails.length; i++) {
        const guardrail = agentAny.outputGuardrails[i];
        const guardrailName =
          guardrail.name || `OutputGuardrail_${i}`;
        const guardrailNode: DiagramNode = {
          id: `output_guardrail_${agentId}_${i}`,
          label: guardrailName,
          nodeType: DiagramNodeType.GUARDRAIL,
          metadata: { guardrailType: 'output', index: i },
        };
        diagram.addNode(guardrailNode);
        diagram.addEdge({
          source: agentNode.id,
          target: guardrailNode.id,
          label: 'filters',
          edgeType: 'guardrail',
        });
      }
    }

    return diagram;
  }

  static async fromWorkflow(workflowId: string): Promise<StringDiagram> {
    // Placeholder for workflow visualization
    // Would load workflow execution trace from DBOS
    // Convert to string diagram showing execution flow
    const diagram = new StringDiagram();

    // TODO: Implement workflow trace loading
    // Add workflow steps as nodes
    // Add state transitions as edges
    // Show parallel vs sequential execution

    return diagram;
  }
}

