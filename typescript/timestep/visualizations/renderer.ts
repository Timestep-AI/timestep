/**
 * Renderer for string diagrams in various formats.
 */

import { StringDiagram, DiagramNodeType, DiagramBuilder } from './stringDiagrams';

export class DiagramRenderer {
  private renderers: Record<
    string,
    (diagram: StringDiagram) => string
  > = {
    mermaid: this.renderMermaid.bind(this),
    svg: this.renderSvg.bind(this),
    dot: this.renderDot.bind(this),
    json: this.renderJson.bind(this),
  };

  render(diagram: StringDiagram, format: string = 'mermaid'): string {
    if (!(format in this.renderers)) {
      throw new Error(`Unsupported format: ${format}`);
    }
    return this.renderers[format](diagram);
  }

  private renderMermaid(diagram: StringDiagram): string {
    return diagram.toMermaid();
  }

  private renderSvg(diagram: StringDiagram): string {
    return diagram.toSvg();
  }

  private renderDot(diagram: StringDiagram): string {
    const lines: string[] = ['digraph G {', '  rankdir=LR;'];

    for (const node of diagram.nodes) {
      const shape = this.getDotShape(node.nodeType);
      lines.push(
        `  ${node.id} [label="${node.label}", shape=${shape}];`
      );
    }

    for (const edge of diagram.edges) {
      const label = edge.label ? ` [label="${edge.label}"]` : '';
      lines.push(`  ${edge.source} -> ${edge.target}${label};`);
    }

    lines.push('}');
    return lines.join('\n');
  }

  private renderJson(diagram: StringDiagram): string {
    const data = {
      nodes: diagram.nodes.map((node) => ({
        id: node.id,
        label: node.label,
        type: node.nodeType,
        metadata: node.metadata || {},
      })),
      edges: diagram.edges.map((edge) => ({
        source: edge.source,
        target: edge.target,
        label: edge.label,
        type: edge.edgeType,
      })),
    };
    return JSON.stringify(data, null, 2);
  }

  private getDotShape(nodeType: DiagramNodeType): string {
    const shapes: Record<DiagramNodeType, string> = {
      [DiagramNodeType.AGENT]: 'box',
      [DiagramNodeType.TOOL]: 'ellipse',
      [DiagramNodeType.GUARDRAIL]: 'diamond',
      [DiagramNodeType.HANDOFF]: 'hexagon',
      [DiagramNodeType.STATE]: 'cylinder',
      [DiagramNodeType.INPUT]: 'parallelogram',
      [DiagramNodeType.OUTPUT]: 'parallelogram',
    };
    return shapes[nodeType] || 'box';
  }
}

export class WebDiagramRenderer {
  private svg: any; // d3.Selection<SVGSVGElement, unknown, null, undefined>

  constructor(containerId: string) {
    // This would use d3 in a real implementation
    // For now, just store the container ID
    this.svg = { containerId };
  }

  render(diagram: StringDiagram): void {
    // Placeholder for D3-based rendering
    // In a real implementation, this would:
    // 1. Create force-directed layout
    // 2. Draw edges
    // 3. Draw nodes
    // 4. Add labels
    // 5. Update positions on simulation tick
  }

  private getNodeColor(nodeType: string): string {
    const colors: Record<string, string> = {
      agent: '#4A90E2',
      tool: '#50C878',
      guardrail: '#FF6B6B',
      handoff: '#FFA500',
    };
    return colors[nodeType] || '#CCCCCC';
  }
}

export class WorkflowVisualizer {
  private diagramBuilder = DiagramBuilder;
  private renderer = new DiagramRenderer();

  async visualizeExecution(
    workflowId: string,
    format: string = 'mermaid'
  ): Promise<string> {
    const diagram = await this.diagramBuilder.fromWorkflow(workflowId);
    return this.renderer.render(diagram, format);
  }

  async *streamVisualizationUpdates(workflowId: string): AsyncGenerator<string> {
    // Placeholder for actual workflow_running check
    const workflowRunning = (wfId: string): boolean => {
      // In a real scenario, this would query DBOS for workflow status
      return true;
    };

    // Poll workflow state and update diagram
    while (workflowRunning(workflowId)) {
      const diagram = await this.diagramBuilder.fromWorkflow(workflowId);
      yield this.renderer.render(diagram, 'json');
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
}

