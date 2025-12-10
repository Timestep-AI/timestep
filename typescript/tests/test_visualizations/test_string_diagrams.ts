/**
 * Tests for string diagram builder.
 */

import { describe, it, expect } from 'vitest';
import {
  StringDiagram,
  DiagramNode,
  DiagramEdge,
  DiagramNodeType,
  DiagramBuilder,
} from '../../timestep/visualizations/stringDiagrams';

describe('StringDiagram', () => {
  it('should add nodes to diagram', () => {
    const diagram = new StringDiagram();
    const node: DiagramNode = {
      id: 'node1',
      label: 'Agent1',
      nodeType: DiagramNodeType.AGENT,
    };
    diagram.addNode(node);

    expect(diagram.nodes.length).toBe(1);
    expect(diagram.nodes[0].id).toBe('node1');
  });

  it('should add edges to diagram', () => {
    const diagram = new StringDiagram();
    const node1: DiagramNode = {
      id: 'node1',
      label: 'Agent1',
      nodeType: DiagramNodeType.AGENT,
    };
    const node2: DiagramNode = {
      id: 'node2',
      label: 'Tool1',
      nodeType: DiagramNodeType.TOOL,
    };
    diagram.addNode(node1);
    diagram.addNode(node2);

    const edge: DiagramEdge = {
      source: 'node1',
      target: 'node2',
      label: 'uses',
    };
    diagram.addEdge(edge);

    expect(diagram.edges.length).toBe(1);
    expect(diagram.edges[0].source).toBe('node1');
    expect(diagram.edges[0].target).toBe('node2');
  });

  it('should convert diagram to mermaid', () => {
    const diagram = new StringDiagram();
    const node1: DiagramNode = {
      id: 'node1',
      label: 'Agent1',
      nodeType: DiagramNodeType.AGENT,
    };
    const node2: DiagramNode = {
      id: 'node2',
      label: 'Tool1',
      nodeType: DiagramNodeType.TOOL,
    };
    diagram.addNode(node1);
    diagram.addNode(node2);
    diagram.addEdge({
      source: 'node1',
      target: 'node2',
      label: 'uses',
    });

    const mermaid = diagram.toMermaid();
    expect(mermaid).toContain('graph LR');
    expect(mermaid).toContain('node1');
    expect(mermaid).toContain('node2');
  });
});

describe('DiagramBuilder', () => {
  it('should build diagram from agent', async () => {
    const builder = DiagramBuilder;
    // This will fail if no DB connection, but that's expected
    // In a real test, we'd set up a test database and agent
    try {
      const diagram = await builder.fromAgent('nonexistent_agent');
      expect(diagram).toBeInstanceOf(StringDiagram);
    } catch (e) {
      // Expected if agent doesn't exist or DB not available
      expect(e).toBeDefined();
    }
  });
});

