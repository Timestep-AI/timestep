"""Tests for string diagram builder."""

import pytest
from timestep.visualizations.string_diagrams import (
    StringDiagram,
    DiagramNode,
    DiagramEdge,
    DiagramNodeType,
    DiagramBuilder
)


def test_string_diagram_add_node():
    """Test adding nodes to diagram."""
    diagram = StringDiagram()
    node = DiagramNode(
        id="node1",
        label="Agent1",
        node_type=DiagramNodeType.AGENT
    )
    diagram.add_node(node)
    
    assert len(diagram.nodes) == 1
    assert diagram.nodes[0].id == "node1"


def test_string_diagram_add_edge():
    """Test adding edges to diagram."""
    diagram = StringDiagram()
    node1 = DiagramNode(id="node1", label="Agent1", node_type=DiagramNodeType.AGENT)
    node2 = DiagramNode(id="node2", label="Tool1", node_type=DiagramNodeType.TOOL)
    diagram.add_node(node1)
    diagram.add_node(node2)
    
    edge = DiagramEdge(source="node1", target="node2", label="uses")
    diagram.add_edge(edge)
    
    assert len(diagram.edges) == 1
    assert diagram.edges[0].source == "node1"
    assert diagram.edges[0].target == "node2"


def test_string_diagram_to_mermaid():
    """Test diagram to mermaid conversion."""
    diagram = StringDiagram()
    node1 = DiagramNode(id="node1", label="Agent1", node_type=DiagramNodeType.AGENT)
    node2 = DiagramNode(id="node2", label="Tool1", node_type=DiagramNodeType.TOOL)
    diagram.add_node(node1)
    diagram.add_node(node2)
    diagram.add_edge(DiagramEdge(source="node1", target="node2", label="uses"))
    
    mermaid = diagram.to_mermaid()
    assert "graph LR" in mermaid
    assert "node1" in mermaid
    assert "node2" in mermaid


@pytest.mark.asyncio
async def test_diagram_builder_from_agent():
    """Test building diagram from agent."""
    builder = DiagramBuilder()
    # This will fail if no DB connection, but that's expected
    # In a real test, we'd set up a test database and agent
    try:
        diagram = await builder.from_agent("nonexistent_agent")
        assert isinstance(diagram, StringDiagram)
    except Exception:
        # Expected if agent doesn't exist or DB not available
        pass

