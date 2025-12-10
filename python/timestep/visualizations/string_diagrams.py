"""String diagram representation for agent workflows."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class DiagramNodeType(Enum):
    """Types of nodes in a string diagram."""
    AGENT = "agent"
    TOOL = "tool"
    GUARDRAIL = "guardrail"
    HANDOFF = "handoff"
    STATE = "state"
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class DiagramNode:
    """Represents a node in a string diagram."""
    id: str
    label: str
    node_type: DiagramNodeType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagramEdge:
    """Represents an edge (wire) in a string diagram."""
    source: str
    target: str
    label: Optional[str] = None
    edge_type: str = "data"  # data, handoff, state, approval


class StringDiagram:
    """Core string diagram representation."""
    
    def __init__(self):
        self.nodes: List[DiagramNode] = []
        self.edges: List[DiagramEdge] = []
    
    def add_node(self, node: DiagramNode):
        """Add a node to the diagram."""
        self.nodes.append(node)
    
    def add_edge(self, edge: DiagramEdge):
        """Add an edge to the diagram."""
        self.edges.append(edge)
    
    def to_mermaid(self) -> str:
        """Export to Mermaid diagram format."""
        lines = ["graph LR"]
        
        # Add nodes
        for node in self.nodes:
            shape = self._get_mermaid_shape(node.node_type)
            lines.append(f'  {node.id}["{node.label}"]')
        
        # Add edges
        for edge in self.edges:
            label = f'|"{edge.label}"|' if edge.label else ""
            lines.append(f'  {edge.source} {label}--> {edge.target}')
        
        return "\n".join(lines)
    
    def to_svg(self) -> str:
        """Render to SVG using custom layout algorithm."""
        # Placeholder for actual SVG rendering
        # Would use graph layout algorithm (e.g., force-directed, hierarchical)
        return f"<svg>...</svg>"  # Simplified for now
    
    def _get_mermaid_shape(self, node_type: DiagramNodeType) -> str:
        """Get Mermaid shape for node type."""
        shapes = {
            DiagramNodeType.AGENT: "rect",
            DiagramNodeType.TOOL: "round",
            DiagramNodeType.GUARDRAIL: "diamond",
            DiagramNodeType.HANDOFF: "hexagon",
            DiagramNodeType.STATE: "cylinder",
            DiagramNodeType.INPUT: "parallelogram",
            DiagramNodeType.OUTPUT: "parallelogram",
        }
        return shapes.get(node_type, "rect")


class DiagramBuilder:
    """Build string diagrams from agent definitions and workflows."""
    
    @staticmethod
    async def from_agent(agent_id: str) -> StringDiagram:
        """
        Build diagram from a single agent.
        
        Args:
            agent_id: Agent ID to visualize
            
        Returns:
            StringDiagram instance
        """
        from ..stores.agent_store.store import load_agent
        
        try:
            agent = await load_agent(agent_id)
        except Exception:
            # Return empty diagram if agent can't be loaded
            return StringDiagram()
        
        diagram = StringDiagram()
        
        # Add agent node
        agent_node = DiagramNode(
            id=f"agent_{agent_id}",
            label=agent.name if hasattr(agent, 'name') else agent_id,
            node_type=DiagramNodeType.AGENT,
            metadata={"agent_id": agent_id}
        )
        diagram.add_node(agent_node)
        
        # Add tool nodes
        for i, tool in enumerate(agent.tools):
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            tool_node = DiagramNode(
                id=f"tool_{agent_id}_{i}",
                label=tool_name,
                node_type=DiagramNodeType.TOOL,
                metadata={"tool_index": i}
            )
            diagram.add_node(tool_node)
            diagram.add_edge(DiagramEdge(
                source=agent_node.id,
                target=tool_node.id,
                label="uses",
                edge_type="tool"
            ))
        
        # Add handoff nodes
        for i, handoff in enumerate(agent.handoffs):
            # Handoff can be Agent or Handoff object
            if hasattr(handoff, 'agent'):
                handoff_agent = handoff.agent
            elif hasattr(handoff, 'name'):
                handoff_agent = handoff
            else:
                continue
            
            handoff_name = handoff_agent.name if hasattr(handoff_agent, 'name') else str(handoff_agent)
            handoff_node = DiagramNode(
                id=f"handoff_{agent_id}_{i}",
                label=handoff_name,
                node_type=DiagramNodeType.HANDOFF,
                metadata={"handoff_index": i}
            )
            diagram.add_node(handoff_node)
            diagram.add_edge(DiagramEdge(
                source=agent_node.id,
                target=handoff_node.id,
                label="handoff",
                edge_type="handoff"
            ))
        
        # Add guardrail nodes
        if hasattr(agent, 'input_guardrails') and agent.input_guardrails:
            for i, guardrail in enumerate(agent.input_guardrails):
                guardrail_name = guardrail.name if hasattr(guardrail, 'name') else f"InputGuardrail_{i}"
                guardrail_node = DiagramNode(
                    id=f"input_guardrail_{agent_id}_{i}",
                    label=guardrail_name,
                    node_type=DiagramNodeType.GUARDRAIL,
                    metadata={"guardrail_type": "input", "index": i}
                )
                diagram.add_node(guardrail_node)
                diagram.add_edge(DiagramEdge(
                    source=guardrail_node.id,
                    target=agent_node.id,
                    label="filters",
                    edge_type="guardrail"
                ))
        
        if hasattr(agent, 'output_guardrails') and agent.output_guardrails:
            for i, guardrail in enumerate(agent.output_guardrails):
                guardrail_name = guardrail.name if hasattr(guardrail, 'name') else f"OutputGuardrail_{i}"
                guardrail_node = DiagramNode(
                    id=f"output_guardrail_{agent_id}_{i}",
                    label=guardrail_name,
                    node_type=DiagramNodeType.GUARDRAIL,
                    metadata={"guardrail_type": "output", "index": i}
                )
                diagram.add_node(guardrail_node)
                diagram.add_edge(DiagramEdge(
                    source=agent_node.id,
                    target=guardrail_node.id,
                    label="filters",
                    edge_type="guardrail"
                ))
        
        return diagram
    
    @staticmethod
    async def from_workflow(workflow_id: str) -> StringDiagram:
        """
        Build diagram from a DBOS workflow execution.
        
        Args:
            workflow_id: Workflow ID to visualize
            
        Returns:
            StringDiagram instance
        """
        # Placeholder for workflow visualization
        # Would load workflow execution trace from DBOS
        # Convert to string diagram showing execution flow
        diagram = StringDiagram()
        
        # TODO: Implement workflow trace loading
        # Add workflow steps as nodes
        # Add state transitions as edges
        # Show parallel vs sequential execution
        
        return diagram

