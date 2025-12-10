"""Renderer for string diagrams in various formats."""

from typing import Optional
import json
import asyncio
from .string_diagrams import StringDiagram, DiagramNodeType, DiagramBuilder


class DiagramRenderer:
    """Render string diagrams to various formats."""
    
    def __init__(self):
        self.renderers = {
            'mermaid': self._render_mermaid,
            'svg': self._render_svg,
            'dot': self._render_dot,
            'json': self._render_json,
        }
    
    def render(self, diagram: StringDiagram, format: str = 'mermaid') -> str:
        """
        Render diagram to specified format.
        
        Args:
            diagram: StringDiagram to render
            format: Output format ('mermaid', 'svg', 'dot', 'json')
            
        Returns:
            Rendered diagram as string
        """
        if format not in self.renderers:
            raise ValueError(f"Unsupported format: {format}")
        return self.renderers[format](diagram)
    
    def _render_mermaid(self, diagram: StringDiagram) -> str:
        """Render to Mermaid format."""
        return diagram.to_mermaid()
    
    def _render_svg(self, diagram: StringDiagram) -> str:
        """Render to SVG format."""
        return diagram.to_svg()
    
    def _render_dot(self, diagram: StringDiagram) -> str:
        """Render to Graphviz DOT format."""
        lines = ["digraph G {"]
        lines.append("  rankdir=LR;")
        
        for node in diagram.nodes:
            shape = self._get_dot_shape(node.node_type)
            lines.append(f'  {node.id} [label="{node.label}", shape={shape}];')
        
        for edge in diagram.edges:
            label = f' [label="{edge.label}"]' if edge.label else ""
            lines.append(f'  {edge.source} -> {edge.target}{label};')
        
        lines.append("}")
        return "\n".join(lines)
    
    def _render_json(self, diagram: StringDiagram) -> str:
        """Render to JSON for web visualization."""
        data = {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "type": node.node_type.value,
                    "metadata": node.metadata or {}
                }
                for node in diagram.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "type": edge.edge_type
                }
                for edge in diagram.edges
            ]
        }
        return json.dumps(data, indent=2)
    
    def _get_dot_shape(self, node_type: DiagramNodeType) -> str:
        """Get Graphviz shape for node type."""
        shapes = {
            DiagramNodeType.AGENT: "box",
            DiagramNodeType.TOOL: "ellipse",
            DiagramNodeType.GUARDRAIL: "diamond",
            DiagramNodeType.HANDOFF: "hexagon",
            DiagramNodeType.STATE: "cylinder",
            DiagramNodeType.INPUT: "parallelogram",
            DiagramNodeType.OUTPUT: "parallelogram",
        }
        return shapes.get(node_type, "box")


class WorkflowVisualizer:
    """Visualize workflow execution in real-time."""
    
    def __init__(self):
        self.diagram_builder = DiagramBuilder()
        self.renderer = DiagramRenderer()
    
    async def visualize_execution(
        self,
        workflow_id: str,
        format: str = 'mermaid'
    ) -> str:
        """
        Generate visualization of workflow execution.
        
        Can be called during or after execution.
        
        Args:
            workflow_id: Workflow ID to visualize
            format: Output format
            
        Returns:
            Rendered diagram as string
        """
        diagram = await self.diagram_builder.from_workflow(workflow_id)
        return self.renderer.render(diagram, format)
    
    async def stream_visualization_updates(
        self,
        workflow_id: str
    ):
        """
        Stream visualization updates as workflow executes.
        
        Yields updated diagram representations.
        
        Args:
            workflow_id: Workflow ID to visualize
            
        Yields:
            JSON representation of updated diagram
        """
        # Placeholder for actual workflow_running check
        def workflow_running(wf_id: str) -> bool:
            # In a real scenario, this would query DBOS for workflow status
            return True
        
        # Poll workflow state and update diagram
        while workflow_running(workflow_id):
            diagram = await self.diagram_builder.from_workflow(workflow_id)
            yield self.renderer.render(diagram, 'json')
            await asyncio.sleep(0.5)

