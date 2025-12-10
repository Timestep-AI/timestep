"""Documentation generator for agent systems."""

import asyncio
from pathlib import Path
from typing import List
from timestep.analysis.olog import OlogBuilder
from timestep.visualizations.string_diagrams import DiagramBuilder
from timestep.visualizations.renderer import DiagramRenderer


class DocumentationGenerator:
    """Generate comprehensive documentation from code."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize documentation generator.
        
        Args:
            output_dir: Directory to write generated documentation
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_all(self, agent_ids: List[str]):
        """
        Generate all documentation artifacts.
        
        Args:
            agent_ids: List of agent IDs to document
        """
        # Generate olog documentation
        olog = await OlogBuilder.from_agent_system(agent_ids)
        await self._write_olog_docs(olog)
        
        # Generate string diagram visualizations
        for agent_id in agent_ids:
            await self._write_agent_diagram(agent_id)
    
    async def _write_olog_docs(self, olog):
        """
        Write olog documentation to files.
        
        Args:
            olog: Olog instance to document
        """
        # Markdown documentation
        md_path = self.output_dir / "ontology.md"
        with open(md_path, 'w') as f:
            f.write(olog.to_markdown())
        
        # Mermaid diagram
        mermaid_path = self.output_dir / "ontology.mmd"
        with open(mermaid_path, 'w') as f:
            f.write(olog.to_mermaid())
    
    async def _write_agent_diagram(self, agent_id: str):
        """
        Write agent string diagram.
        
        Args:
            agent_id: Agent ID to visualize
        """
        builder = DiagramBuilder()
        diagram = await builder.from_agent(agent_id)
        
        renderer = DiagramRenderer()
        
        # Write multiple formats
        for fmt in ['mermaid', 'svg', 'json']:
            output = renderer.render(diagram, fmt)
            ext = 'mmd' if fmt == 'mermaid' else fmt
            path = self.output_dir / f"agent_{agent_id}.{ext}"
            with open(path, 'w') as f:
                f.write(output)

