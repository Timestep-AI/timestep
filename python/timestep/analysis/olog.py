"""Olog (Ontology Log) builder for agent systems."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class OlogRelationType(Enum):
    """Types of relationships in an olog."""
    HAS = "has"
    IS = "is"
    USES = "uses"
    DELEGATES_TO = "delegates_to"
    FILTERS_WITH = "filters_with"
    EXECUTES_IN = "executes_in"
    REQUIRES = "requires"
    PRODUCES = "produces"
    MAINTAINS = "maintains"


@dataclass
class OlogType:
    """Represents a type (box) in an olog."""
    name: str
    description: Optional[str] = None
    examples: List[str] = field(default_factory=list)


@dataclass
class OlogAspect:
    """Represents an aspect (arrow) in an olog."""
    source: str
    target: str
    relation: OlogRelationType
    description: Optional[str] = None


class Olog:
    """Ontology log representation."""
    
    def __init__(self):
        self.types: Dict[str, OlogType] = {}
        self.aspects: List[OlogAspect] = []
    
    def add_type(self, olog_type: OlogType):
        """Add a type to the olog."""
        self.types[olog_type.name] = olog_type
    
    def add_aspect(self, aspect: OlogAspect):
        """Add an aspect (relationship) to the olog."""
        self.aspects.append(aspect)
    
    def to_markdown(self) -> str:
        """Generate Markdown documentation from olog."""
        lines = ["# Agent System Ontology", ""]
        
        # Document types
        lines.append("## Types")
        lines.append("")
        for type_name, olog_type in self.types.items():
            lines.append(f"### {type_name}")
            if olog_type.description:
                lines.append(f"{olog_type.description}")
            if olog_type.examples:
                lines.append("")
                lines.append("**Examples:**")
                for example in olog_type.examples:
                    lines.append(f"- {example}")
            lines.append("")
        
        # Document relationships
        lines.append("## Relationships")
        lines.append("")
        for aspect in self.aspects:
            lines.append(f"- **{aspect.source}** {aspect.relation.value} **{aspect.target}**")
            if aspect.description:
                lines.append(f"  - {aspect.description}")
        lines.append("")
        
        return "\n".join(lines)
    
    def to_mermaid(self) -> str:
        """Generate Mermaid diagram from olog."""
        lines = ["graph TD"]
        
        # Add type nodes
        for type_name in self.types.keys():
            lines.append(f'  {type_name}["{type_name}"]')
        
        # Add aspect edges
        for aspect in self.aspects:
            lines.append(f'  {aspect.source} -->|{aspect.relation.value}| {aspect.target}')
        
        return "\n".join(lines)


class OlogBuilder:
    """Build ologs from agent definitions."""
    
    @staticmethod
    async def from_agent_system(agent_ids: List[str]) -> Olog:
        """
        Build olog from a collection of agents.
        
        Args:
            agent_ids: List of agent IDs to include in the olog
            
        Returns:
            Olog instance
        """
        from ..stores.agent_store.store import load_agent
        
        olog = Olog()
        
        # Add Agent type
        olog.add_type(OlogType(
            name="Agent",
            description="An autonomous agent with instructions, tools, and handoffs",
            examples=["WeatherAgent", "Assistant", "ResearchAgent"]
        ))
        
        # Add Tool type
        olog.add_type(OlogType(
            name="Tool",
            description="A capability that an agent can use",
            examples=["get_weather", "WebSearchTool", "CodeInterpreterTool"]
        ))
        
        # Add Guardrail type
        olog.add_type(OlogType(
            name="Guardrail",
            description="Input or output filter for agents",
            examples=["InputGuardrail", "OutputGuardrail"]
        ))
        
        # Add Workflow type
        olog.add_type(OlogType(
            name="Workflow",
            description="DBOS workflow for durable agent execution",
            examples=["AgentWorkflow", "ScheduledWorkflow"]
        ))
        
        # Add Session type
        olog.add_type(OlogType(
            name="Session",
            description="Conversation state management",
            examples=["OpenAIConversationsSession", "SQLiteSession"]
        ))
        
        # Process each agent
        for agent_id in agent_ids:
            try:
                agent = await load_agent(agent_id)
                
                # Agent has name
                olog.add_aspect(OlogAspect(
                    source="Agent",
                    target="Name",
                    relation=OlogRelationType.HAS,
                    description=f"Every agent has a unique name (e.g., {agent.name})"
                ))
                
                # Agent uses tools
                for tool in agent.tools:
                    tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                    olog.add_aspect(OlogAspect(
                        source="Agent",
                        target="Tool",
                        relation=OlogRelationType.USES,
                        description=f"Agent {agent.name} uses tool {tool_name}"
                    ))
                
                # Agent delegates to other agents
                for handoff in agent.handoffs:
                    # Handoff can be Agent or Handoff object
                    if hasattr(handoff, 'agent'):
                        handoff_agent = handoff.agent
                    elif hasattr(handoff, 'name'):
                        handoff_agent = handoff
                    else:
                        continue
                    
                    handoff_name = handoff_agent.name if hasattr(handoff_agent, 'name') else str(handoff_agent)
                    olog.add_aspect(OlogAspect(
                        source="Agent",
                        target="Agent",
                        relation=OlogRelationType.DELEGATES_TO,
                        description=f"Agent {agent.name} can delegate to {handoff_name}"
                    ))
                
                # Agent filters with guardrails
                if hasattr(agent, 'input_guardrails') and agent.input_guardrails:
                    olog.add_aspect(OlogAspect(
                        source="Agent",
                        target="Guardrail",
                        relation=OlogRelationType.FILTERS_WITH,
                        description=f"Agent {agent.name} uses input guardrails"
                    ))
                
                if hasattr(agent, 'output_guardrails') and agent.output_guardrails:
                    olog.add_aspect(OlogAspect(
                        source="Agent",
                        target="Guardrail",
                        relation=OlogRelationType.FILTERS_WITH,
                        description=f"Agent {agent.name} uses output guardrails"
                    ))
                
                # Agent executes in workflow
                olog.add_aspect(OlogAspect(
                    source="Agent",
                    target="Workflow",
                    relation=OlogRelationType.EXECUTES_IN,
                    description=f"Agent {agent.name} can execute in DBOS workflows"
                ))
                
                # Agent maintains session
                olog.add_aspect(OlogAspect(
                    source="Agent",
                    target="Session",
                    relation=OlogRelationType.MAINTAINS,
                    description=f"Agent {agent.name} maintains conversation state"
                ))
                
            except Exception as e:
                # Skip agents that can't be loaded
                continue
        
        return olog
    
    @staticmethod
    async def from_database_schema() -> Olog:
        """
        Build olog from database schema.
        
        Returns:
            Olog instance based on database structure
        """
        olog = Olog()
        
        # Extract types from schema
        olog.add_type(OlogType(
            name="Agent",
            description="Stored agent definition with configuration"
        ))
        olog.add_type(OlogType(
            name="Tool",
            description="Stored tool definition"
        ))
        olog.add_type(OlogType(
            name="Guardrail",
            description="Input or output filter"
        ))
        olog.add_type(OlogType(
            name="Workflow",
            description="DBOS workflow definition"
        ))
        olog.add_type(OlogType(
            name="Session",
            description="Session state storage"
        ))
        
        # Extract relationships from schema
        olog.add_aspect(OlogAspect(
            source="Agent",
            target="Tool",
            relation=OlogRelationType.USES,
            description="Many-to-many relationship via agent_tools table"
        ))
        
        olog.add_aspect(OlogAspect(
            source="Agent",
            target="Guardrail",
            relation=OlogRelationType.FILTERS_WITH,
            description="Many-to-many relationship via agent_guardrails table"
        ))
        
        olog.add_aspect(OlogAspect(
            source="Agent",
            target="Agent",
            relation=OlogRelationType.DELEGATES_TO,
            description="One-to-many relationship via agent_handoffs table"
        ))
        
        return olog


class OlogValidator:
    """Validate olog consistency and completeness."""
    
    def validate(self, olog: Olog) -> List[str]:
        """
        Validate olog and return list of issues.
        
        Args:
            olog: Olog to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check all aspects reference valid types
        for aspect in olog.aspects:
            if aspect.source not in olog.types:
                issues.append(f"Aspect references unknown type: {aspect.source}")
            if aspect.target not in olog.types:
                issues.append(f"Aspect references unknown type: {aspect.target}")
        
        # Check for isolated types (no relationships)
        for type_name in olog.types.keys():
            has_aspect = any(
                a.source == type_name or a.target == type_name
                for a in olog.aspects
            )
            if not has_aspect:
                issues.append(f"Type {type_name} has no relationships")
        
        return issues

