"""Tests for olog builder."""

import pytest
from timestep.analysis.olog import Olog, OlogType, OlogAspect, OlogRelationType, OlogValidator


def test_olog_add_type():
    """Test adding types to olog."""
    olog = Olog()
    olog_type = OlogType(
        name="Agent",
        description="An agent",
        examples=["Agent1", "Agent2"]
    )
    olog.add_type(olog_type)
    
    assert "Agent" in olog.types
    assert olog.types["Agent"].name == "Agent"


def test_olog_add_aspect():
    """Test adding aspects to olog."""
    olog = Olog()
    olog.add_type(OlogType(name="Agent", description="An agent"))
    olog.add_type(OlogType(name="Tool", description="A tool"))
    
    aspect = OlogAspect(
        source="Agent",
        target="Tool",
        relation=OlogRelationType.USES,
        description="Agent uses tool"
    )
    olog.add_aspect(aspect)
    
    assert len(olog.aspects) == 1
    assert olog.aspects[0].source == "Agent"
    assert olog.aspects[0].target == "Tool"


def test_olog_to_markdown():
    """Test olog to markdown conversion."""
    olog = Olog()
    olog.add_type(OlogType(name="Agent", description="An agent", examples=["Agent1"]))
    olog.add_type(OlogType(name="Tool", description="A tool"))
    olog.add_aspect(OlogAspect(
        source="Agent",
        target="Tool",
        relation=OlogRelationType.USES
    ))
    
    markdown = olog.to_markdown()
    assert "# Agent System Ontology" in markdown
    assert "Agent" in markdown
    assert "Tool" in markdown


def test_olog_to_mermaid():
    """Test olog to mermaid conversion."""
    olog = Olog()
    olog.add_type(OlogType(name="Agent", description="An agent"))
    olog.add_type(OlogType(name="Tool", description="A tool"))
    olog.add_aspect(OlogAspect(
        source="Agent",
        target="Tool",
        relation=OlogRelationType.USES
    ))
    
    mermaid = olog.to_mermaid()
    assert "graph TD" in mermaid
    assert "Agent" in mermaid
    assert "Tool" in mermaid


def test_olog_validator():
    """Test olog validator."""
    olog = Olog()
    olog.add_type(OlogType(name="Agent", description="An agent"))
    olog.add_aspect(OlogAspect(
        source="Agent",
        target="Tool",  # Tool type doesn't exist
        relation=OlogRelationType.USES
    ))
    
    validator = OlogValidator()
    issues = validator.validate(olog)
    
    # Should find that Tool type is missing
    assert len(issues) > 0
    assert any("Tool" in issue for issue in issues)

