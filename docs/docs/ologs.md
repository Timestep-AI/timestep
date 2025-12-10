# Ologs (Ontology Logs)

## Overview

Ologs (ontology logs) are a diagrammatic way to represent knowledge and data, based on category theory. They provide a formal way to document the structure and relationships in multi-agent systems.

## Core Olog: Agent System

```
Agent ──has──> Name
Agent ──has──> Instructions
Agent ──has──> Model
Agent ──uses──> Tool
Agent ──delegates_to──> Agent (via handoff)
Agent ──filters_with──> Guardrail
Agent ──executes_in──> Workflow
Agent ──maintains──> Session
```

## Tool Olog

```
Tool ──has──> Type (hosted|function|mcp|agent)
Tool ──has──> Name
Tool ──has──> Parameters
Tool ──requires──> Approval (optional)
FunctionTool ──is──> Tool
HostedTool ──is──> Tool
MCPTool ──is──> Tool
AgentAsTool ──is──> Tool
```

## Workflow Olog

```
Workflow ──executes──> Agent
Workflow ──has──> State
Workflow ──can──> Resume (on interruption)
Workflow ──can──> Queue (for rate limiting)
Workflow ──can──> Schedule (cron-based)
```

## Handoff Olog

```
Handoff ──from──> Agent
Handoff ──to──> Agent
Handoff ──has──> Description
Handoff ──appears_as──> Tool (to LLM)
```

## Using Ologs

Ologs can be automatically generated from your agent definitions using the `OlogBuilder`:

```python
from timestep.analysis.olog import OlogBuilder

# Generate olog from agent system
olog = await OlogBuilder.from_agent_system(agent_ids)

# Export to Markdown
markdown = olog.to_markdown()

# Export to Mermaid diagram
mermaid = olog.to_mermaid()
```

## Benefits

- **Self-Documenting**: Automatically generated from code
- **Visual**: Can be rendered as diagrams
- **Formal**: Based on category theory foundations
- **Queryable**: Structure enables programmatic queries

## See Also

- [Category Theory](category-theory.md) - Mathematical foundations
- [String Diagrams](string-diagrams.md) - Visual representation
- [Generated Documentation](../generated/ontology.md) - Auto-generated olog documentation

