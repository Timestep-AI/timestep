# String Diagrams

## Overview

String diagrams are a graphical calculus for monoidal categories, providing an intuitive way to visualize agent workflows, tool compositions, and data flow.

## Basic Agent Execution

```
[Input] ──> [Guardrail] ──> [Agent] ──> [Tool] ──> [Agent] ──> [Guardrail] ──> [Output]
```

## Agent Handoff Chain

```
[Agent A] ──handoff──> [Agent B] ──handoff──> [Agent C]
   │                      │                      │
   └──tool──> [Tool 1]    └──tool──> [Tool 2]    └──tool──> [Tool 3]
```

## Parallel Tool Execution

```
[Agent] ──┬──> [Tool 1] ──┐
          │                │
          ├──> [Tool 2] ──┼──> [Merge] ──> [Output]
          │                │
          └──> [Tool 3] ──┘
```

## Workflow with State Persistence

```
[Input] ──> [Load State] ──> [Agent Workflow] ──> [Save State] ──> [Output]
                ↑                                      │
                └──────────[Resume on Interrupt]──────┘
```

## Using String Diagrams

String diagrams can be generated from agent definitions:

```python
from timestep.visualizations.string_diagrams import DiagramBuilder
from timestep.visualizations.renderer import DiagramRenderer

# Build diagram from agent
builder = DiagramBuilder()
diagram = await builder.from_agent(agent_id)

# Render to various formats
renderer = DiagramRenderer()
mermaid = renderer.render(diagram, 'mermaid')
svg = renderer.render(diagram, 'svg')
json = renderer.render(diagram, 'json')
```

## CLI Tool

You can also use the CLI tool to generate diagrams:

```bash
# Python
python -m timestep.cli.visualize <agent_id> --format mermaid

# TypeScript
pnpm visualize <agent_id> --format mermaid
```

## Supported Formats

- **Mermaid**: For embedding in Markdown documentation
- **SVG**: For high-quality vector graphics
- **DOT**: For Graphviz rendering
- **JSON**: For programmatic processing and web visualization

## Benefits

- **Visual Understanding**: Quickly grasp complex agent architectures
- **Debugging**: Visualize execution traces
- **Documentation**: Include diagrams in documentation
- **Communication**: Share system design with stakeholders

## See Also

- [Category Theory](category-theory.md) - Mathematical foundations
- [Ologs](ologs.md) - Ontology representation
- [Generated Diagrams](../generated/agent-diagrams/) - Auto-generated visualizations

