"""CLI tool for visualizing agent systems."""

import asyncio
import argparse
from timestep.visualizations.string_diagrams import DiagramBuilder
from timestep.visualizations.renderer import DiagramRenderer


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Visualize agent systems')
    parser.add_argument('agent_id', help='Agent ID to visualize')
    parser.add_argument('--format', choices=['mermaid', 'svg', 'dot', 'json'], 
                       default='mermaid', help='Output format')
    parser.add_argument('--output', help='Output file path (default: stdout)')
    
    args = parser.parse_args()
    
    builder = DiagramBuilder()
    diagram = await builder.from_agent(args.agent_id)
    
    renderer = DiagramRenderer()
    output = renderer.render(diagram, args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)


if __name__ == '__main__':
    asyncio.run(main())

