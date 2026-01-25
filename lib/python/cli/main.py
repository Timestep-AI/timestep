"""CLI tool for Timestep library."""

import asyncio
import click
import sys
from pathlib import Path

from timestep.core import Agent, Environment
from timestep.evals import verify_handoff, verify_tool_call, TraceEval


@click.group()
def cli():
    """Timestep CLI - Multi-agent systems with A2A and MCP protocols."""
    pass


@cli.command()
@click.option("--id", required=True, help="Agent ID")
@click.option("--name", required=True, help="Agent name")
@click.option("--model", default="gpt-4o-mini", help="OpenAI model to use")
@click.option("--port", default=8000, type=int, help="Port for A2A server")
@click.option("--context-id", multiple=True, help="Context ID (can specify multiple)")
@click.option("--environment-uri", multiple=True, help="Environment URI for each context (must match context-id count)")
@click.option("--trace-file", default="traces.jsonl", help="Trace file path")
@click.option("--human-in-loop", is_flag=True, help="Enable human-in-the-loop")
def agent_start(id, name, model, port, context_id, environment_uri, trace_file, human_in_loop):
    """Start an agent (A2A server)."""
    if len(context_id) != len(environment_uri):
        click.echo("Error: Number of context-id and environment-uri must match", err=True)
        sys.exit(1)
    
    context_id_to_environment_uri = dict(zip(context_id, environment_uri))
    
    agent = Agent(
        agent_id=id,
        name=name,
        model=model,
        context_id_to_environment_uri=context_id_to_environment_uri,
        human_in_loop=human_in_loop,
        trace_to_file=trace_file,
    )
    
    click.echo(f"Starting agent {id} on port {port}...")
    agent.run(port=port)


@cli.command()
@click.option("--id", required=True, help="Environment ID")
@click.option("--context-id", required=True, help="Context ID (maps to A2A context_id)")
@click.option("--agent-id", required=True, help="Agent ID this environment is for")
@click.option("--port", default=8080, type=int, help="Port for MCP server")
@click.option("--human-in-loop", is_flag=True, help="Enable human-in-the-loop")
def environment_start(id, context_id, agent_id, port, human_in_loop):
    """Start an environment (MCP server)."""
    environment = Environment(
        environment_id=id,
        context_id=context_id,
        agent_id=agent_id,
        human_in_loop=human_in_loop,
    )
    
    click.echo(f"Starting environment {id} on port {port}...")
    click.echo(f"Environment URI: http://localhost:{port}/mcp")
    click.echo("Note: You need to register tools and prompts on the environment before starting.")
    click.echo("See examples for how to do this.")
    
    asyncio.run(environment.run())


@cli.command()
@click.option("--trace-file", required=True, help="Trace file to evaluate")
@click.option("--eval-type", type=click.Choice(["handoff", "tool-call", "custom"]), required=True, help="Eval type")
@click.option("--from-agent", help="Source agent (for handoff eval)")
@click.option("--to-agent", help="Target agent (for handoff eval)")
@click.option("--agent-id", help="Agent ID (for tool-call eval)")
@click.option("--tool-name", help="Tool name (for tool-call eval)")
@click.option("--tool-arg", multiple=True, help="Tool argument (key=value format, can specify multiple)")
def eval_run(trace_file, eval_type, from_agent, to_agent, agent_id, tool_name, tool_arg):
    """Run evaluation on traces."""
    if not Path(trace_file).exists():
        click.echo(f"Error: Trace file not found: {trace_file}", err=True)
        sys.exit(1)
    
    if eval_type == "handoff":
        if not from_agent or not to_agent:
            click.echo("Error: --from-agent and --to-agent required for handoff eval", err=True)
            sys.exit(1)
        
        result = asyncio.run(verify_handoff(trace_file, from_agent, to_agent))
        click.echo(f"Handoff verified: {result}")
    
    elif eval_type == "tool-call":
        if not agent_id or not tool_name:
            click.echo("Error: --agent-id and --tool-name required for tool-call eval", err=True)
            sys.exit(1)
        
        # Parse tool arguments
        tool_args = {}
        for arg in tool_arg:
            if "=" in arg:
                key, value = arg.split("=", 1)
                tool_args[key] = value
        
        result = asyncio.run(verify_tool_call(trace_file, agent_id, tool_name, **tool_args))
        click.echo(f"Tool call verified: {result}")
    
    else:
        click.echo("Error: Custom eval type not yet implemented", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
