"""CLI entry point for A2A server."""

import click

from .server import run_server


@click.command()
@click.option("--host", default="0.0.0.0", help="Host address to bind to")
@click.option("--port", default=8080, type=int, help="Port number to listen on")
@click.option("--model", default="gpt-4.1", help="OpenAI model name to use")
def main(host: str, port: int, model: str) -> None:
    """Start the Timestep A2A server."""
    click.echo(f"Starting Timestep A2A server on {host}:{port}")
    click.echo(f"Agent Card: http://{host}:{port}/.well-known/agent-card.json")
    run_server(host=host, port=port, model=model)


if __name__ == "__main__":
    main()

