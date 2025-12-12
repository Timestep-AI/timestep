"""CLI entry point for MCP server."""

import os

import click

from .server import run_server


@click.command()
@click.option("--host", default="0.0.0.0", help="Host address to bind to")
@click.option("--port", default=3000, type=int, help="Port number to listen on")
def main(host: str, port: int) -> None:
    """Start the Timestep MCP server."""
    # Allow environment variables to override defaults
    host = os.environ.get("MCP_HOST", host)
    port = int(os.environ.get("MCP_PORT", port))
    
    click.echo(f"Starting Timestep MCP server on {host}:{port}")
    click.echo(f"MCP endpoint: http://{host}:{port}/mcp")
    run_server(host=host, port=port)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]

