# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "a2a-sdk[http-server,telemetry]",
#   "mcp",
#   "openai",
#   "fastapi",
#   "uvicorn",
#   "httpx",
#   "opentelemetry-sdk",
#   "opentelemetry-api",
#   "opentelemetry-exporter-otlp-proto-grpc",
#   "opentelemetry-instrumentation-fastapi",
#   "opentelemetry-instrumentation-httpx",
#   "opentelemetry-instrumentation-requests",
#   "opentelemetry-instrumentation-openai",
# ]
# ///

"""Weather Assistant Agent - A2A Server.

This script runs a Weather Assistant Agent that connects to an Environment
via MCP to get system prompts and tools.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI

# Fix package import: lib/python/ contains the timestep package
# but Python needs to import it as 'timestep', not 'python'
script_dir = Path(__file__).parent
lib_dir = script_dir.parent / "lib"
lib_python_dir = lib_dir / "python"

# Add lib/python to path
if str(lib_python_dir) not in sys.path:
    sys.path.insert(0, str(lib_python_dir))

# Create a 'timestep' module that points to the python directory
# This allows imports like 'from timestep.core import Agent' to work
import types
timestep_module = types.ModuleType('timestep')
timestep_module.__path__ = [str(lib_python_dir)]
sys.modules['timestep'] = timestep_module

# Now import the core module which will set up timestep.core
import importlib.util
core_init_path = lib_python_dir / "core" / "__init__.py"
spec = importlib.util.spec_from_file_location("timestep.core", core_init_path)
core_module = importlib.util.module_from_spec(spec)
sys.modules['timestep.core'] = core_module
spec.loader.exec_module(core_module)

# Now we can import Agent, Environment, and Loop
from timestep.core import Agent, Environment, Loop


def main():
    """Run the Weather Assistant Agent."""
    
    # Get port from environment variable or use default
    port = int(os.getenv("WEATHER_AGENT_PORT", "10000"))
    host = "0.0.0.0"
    http_host = "localhost" if host == "0.0.0.0" else host
    
    # Set A2A_BASE_URL environment variable for agent card generation
    # This ensures the agent card has the correct URL
    # Always set it to match the port we're actually running on
    os.environ["A2A_BASE_URL"] = f"http://{http_host}:{port}"
    
    # Create Environment instance
    # Disable handoff tool for weather assistant (it doesn't need handoff capability)
    environment = Environment(
        environment_id="weather-assistant-env",
        context_id="weather-context",
        agent_id="weather-assistant",
        human_in_loop=False,
        enable_handoff=False,  # Disable handoff tool
    )
    
    # Add get_weather tool to the environment
    @environment.tool()
    async def get_weather(location: str) -> Dict[str, Any]:
        """Get the current weather for a specific location."""
        # Return hardcoded weather data
        return {
            "location": location,
            "temperature": "72°F",
            "condition": "Sunny",
            "humidity": "65%"
        }
    
    # Update system prompt for weather assistant
    @environment.prompt()
    def system_prompt(agent_name: str) -> str:
        """System prompt for the weather assistant agent."""
        return f"You are {agent_name}, a helpful weather assistant. You can get weather information for any location using the get_weather tool."
    
    # Get MCP app from environment
    mcp_app = environment.streamable_http_app()
    
    # Create agent
    agent = Agent(
        agent_id="weather-assistant",
        name="Weather Assistant",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        context_id_to_environment_uri={
            "weather-context": f"http://{http_host}:{port}/mcp"
        },
        human_in_loop=False,
    )
    
    # Get FastAPI app from agent
    fastapi_app = agent.fastapi_app
    
    # Manually manage MCP task group (required for streamable HTTP)
    # This is what run_streamable_http_async() does internally
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: Create and initialize task group
        import anyio
        tg = anyio.create_task_group()
        await tg.__aenter__()
        app._mcp_task_group = tg
        
        # Set task group on session manager
        session_manager = environment.session_manager
        if hasattr(session_manager, '_task_group') and session_manager._task_group is None:
            session_manager._task_group = tg
        
        yield
        
        # Shutdown: Clean up task group
        if hasattr(app, '_mcp_task_group') and app._mcp_task_group:
            try:
                await app._mcp_task_group.__aexit__(None, None, None)
            except Exception:
                pass
    
    # Create a new FastAPI app with lifespan
    combined_app = FastAPI(
        title="Weather Assistant Agent",
        lifespan=lifespan,
    )
    
    # Enable OpenTelemetry tracing (if available)
    from timestep.observability.tracing import enable_tracing
    enable_tracing(combined_app)
    
    # Include all routes from the agent's FastAPI app
    for route in fastapi_app.routes:
        combined_app.routes.append(route)
    
    # Include all routes from the MCP app
    for route in mcp_app.routes:
        combined_app.routes.append(route)
    
    # Get agent base URL for Loop
    agent_base_url = f"http://{http_host}:{port}"
    
    # Create Loop instance
    loop = Loop(
        agent=agent,
        agent_base_url=agent_base_url,
        context_id_to_environment_uri=agent.context_id_to_environment_uri,
        sampling_callback=None,  # Weather assistant doesn't need handoffs
    )
    
    # Mount Loop routes
    for route in loop.fastapi_app.routes:
        combined_app.routes.append(route)
    
    # All /v1/responses endpoint code has been moved to Loop
    # The endpoint is now registered via loop.fastapi_app above
    
    # Run combined app (blocking)
    print(f"Starting Weather Assistant Agent on port {port}...")
    print(f"Environment mounted at: http://{http_host}:{port}/mcp")
    uvicorn.run(combined_app, host=host, port=port)


if __name__ == "__main__":
    main()
