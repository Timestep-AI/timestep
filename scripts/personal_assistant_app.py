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

"""Personal Assistant Agent - A2A Server.

This script runs a Personal Assistant Agent that connects to an Environment
via MCP to get system prompts and tools. It can hand off tasks to other agents.
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from uuid import uuid4
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

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
from timestep.utils.message_helpers import (
    extract_user_text_and_tool_results,
    TOOL_CALLS_KEY,
    TOOL_RESULTS_KEY,
)
from timestep.utils.event_helpers import extract_event_data, extract_task_from_tuple
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Message,
    Part,
    DataPart,
    Role,
)
from a2a.client.helpers import create_text_message_object
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH


def main():
    """Run the Personal Assistant Agent."""
    # Initialize OpenTelemetry tracing (if available)
    try:
        from timestep.observability.tracing import setup_tracing
        setup_tracing()
    except ImportError:
        pass  # Graceful degradation if OpenTelemetry not available
    
    # Get port from environment variable or use default
    port = int(os.getenv("PERSONAL_AGENT_PORT", "9999"))
    host = "0.0.0.0"
    http_host = "localhost" if host == "0.0.0.0" else host
    
    # Set A2A_BASE_URL environment variable for agent card generation
    # This ensures the agent card has the correct URL
    # Always set it to match the port we're actually running on
    os.environ["A2A_BASE_URL"] = f"http://{http_host}:{port}"
    
    # Create Environment instance
    # Handoff tool is registered by default in Environment (enable_handoff=True by default)
    environment = Environment(
        environment_id="personal-assistant-env",
        context_id="personal-context",
        agent_id="personal-assistant",
        human_in_loop=False,
        enable_handoff=True,  # Enable handoff tool (default)
    )
    
    # Update system prompt for personal assistant
    @environment.prompt()
    def system_prompt(agent_name: str) -> str:
        """System prompt for the personal assistant agent."""
        return f"""You are {agent_name}, a helpful personal assistant. 
You can help with various tasks. For weather-related questions, use the handoff tool 
to delegate to the weather assistant agent.

IMPORTANT: When using the handoff tool, you MUST use the exact agent_uri: http://localhost:10000
Do NOT use placeholder values like weather_agent or weather_service. 
The agent_uri must be the full URL: http://localhost:10000

Example handoff tool call:
- agent_uri: http://localhost:10000
- message: What is the current weather in Oakland?"""
    
    # Get MCP app from environment
    mcp_app = environment.streamable_http_app()
    
    # Create agent
    agent = Agent(
        agent_id="personal-assistant",
        name="Personal Assistant",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        context_id_to_environment_uri={
            "personal-context": f"http://{http_host}:{port}/mcp"
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
        title="Personal Assistant Agent",
        lifespan=lifespan,
    )
    
    # Instrument FastAPI app for tracing (if available)
    try:
        from timestep.observability.tracing import instrument_fastapi_app
        instrument_fastapi_app(combined_app)
    except ImportError:
        pass  # Graceful degradation if OpenTelemetry not available
    
    # Include all routes from the agent's FastAPI app
    for route in fastapi_app.routes:
        combined_app.routes.append(route)
    
    # Include all routes from the MCP app
    for route in mcp_app.routes:
        combined_app.routes.append(route)
    
    # Get agent base URL for Loop
    agent_base_url = f"http://{http_host}:{port}"
    
    # Create Loop instance
    # Handoff functionality is built-in (sampling callback auto-created)
    loop = Loop(
        agent=agent,
        agent_base_url=agent_base_url,
        context_id_to_environment_uri=agent.context_id_to_environment_uri,
    )
    
    # Mount Loop routes
    for route in loop.fastapi_app.routes:
        combined_app.routes.append(route)
    
    # All /v1/responses endpoint code has been moved to Loop
    # The endpoint is now registered via loop.fastapi_app above
    
    # Run combined app (blocking)
    print(f"Starting Personal Assistant Agent on port {port}...")
    print(f"Environment mounted at: http://{http_host}:{port}/mcp")
    uvicorn.run(combined_app, host=host, port=port)


if __name__ == "__main__":
    main()
