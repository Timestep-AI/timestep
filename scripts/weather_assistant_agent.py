# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "a2a-sdk[http-server]",
#   "mcp",
#   "openai",
#   "fastapi",
#   "uvicorn",
#   "opentelemetry-api>=1.20.0",
#   "opentelemetry-sdk>=1.20.0",
# ]
# ///

"""Weather Assistant Agent - A2A Server.

This script runs a Weather Assistant Agent that connects to an Environment
via MCP to get system prompts and tools.
"""

import os
import sys
from pathlib import Path

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

# Now we can import Agent
from timestep.core import Agent


def main():
    """Run the Weather Assistant Agent."""
    # Get environment URI from environment variable or use default
    environment_uri = os.getenv(
        "WEATHER_ENVIRONMENT_URI",
        "http://localhost:8080/mcp"
    )
    
    # Get port from environment variable or use default
    port = int(os.getenv("WEATHER_AGENT_PORT", "9999"))
    
    # Get trace file from environment variable or use default
    trace_file = os.getenv(
        "WEATHER_AGENT_TRACE_FILE",
        "traces/weather_assistant.jsonl"
    )
    
    # Create agent
    agent = Agent(
        agent_id="weather-assistant",
        name="Weather Assistant",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        context_id_to_environment_uri={
            "weather-context": environment_uri
        },
        human_in_loop=False,
        trace_to_file=trace_file,
    )
    
    # Run agent (blocking)
    print(f"Starting Weather Assistant Agent on port {port}...")
    print(f"Environment URI: {environment_uri}")
    print(f"Trace file: {trace_file}")
    agent.run(port=port)


if __name__ == "__main__":
    main()
