# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "a2a-sdk[http-server]",
#   "mcp",
#   "openai",
#   "fastapi",
#   "uvicorn",
#   "httpx",
#   "opentelemetry-sdk",
#   "opentelemetry-api",
#   "opentelemetry-exporter-otlp-proto-http",
#   "opentelemetry-exporter-otlp-proto-grpc",
#   "opentelemetry-instrumentation-openai-v2",
#   "opentelemetry-instrumentation-fastapi",
#   "opentelemetry-instrumentation-httpx",
# ]
# ///

"""Research Assistant Agent - A2A Server.

This script runs a Research Assistant Agent that connects to an Environment
via MCP to get system prompts and tools. It provides web search capabilities.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import uvicorn
import httpx
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

# Prevent A2A SDK from detecting OpenTelemetry by blocking the import
# A2A SDK checks for OpenTelemetry at import time - if available, it uses real telemetry
# We need OpenTelemetry for OpenAI tracing, but we don't want A2A SDK to use it
# Install the hook BEFORE executing timestep.core so A2A SDK sees OpenTelemetry as unavailable
import importlib.util
from importlib.machinery import ModuleSpec

class _BlockOpenTelemetryFinder:
    """Import finder that blocks opentelemetry imports."""
    def find_spec(self, name, path, target=None):
        if name.startswith('opentelemetry'):
            # Raise ImportError to prevent the import
            raise ImportError(f"Module '{name}' is blocked")
        return None

# Install the import hook BEFORE executing core module
_block_finder = _BlockOpenTelemetryFinder()
sys.meta_path.insert(0, _block_finder)

# Now import the core module which will set up timestep.core
# A2A SDK will see OpenTelemetry as unavailable and use NoOp telemetry
core_init_path = lib_python_dir / "core" / "__init__.py"
spec = importlib.util.spec_from_file_location("timestep.core", core_init_path)
core_module = importlib.util.module_from_spec(spec)
sys.modules['timestep.core'] = core_module
spec.loader.exec_module(core_module)

# Now import A2A SDK - it will see OpenTelemetry as unavailable and use NoOp
from timestep.core import Agent, Environment, Loop

# Remove the import hook so OpenTelemetry can be imported normally
sys.meta_path.remove(_block_finder)


def _extract_domain(url: str) -> str:
    """Extract domain from URL, handling subdomains."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        return domain.lower()
    except Exception:
        return ""


def _domain_matches(domain: str, allowed_domains: List[str]) -> bool:
    """Check if domain matches any allowed domain (including subdomains)."""
    domain = domain.lower()
    for allowed in allowed_domains:
        allowed = allowed.lower().strip()
        # Remove http/https prefix if present
        if allowed.startswith('http://') or allowed.startswith('https://'):
            allowed = urlparse(allowed).netloc
        # Remove port if present
        if ':' in allowed:
            allowed = allowed.split(':')[0]
        allowed = allowed.lower()
        
        # Exact match or subdomain match
        if domain == allowed or domain.endswith('.' + allowed):
            return True
    return False


def main():
    """Run the Research Assistant Agent."""
    
    # CRITICAL: Configure OpenTelemetry tracing FIRST, before any httpx clients are created
    # This ensures httpx instrumentation is active before A2A/MCP clients make HTTP requests
    from timestep.observability.tracing import configure_tracing
    configure_tracing()
    
    # Get port from environment variable or use default
    port = int(os.getenv("RESEARCH_ASSISTANT_AGENT_PORT", "10000"))
    host = "0.0.0.0"
    http_host = "localhost" if host == "0.0.0.0" else host
    
    # Set A2A_BASE_URL environment variable for agent card generation
    # This ensures the agent card has the correct URL
    # Always set it to match the port we're actually running on
    os.environ["A2A_BASE_URL"] = f"http://{http_host}:{port}"
    
    # Create Environment instance
    # Disable handoff tool for research assistant (it doesn't need handoff capability)
    environment = Environment(
        environment_id="research-assistant-env",
        agent_id="research-assistant",
        human_in_loop=False,
        enable_handoff=False,  # Disable handoff tool
    )
    
    # Add web_search tool to the environment (OpenAI-style interface)
    @environment.tool()
    async def web_search(
        query: str,
        max_results: int = 10,
        allowed_domains: Optional[List[str]] = None,
        user_location: Optional[Dict[str, Any]] = None,
        external_web_access: bool = True,
    ) -> Dict[str, Any]:
        """Performs a web search using Firecrawl.
        
        Matches OpenAI's web_search tool interface:
        - Supports domain filtering via allowed_domains
        - Returns results with URLs, titles, and content for citations
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 10)
            allowed_domains: Optional list of domains to filter results
            user_location: Optional (unused; for interface compatibility)
            external_web_access: Whether to use live search (default: True)
        
        Returns:
            Dict with action, queries, results (title, url, content), sources, and optionally domains.
        """
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return {
                "action": "search",
                "queries": [query],
                "error": "FIRECRAWL_API_KEY environment variable is not set",
                "results": [],
                "sources": [],
            }
        try:
            payload: Dict[str, Any] = {
                "query": query,
                "limit": max_results,
                "sources": ["web"],
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.firecrawl.dev/v2/search",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                raw = resp.json()
            web = raw.get("data", {}).get("web") if isinstance(raw.get("data"), dict) else raw.get("web")
            if not web:
                web = raw if isinstance(raw, list) else []

            results = []
            sources = []
            for item in web:
                url = item.get("url", "")
                title = item.get("title", "")
                content = item.get("description", "") or item.get("markdown", "") or ""
                if allowed_domains:
                    domain = _extract_domain(url)
                    if not _domain_matches(domain, allowed_domains):
                        continue
                results.append({"title": title, "url": url, "content": content})
                sources.append(url)

            response = {
                "action": "search",
                "queries": [query],
                "results": results,
                "sources": sources,
            }
            if allowed_domains:
                response["domains"] = allowed_domains
            return response

        except Exception as e:
            return {
                "action": "search",
                "queries": [query],
                "error": str(e),
                "results": [],
                "sources": [],
            }
    
    # Update system prompt for research assistant
    @environment.prompt()
    def system_prompt(agent_name: str) -> str:
        """System prompt for the research assistant agent."""
        return f"""You are {agent_name}, a helpful research assistant. You can perform web searches 
to find current information and answer research questions. Use the web_search tool to search for 
information on the internet. When providing answers, cite your sources using the URLs from the 
search results."""
    
    # Get MCP app from environment
    mcp_app = environment.streamable_http_app()
    
    # Create agent
    agent = Agent(
        agent_id="research-assistant",
        name="Research Assistant",
        environment_uri=f"http://{http_host}:{port}/mcp",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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
        title="Research Assistant Agent",
        lifespan=lifespan,
    )
    
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
        environment_uri=agent.environment_uri,
    )
    
    # Mount Loop routes
    for route in loop.fastapi_app.routes:
        combined_app.routes.append(route)
    
    # Instrument FastAPI app AFTER all routes are added
    # This ensures we only trace the final combined app, not individual route additions
    from timestep.observability.tracing import configure_tracing
    configure_tracing(app=combined_app)
    
    # All /v1/responses endpoint code has been moved to Loop
    # The endpoint is now registered via loop.fastapi_app above
    
    # Run combined app (blocking)
    print(f"Starting Research Assistant Agent on port {port}...")
    print(f"Environment mounted at: http://{http_host}:{port}/mcp")
    uvicorn.run(combined_app, host=host, port=port)


if __name__ == "__main__":
    main()
