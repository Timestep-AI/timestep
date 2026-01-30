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
#   "opentelemetry-instrumentation-openai",
#   "ddgs",
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
from fastapi import FastAPI
from ddgs import DDGS

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


def _map_user_location_to_region(user_location: Optional[Dict[str, Any]]) -> str:
    """Map user_location dict to ddgs region string."""
    if not user_location:
        return "us-en"
    
    country = user_location.get("country", "").upper()
    # Map ISO country codes to ddgs regions
    region_map = {
        "US": "us-en",
        "GB": "uk-en",
        "CA": "ca-en",
        "AU": "au-en",
        "DE": "de-de",
        "FR": "fr-fr",
        "ES": "es-es",
        "IT": "it-it",
        "RU": "ru-ru",
        "JP": "jp-jp",
        "CN": "cn-zh",
    }
    return region_map.get(country, "us-en")


def main():
    """Run the Research Assistant Agent."""
    
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
        """Performs a web search using DDGS metasearch.
        
        Matches OpenAI's web_search tool interface:
        - Supports domain filtering via allowed_domains
        - Supports user location for geographic refinement
        - Returns results with URLs, titles, and content for citations
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 10)
            allowed_domains: Optional list of domains to filter results (e.g., ["pubmed.ncbi.nlm.nih.gov"])
            user_location: Optional dict with country, city, region, timezone for geographic refinement
            external_web_access: Whether to use live search (default: True, ddgs always uses live)
        
        Returns:
            Dict with:
            - action: "search" (matching OpenAI's web_search_call.action)
            - queries: List of search queries used
            - domains: List of domains searched (if domain filtering applied)
            - results: List of dicts with 'title', 'url', 'content'
            - sources: List of all URLs consulted
        """
        try:
            # Map user_location to ddgs region
            region = _map_user_location_to_region(user_location)
            
            # Perform search using DDGS
            ddgs = DDGS()
            search_results = ddgs.text(
                query=query,
                region=region,
                max_results=max_results,
                backend="auto"
            )
            
            # Convert ddgs results to our format
            results = []
            sources = []
            
            for result in search_results:
                title = result.get("title", "")
                href = result.get("href", "")
                body = result.get("body", "")
                
                # Apply domain filtering if specified
                if allowed_domains:
                    domain = _extract_domain(href)
                    if not _domain_matches(domain, allowed_domains):
                        continue
                
                results.append({
                    "title": title,
                    "url": href,
                    "content": body,
                })
                sources.append(href)
            
            # Build response matching OpenAI format
            response = {
                "action": "search",
                "queries": [query],
                "results": results,
                "sources": sources,
            }
            
            # Add domains if domain filtering was applied
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
    
    # Configure OpenTelemetry tracing
    from timestep.observability.tracing import configure_tracing
    configure_tracing()
    
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
    
    # All /v1/responses endpoint code has been moved to Loop
    # The endpoint is now registered via loop.fastapi_app above
    
    # Run combined app (blocking)
    print(f"Starting Research Assistant Agent on port {port}...")
    print(f"Environment mounted at: http://{http_host}:{port}/mcp")
    uvicorn.run(combined_app, host=host, port=port)


if __name__ == "__main__":
    main()
