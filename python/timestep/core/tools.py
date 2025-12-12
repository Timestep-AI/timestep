"""Common tools for agents, including web search using Firecrawl."""

import asyncio
import os
from typing import Any, Callable, Literal
from urllib.parse import urlparse

from firecrawl import Firecrawl
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field


def _get_firecrawl_client() -> Firecrawl:
    """Get or create Firecrawl client instance."""
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError(
            "FIRECRAWL_API_KEY environment variable is required for web search. "
            "Please set it to your Firecrawl API key."
        )
    return Firecrawl(api_key=api_key)


def _map_search_context_size_to_limit(search_context_size: str) -> int:
    """Map search_context_size to Firecrawl limit parameter."""
    mapping = {
        "low": 5,
        "medium": 10,
        "high": 20,
    }
    return mapping.get(search_context_size, 10)


def _extract_domain(url: str) -> str:
    """Extract and normalize domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix for comparison
        if domain.startswith("www."):
            domain = domain[4:]
    except Exception:
        domain = ""
    return domain


def _matches_domain(url: str, allowed_domains: list[str]) -> bool:
    """Check if URL's domain matches any allowed domain."""
    domain = _extract_domain(url)
    allowed_domains_set = [
        d.lower().strip().replace("www.", "")
        for d in allowed_domains
        if d
    ]
    allowed_domains_set = [d for d in allowed_domains_set if d]

    return any(
        domain == allowed_domain or domain.endswith("." + allowed_domain)
        for allowed_domain in allowed_domains_set
    )


class GetWeather(BaseModel):
    """Returns weather info for the specified city."""
    city: str = Field(..., description="The city name")


GetWeather.__name__ = "get_weather"


async def call_mcp_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Call a tool on the MCP server.

    Args:
        tool_name: Name of the tool to call
        args: Tool arguments

    Returns:
        Tool result as string

    Raises:
        ValueError: If the MCP tool call fails
    """
    mcp_server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:3000/mcp")

    async with streamablehttp_client(mcp_server_url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)

            # Extract text content from result
            text_parts: list[str] = []
            for item in result.content:
                if hasattr(item, "text"):
                    text_parts.append(item.text)

            return "\n".join(text_parts) if text_parts else "Tool executed successfully (no text output)"


class WebSearchFilters(BaseModel):
    """Filters for web search."""
    allowed_domains: list[str] | None = Field(None, description="List of allowed domains to filter search results")


class WebSearch(BaseModel):
    """A tool that lets the LLM search the web using Firecrawl.

    Args:
        query: The search query string.
        user_location: Optional location for the search. Lets you customize results to be relevant to a location.
        filters: A filter to apply. Should support 'allowed_domains' list.
        search_context_size: The amount of context to use for the search. One of 'low', 'medium', or 'high'. 'medium' is the default.
    """
    query: str = Field(..., description="The search query string")
    user_location: str | None = Field(None, description="Optional location for the search. Lets you customize results to be relevant to a location")
    filters: WebSearchFilters | None = Field(None, description="A filter to apply. Should support 'allowed_domains' list")
    search_context_size: Literal["low", "medium", "high"] = Field("medium", description="The amount of context to use for the search. One of 'low', 'medium', or 'high'. 'medium' is the default")


WebSearch.__name__ = "web_search"


class Handoff(BaseModel):
    """Hand off a message to another agent via A2A protocol."""

    message: str = Field(..., description="The message to send to the target agent")
    agent_id: str = Field(..., description="The ID of the agent to hand off to (e.g., 'weather-assistant', 'personal-assistant')", alias="agentId")


Handoff.__name__ = "handoff"


async def web_search(args: dict[str, Any]) -> str:
    """Execute web search tool."""
    try:
        client = _get_firecrawl_client()

        # Map search_context_size to limit
        search_context_size = args.get("search_context_size", "medium")
        limit = _map_search_context_size_to_limit(search_context_size)

        # Prepare search parameters
        search_params = {
            "query": args["query"],
            "limit": limit,
        }

        # Add location if provided
        if args.get("user_location"):
            search_params["location"] = args["user_location"]

        # Perform search
        results = client.search(**search_params)

        # Extract web results
        web_results = results.get("data", {}).get("web", [])

        # Filter by allowed domains if specified
        filters = args.get("filters")
        if filters and isinstance(filters, dict):
            allowed_domains = filters.get("allowed_domains")
            if allowed_domains and isinstance(allowed_domains, list):
                allowed_domains_list = [d for d in allowed_domains if d]
                if allowed_domains_list:
                    filtered_results = [
                        result
                        for result in web_results
                        if _matches_domain(result.get("url", ""), allowed_domains_list)
                    ]
                    web_results = filtered_results

        # Format results
        if not web_results:
            return "No search results found."

        formatted_results = []
        for i, result in enumerate(web_results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            description = result.get("description", "No description available")
            formatted_results.append(f"{i}. {title}\n   URL: {url}\n   {description}")

        return "\n\n".join(formatted_results)

    except ValueError:
        # Re-raise ValueError (e.g., missing API key)
        raise
    except Exception as exc:
        return f"Error performing web search: {exc!s}"


async def call_function(
    name: str,
    args: dict[str, Any],
    on_approval_required: Callable[[dict[str, Any]], Any] | None = None,
    source_context_id: str | None = None,
    on_child_message: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Call function that maps tool names to execute functions.

    Args:
        name: Tool name
        args: Tool arguments
        on_approval_required: Optional callback for tool approvals (used by handoff)
        source_context_id: Optional source context ID for handoff tool
        on_child_message: Optional callback for child messages from handoff

    Returns:
        Tool result as string
    """
    # Route MCP tools to MCP server
    if name == "get_weather":
        return await call_mcp_tool(name, args)
    if name == "web_search":
        return await web_search(args)
    if name == "handoff":
        # Lazy import to avoid circular dependency
        from ..a2a.handoff import HandoffCallbacks, handoff as handoff_tool
        
        return await handoff_tool(
            {
                **args,
                "source_context_id": source_context_id,
            },
            HandoffCallbacks(
                on_approval_required=on_approval_required,
                on_child_message=on_child_message,
            ),
        )
    raise ValueError(f"Unknown tool: {name}")


# Export tools and call_function
__all__ = ["GetWeather", "WebSearch", "Handoff", "call_function"]

