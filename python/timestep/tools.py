"""Common tools for agents, including web search using Firecrawl."""

import os
from typing import Any, Literal, Optional
from agents import function_tool
from firecrawl import Firecrawl


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


@function_tool
def web_search(
    query: str,
    user_location: Optional[str] = None,
    filters: Optional[Any] = None,
    search_context_size: Literal["low", "medium", "high"] = "medium",
) -> str:
    """A tool that lets the LLM search the web using Firecrawl.
    
    Args:
        query: The search query string.
        user_location: Optional location for the search. Lets you customize results to be relevant to a location.
        filters: A filter to apply. Should support 'allowed_domains' list.
        search_context_size: The amount of context to use for the search. One of 'low', 'medium', or 'high'. 'medium' is the default.
    
    Returns:
        A formatted string containing search results with titles, URLs, and descriptions.
    """
    try:
        client = _get_firecrawl_client()
        
        # Map search_context_size to limit
        limit = _map_search_context_size_to_limit(search_context_size)
        
        # Prepare search parameters
        search_params = {
            "query": query,
            "limit": limit,
        }
        
        # Add location if provided
        if user_location:
            search_params["location"] = user_location
        
        # Handle domain filters - Firecrawl doesn't directly support domain filtering in search,
        # so we'll need to filter results after getting them
        allowed_domains = None
        if filters and isinstance(filters, dict):
            allowed_domains = filters.get("allowed_domains")
            if allowed_domains and not isinstance(allowed_domains, list):
                allowed_domains = None
        
        # Perform search
        results = client.search(**search_params)
        
        # Extract web results
        web_results = results.get("data", {}).get("web", [])
        
        # Filter by allowed domains if specified
        if allowed_domains:
            allowed_domains_set = {domain.lower().strip() for domain in allowed_domains if domain}
            filtered_results = []
            for result in web_results:
                url = result.get("url", "")
                # Extract domain from URL
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()
                    # Remove www. prefix for comparison
                    if domain.startswith("www."):
                        domain = domain[4:]
                    # Check if domain matches any allowed domain
                    if any(
                        allowed_domain.lower().replace("www.", "") == domain
                        or domain.endswith("." + allowed_domain.lower().replace("www.", ""))
                        for allowed_domain in allowed_domains_set
                    ):
                        filtered_results.append(result)
                except Exception:
                    # If URL parsing fails, include the result
                    filtered_results.append(result)
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
    
    except ValueError as e:
        # Re-raise ValueError (e.g., missing API key)
        raise
    except Exception as e:
        return f"Error performing web search: {str(e)}"

