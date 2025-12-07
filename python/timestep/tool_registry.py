"""Tool registry for mapping tool identifiers to Tool objects."""

from typing import Dict, Optional, Any
from ._vendored_imports import Tool


# Global registry mapping tool identifiers to Tool objects
_tool_registry: Dict[str, Tool] = {}


def register_tool(identifier: str, tool: Tool) -> None:
    """
    Register a tool in the registry.
    
    Args:
        identifier: Unique identifier for the tool (typically tool.name)
        tool: The Tool object to register
    """
    _tool_registry[identifier] = tool


def get_tool(identifier: str) -> Optional[Tool]:
    """
    Get a tool from the registry by identifier.
    
    Args:
        identifier: The tool identifier
        
    Returns:
        The Tool object if found, None otherwise
    """
    return _tool_registry.get(identifier)


def clear_registry() -> None:
    """Clear the tool registry (useful for testing)."""
    _tool_registry.clear()


