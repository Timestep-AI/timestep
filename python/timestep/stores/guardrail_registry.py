"""Guardrail registry for mapping guardrail identifiers to Guardrail objects."""

from typing import Dict, Optional
from .._vendored_imports import InputGuardrail, OutputGuardrail


# Global registry mapping guardrail identifiers to Guardrail objects
_input_guardrail_registry: Dict[str, InputGuardrail] = {}
_output_guardrail_registry: Dict[str, OutputGuardrail] = {}


def register_input_guardrail(identifier: str, guardrail: InputGuardrail) -> None:
    """
    Register an input guardrail in the registry.
    
    Args:
        identifier: Unique identifier for the guardrail (typically guardrail.name or guardrail.get_name())
        guardrail: The InputGuardrail object to register
    """
    _input_guardrail_registry[identifier] = guardrail


def register_output_guardrail(identifier: str, guardrail: OutputGuardrail) -> None:
    """
    Register an output guardrail in the registry.
    
    Args:
        identifier: Unique identifier for the guardrail (typically guardrail.name or guardrail.get_name())
        guardrail: The OutputGuardrail object to register
    """
    _output_guardrail_registry[identifier] = guardrail


def get_input_guardrail(identifier: str) -> Optional[InputGuardrail]:
    """
    Get an input guardrail from the registry by identifier.
    
    Args:
        identifier: The guardrail identifier
        
    Returns:
        The InputGuardrail object if found, None otherwise
    """
    return _input_guardrail_registry.get(identifier)


def get_output_guardrail(identifier: str) -> Optional[OutputGuardrail]:
    """
    Get an output guardrail from the registry by identifier.
    
    Args:
        identifier: The guardrail identifier
        
    Returns:
        The OutputGuardrail object if found, None otherwise
    """
    return _output_guardrail_registry.get(identifier)


def clear_registry() -> None:
    """Clear the guardrail registries (useful for testing)."""
    _input_guardrail_registry.clear()
    _output_guardrail_registry.clear()


