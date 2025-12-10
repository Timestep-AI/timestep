"""Analysis module for category theory modeling of multi-agent systems."""

from .categories import Category, AgentCategory, ToolCategory
from .functors import Functor, AgentToolFunctor, HandoffFunctor, StateFunctor
from .monoidal import MonoidalCategory, AgentComposition
from .olog import Olog, OlogType, OlogAspect, OlogRelationType, OlogBuilder, OlogValidator
from .safety import (
    CircularDependencyChecker,
    ToolCompatibilityChecker,
    StateInvariant,
    StateVerifier
)

__all__ = [
    # Categories
    "Category",
    "AgentCategory",
    "ToolCategory",
    # Functors
    "Functor",
    "AgentToolFunctor",
    "HandoffFunctor",
    "StateFunctor",
    # Monoidal
    "MonoidalCategory",
    "AgentComposition",
    # Olog
    "Olog",
    "OlogType",
    "OlogAspect",
    "OlogRelationType",
    "OlogBuilder",
    "OlogValidator",
    # Safety
    "CircularDependencyChecker",
    "ToolCompatibilityChecker",
    "StateInvariant",
    "StateVerifier",
]

