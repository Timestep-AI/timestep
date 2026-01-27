"""Core module - Agent, Environment, Loop, and ResponsesAPI entities."""

from timestep.core.agent import Agent
from timestep.core.environment import Environment
from timestep.core.loop import Loop
from timestep.core.responses_api import ResponsesAPI

# Tracing is optional - import only if available
try:
    from timestep.core.tracing import setup_tracing, instrument_fastapi_app
    __all__ = ["Agent", "Environment", "Loop", "ResponsesAPI", "setup_tracing", "instrument_fastapi_app"]
except ImportError:
    __all__ = ["Agent", "Environment", "Loop", "ResponsesAPI"]
