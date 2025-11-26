"""Timestep AI - Multi-model provider implementations."""

from .ollama_model import OllamaModel
from .ollama_model_provider import OllamaModelProvider
from .multi_model_provider import MultiModelProvider, MultiModelProviderMap

__all__ = [
    "OllamaModel",
    "OllamaModelProvider",
    "MultiModelProvider",
    "MultiModelProviderMap",
    "run_agent",
]

from agents import Agent, Runner, RunConfig, TResponseInputItem
from agents.memory.session import SessionABC

async def run_agent(agent: Agent, run_input: list[TResponseInputItem], session: SessionABC, stream: bool):
    """Run an agent with the given session and stream setting."""
    async def session_input_callback(existing_items: list, new_input: list) -> list:
        """Callback to merge new input with existing session items."""
        return existing_items + new_input
    
    run_config = RunConfig(session_input_callback=session_input_callback)
    
    if stream:
        result = Runner.run_streamed(agent, run_input, run_config=run_config, session=session)
        async for event in result.stream_events():
            pass  # Consume all events
    else:
        result = await Runner.run(agent, run_input, run_config=run_config, session=session)
