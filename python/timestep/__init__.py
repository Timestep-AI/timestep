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
    "ApprovalCallback",
]

from typing import Awaitable, Callable
from agents import Agent, Runner, RunConfig, TResponseInputItem
from agents.memory.session import SessionABC

ApprovalCallback = Callable[[any], Awaitable[bool]]

async def run_agent(
    agent: Agent,
    run_input: list[TResponseInputItem],
    session: SessionABC,
    stream: bool,
    approval_callback: ApprovalCallback | None = None
):
    """Run an agent with the given session and stream setting."""
    async def session_input_callback(existing_items: list, new_input: list) -> list:
        """Callback to merge new input with existing session items."""
        return existing_items + new_input

    run_config = RunConfig(session_input_callback=session_input_callback)

    if stream:
        result = Runner.run_streamed(agent, run_input, run_config=run_config, session=session)
        async for event in result.stream_events():
            pass  # Consume all events

        # Handle interruptions
        while result.interruptions:
            state = result.to_state()
            for interruption in result.interruptions:
                approved = await approval_callback(interruption) if approval_callback else True
                if approved:
                    state.approve(interruption)
                else:
                    state.reject(interruption)

            # Resume execution
            result = Runner.run_streamed(agent, state, run_config=run_config, session=session)
            async for event in result.stream_events():
                pass  # Consume all events
    else:
        result = await Runner.run(agent, run_input, run_config=run_config, session=session)

        # Handle interruptions
        while result.interruptions:
            state = result.to_state()
            for interruption in result.interruptions:
                approved = await approval_callback(interruption) if approval_callback else True
                if approved:
                    state.approve(interruption)
                else:
                    state.reject(interruption)

            # Resume execution
            result = await Runner.run(agent, state, run_config=run_config, session=session)
