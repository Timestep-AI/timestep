"""Timestep AI - Multi-model provider implementations."""

from .ollama_model import OllamaModel
from .ollama_model_provider import OllamaModelProvider
from .multi_model_provider import MultiModelProvider, MultiModelProviderMap
from .tools import web_search

__all__ = [
    "OllamaModel",
    "OllamaModelProvider",
    "MultiModelProvider",
    "MultiModelProviderMap",
    "run_agent",
    "consume_result",
    "InterruptionException",
    "RunStateStore",
    "web_search",
]

from typing import Any
from agents import Agent, Runner, RunConfig, RunState, TResponseInputItem
from agents.memory.session import SessionABC
from pathlib import Path
import json

class InterruptionException(Exception):
    """Exception raised when agent execution is interrupted for approval."""
    def __init__(self, message: str = "Agent execution interrupted for approval"):
        super().__init__(message)

class RunStateStore:
    """Store for persisting run state to file."""
    def __init__(self, file_path: str, agent: Agent):
        self.file_path = Path(file_path)
        self.agent = agent

    async def save(self, state: Any) -> None:
        """Save state to file."""
        self.file_path.write_text(json.dumps(state.to_json()))

    async def load(self) -> Any:
        """Load state from file."""
        content = self.file_path.read_text()
        state_json = json.loads(content)
        return await RunState.from_json(self.agent, state_json)

    async def clear(self) -> None:
        """Delete the state file."""
        if self.file_path.exists():
            self.file_path.unlink()

async def consume_result(result: Any) -> Any:
    """
    Consume all events from a result (streaming or non-streaming).

    Args:
        result: RunResult or RunResultStreaming from run_agent

    Returns:
        The same result object after consuming stream (if applicable)
    """
    if hasattr(result, 'stream_events'):
        async for _ in result.stream_events():
            pass
    return result


async def run_agent(
    agent: Agent,
    run_input: list[TResponseInputItem] | RunState,
    session: SessionABC,
    stream: bool
):
    """Run an agent with the given session and stream setting."""
    async def session_input_callback(existing_items: list, new_input: list) -> list:
        """Callback to merge new input with existing session items."""
        return existing_items + new_input

    run_config = RunConfig(
        nest_handoff_history=False, # Match TypeScript: don't nest handoff history
        session_input_callback=session_input_callback
    )

    if stream:
        result = Runner.run_streamed(agent, run_input, run_config=run_config, session=session)
    else:
        result = await Runner.run(agent, run_input, run_config=run_config, session=session)

    return result
