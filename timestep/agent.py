"""Agent base class for timestep."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class Agent(ABC):
    """Base agent class that chooses an action given the current observation."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent's internal state."""
        ...

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Choose an action given the current observation."""
        ...


class OpenAIAgent(Agent):
    """
    Base OpenAI agent with memory management.

    Handles OpenAI message format: observations and actions use {"messages": [...]} format.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI agent.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        if OpenAI is None:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")

        self.model = model
        self.api_key = api_key

        if not self.api_key:
            import os

            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)

        # Memory is a list of OpenAI messages
        self.memory: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset agent state (clear conversation history)."""
        self.memory = []

    def act(self, observation: Any) -> dict[str, Any]:
        """
        Process observation and return assistant message as action.

        Args:
            observation: Dict with "messages" key containing list of OpenAI messages,
                         and optionally "tools" key containing list of tool schemas

        Returns:
            Dict with "messages" key containing list with assistant message: {"messages": [assistant_message]}
        """
        # Extract messages list from observation
        messages = observation["messages"]

        # Append all messages to memory immediately
        self.memory.extend(messages)

        # Extract tools from observation (if provided)
        tools = observation.get("tools", [])

        # Call LLM with memory and tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.memory,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            temperature=0,
        )

        message = response.choices[0].message

        # Convert message to dict format
        message_dict = {
            "role": message.role,
            "content": message.content,
        }

        # Add tool calls if present
        if message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        # Append assistant response to memory immediately
        self.memory.append(message_dict)

        # Return action as dict with list: {"messages": [assistant_message]}
        return {"messages": [message_dict]}
