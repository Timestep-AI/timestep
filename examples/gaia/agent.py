"""GAIA-specific agent implementation."""

from __future__ import annotations

from typing import Optional

from timestep.agent import OpenAIAgent


class GAIAAgent(OpenAIAgent):
    """
    Agent that uses OpenAI GPT-4o with web search tool to answer GAIA questions.

    Extends OpenAIAgent with GAIA-specific configuration.
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: Optional[str] = None,
    ):
        """
        Initialize GAIA agent.

        Args:
            model: OpenAI model to use (default: "gpt-5.2")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        super().__init__(
            model=model,
            api_key=api_key,
        )


class DraftAgent(OpenAIAgent):
    """
    Draft agent that produces initial answers.

    Can use tools and produces standard assistant messages with content (the draft answer).
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: Optional[str] = None,
    ):
        """
        Initialize draft agent.

        Args:
            model: OpenAI model to use (default: "gpt-5.2")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        super().__init__(model=model, api_key=api_key)


class ReflectionAgent(OpenAIAgent):
    """
    Reflection agent that reviews draft answers.

    Produces standard assistant messages with reflection/critique content.
    Receives draft string in its system/user message from environment.
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: Optional[str] = None,
    ):
        """
        Initialize reflection agent.

        Args:
            model: OpenAI model to use (default: "gpt-5.2")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        super().__init__(model=model, api_key=api_key)


class RevisionAgent(OpenAIAgent):
    """
    Revision agent that refines answers based on reflection.

    Produces standard final answer (more succinct than draft).
    Receives draft + reflection in system/user message from environment.
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: Optional[str] = None,
    ):
        """
        Initialize revision agent.

        Args:
            model: OpenAI model to use (default: "gpt-5.2")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        super().__init__(model=model, api_key=api_key)


class DecisionAgent(OpenAIAgent):
    """
    Decision agent that determines if draft needs retry.

    Produces standard assistant messages with decision: "retry" or "proceed_to_revision".
    Receives question, draft answer, and reflection critique from environment.
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: Optional[str] = None,
    ):
        """
        Initialize decision agent.

        Args:
            model: OpenAI model to use (default: "gpt-5.2")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        super().__init__(model=model, api_key=api_key)
