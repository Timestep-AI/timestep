"""GAIA-specific loop implementation."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

# Add project root to path for imports when running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from examples.gaia.agent import DecisionAgent, DraftAgent, GAIAAgent, ReflectionAgent, RevisionAgent
from examples.gaia.environment import GAIAConfig, GAIAEnvironment
from timestep import Runner
from timestep.loop import EpisodeResult


class GAIALoop(Runner):
    """
    GAIA-specific loop that extends Runner.

    Encapsulates GAIA environment and agent setup.
    """

    def __init__(
        self,
        config: GAIAConfig,
        model: str = "gpt-5.2",
        api_key: Optional[str] = None,
        use_reflection: bool = False,
    ):
        """
        Initialize GAIA loop.

        Args:
            config: GAIA configuration
            model: OpenAI model to use (default: "gpt-5.2")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            use_reflection: If True, use multi-agent reflection (draft, reflection, revision)
        """
        env = GAIAEnvironment(config, use_reflection=use_reflection)
        if use_reflection:
            agents = {
                "draft": DraftAgent(model=model, api_key=api_key),
                "reflection": ReflectionAgent(model=model, api_key=api_key),
                "decision": DecisionAgent(model=model, api_key=api_key),
                "revision": RevisionAgent(model=model, api_key=api_key),
            }
        else:
            agents = {"draft": GAIAAgent(model=model, api_key=api_key)}
        super().__init__(env, agents)

    def run(
        self,
        seeds: Iterable[int] = range(3),
        max_steps: int = 5,
    ) -> list[EpisodeResult]:
        """
        Run GAIA evaluation episodes.

        Args:
            seeds: Iterable of seeds for each episode
            max_steps: Maximum number of steps per episode

        Returns:
            List of EpisodeResults
        """
        results = self.run_many(seeds=seeds, max_steps=max_steps)
        return results


def main() -> None:
    """Run GAIA evaluation with GAIALoop."""
    loop = GAIALoop(GAIAConfig(config_name="2023_level1", split="validation"), use_reflection=True)
    results = loop.run(seeds=range(3), max_steps=50)

    for r in results:
        print(f"\nEpisode: {r.episode_id}")
        print(f"Metrics: {r.metrics}")
        if r.steps:
            first_step = r.steps[0]
            obs = first_step.observation
            # Handle both single-agent (dict with "messages") and multi-agent (dict of dicts)
            if isinstance(obs, dict):
                # Check if multi-agent mode (dict of dicts)
                if all(isinstance(v, dict) and "messages" in v for v in obs.values()) and len(obs) > 1:
                    # Multi-agent mode: get draft agent's observation
                    draft_obs = obs.get("draft", {})
                    messages = draft_obs.get("messages", [])
                elif "messages" in obs:
                    # Single-agent mode
                    messages = obs["messages"]
                else:
                    messages = []

                if isinstance(messages, list) and len(messages) > 0:
                    # Find user message
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            content = msg.get("content", "")
                            # Extract question from content
                            if "Question:" in content:
                                question = content.split("Question:")[1].split("\n\n")[0].strip()
                                print(f"Question: {question[:100]}...")
                            else:
                                print(f"Question: {content[:100]}...")
                            break
            if r.steps[-1].info.get("reference_answer"):
                print(f"Reference Answer: {r.steps[-1].info.get('reference_answer')}")
                print(f"Predicted Answer: {r.steps[-1].info.get('predicted_answer')}")


if __name__ == "__main__":
    main()
