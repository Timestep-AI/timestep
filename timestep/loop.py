"""Execution loop and data structures for timestep."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

from timestep.agent import Agent
from timestep.environment import Environment


def is_multi_agent(obs_dict: Any) -> bool:
    """Check if observation dict represents multi-agent mode."""
    return (
        isinstance(obs_dict, dict)
        and all(isinstance(v, dict) and "messages" in v for v in obs_dict.values())
        and len(obs_dict) > 1
    )

# ---------- Data structures ----------


@dataclass
class StepRecord:
    """Record of a single step in an episode."""

    t: int
    observation: Any
    action: Any
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodeResult:
    """Result of a complete episode."""

    episode_id: str
    steps: list[StepRecord]
    metrics: dict[str, Any]


# ---------- Runner ----------


class Runner:
    """Executes episodes and records trajectories."""

    def __init__(
        self,
        env: Environment,
        agents: dict[str, Agent] | Agent,
    ):
        """
        Initialize runner.

        Args:
            env: Environment to run episodes in
            agents: Either a single Agent (for backward compatibility) or dict[str, Agent] mapping agent IDs to Agent instances
        """
        self.env = env
        # Handle both single agent (backward compatibility) and dict of agents
        if isinstance(agents, dict):
            self.agents = agents
        else:
            # Single agent mode - create dict with "draft" key for backward compatibility
            self.agents = {"draft": agents}

    def run_episode(
        self,
        *,
        seed: Optional[int] = None,
        max_steps: int = 50,
    ) -> EpisodeResult:
        """
        Run a single episode.

        Args:
            seed: Optional random seed for environment reset
            max_steps: Maximum number of steps before truncation

        Returns:
            EpisodeResult with trajectory and computed metrics
        """
        episode_id = str(uuid.uuid4())
        print(f"\nEpisode: {episode_id}")  # noqa: T201

        # Reset all agents
        for agent in self.agents.values():
            agent.reset()

        obs_dict = self.env.reset(seed=seed)
        steps: list[StepRecord] = []

        # Check if multi-agent mode (obs_dict is dict of dicts) or single-agent mode
        if is_multi_agent(obs_dict):
            # Multi-agent mode: use agent_iter()
            for step_count, agent_id in enumerate(self.env.agent_iter()):
                if step_count >= max_steps:
                    break

                # Get observation for current agent
                obs_dict, reward, done, info = self.env.last()
                obs = obs_dict[agent_id]

                # Print agent label
                print(f"\nAgent: {agent_id.capitalize()}")  # noqa: T201

                # Print observation messages
                for msg in obs["messages"]:
                    print(json.dumps(msg, indent=2))  # noqa: T201

                # Get agent and act
                agent = self.agents[agent_id]
                action = agent.act(obs)

                # Print action messages
                for msg in action["messages"]:
                    print(json.dumps(msg, indent=2))  # noqa: T201

                # Environment steps (handles agent cycling)
                next_obs_dict, reward, done, info = self.env.step(action, agent_id=agent_id)

                steps.append(
                    StepRecord(
                        t=step_count,
                        observation=obs_dict,
                        action=action,
                        reward=reward,
                        done=done,
                        info=info or {},
                    )
                )

                obs_dict = next_obs_dict
                if done:
                    break
        else:
            # Single-agent mode (backward compatibility)
            # Handle both dict of dicts (with single agent) and single dict
            if isinstance(obs_dict, dict) and "messages" in obs_dict and "tools" in obs_dict:
                # Single dict format
                obs = obs_dict
            else:
                # Extract from dict of dicts if needed
                agent_id = list(self.agents.keys())[0]
                if isinstance(obs_dict, dict) and agent_id in obs_dict:
                    obs = obs_dict[agent_id]
                else:
                    obs = obs_dict

            for t in range(max_steps):
                # Print agent label
                print("\nAgent: Draft")  # noqa: T201

                # Print observation messages
                for msg in obs["messages"]:
                    print(json.dumps(msg, indent=2))  # noqa: T201

                agent = list(self.agents.values())[0]
                action = agent.act(obs)
                # Print action messages
                for msg in action["messages"]:
                    print(json.dumps(msg, indent=2))  # noqa: T201

                next_obs, reward, done, info = self.env.step(action)

                steps.append(
                    StepRecord(
                        t=t,
                        observation=obs,
                        action=action,
                        reward=reward,
                        done=done,
                        info=info or {},
                    )
                )

                obs = next_obs
                if done:
                    break

        # Extract metrics from final step's info dict
        metric_values = {}
        if steps:
            final_info = steps[-1].info
            # Collect all metric keys from info (excluding non-metric keys like task_id, predicted_answer, etc.)
            metric_keys = {"steps_taken", "tool_calls_count", "exact_match"}
            for key in metric_keys:
                if key in final_info:
                    metric_values[key] = final_info[key]

        return EpisodeResult(episode_id=episode_id, steps=steps, metrics=metric_values)

    def run_many(
        self,
        seeds: Iterable[int],
        *,
        max_steps: int = 50,
    ) -> list[EpisodeResult]:
        """
        Run multiple episodes.

        Args:
            seeds: Iterable of seeds for each episode
            max_steps: Maximum number of steps per episode

        Returns:
            List of EpisodeResults
        """
        return [self.run_episode(seed=s, max_steps=max_steps) for s in seeds]
