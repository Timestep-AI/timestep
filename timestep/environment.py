"""Environment base class for timestep."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Environment(ABC):
    """Base environment class with gym-like interface."""

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None) -> Any:
        """
        Reset the environment and return initial observation.

        Note:
            Observations may include a "tools" field containing a list of tool schemas
            (e.g., OpenAI function calling format). This allows the environment to
            control which tools are available to the agent at each step.
        """
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Returns:
            Tuple of (observation, reward, done, info)

        Note:
            The info dict may contain metric keys (e.g., "steps_taken", "tool_calls_count")
            that are computed by the environment. Metrics are typically included in the
            final step when done=True.
        """
        ...

    def agent_iter(self):
        """
        Generator that yields the current active agent ID.

        Yields the current agent once per iteration. After step() is called,
        _agent_selection is updated, and the next iteration yields the new agent.
        Continues until _agent_selection is None (episode done).

        Yields:
            Current active agent ID (str)
        """
        while hasattr(self, "_agent_selection") and self._agent_selection is not None:
            yield self._agent_selection
            # After yielding, the for loop will call next() again
            # step() should have been called to update _agent_selection
            # If _agent_selection is still the same (e.g., tool calls), we yield again
            # If _agent_selection changed or is None, we continue or break

    def last(self) -> tuple[Any, float, bool, dict[str, Any]]:
        """
        Return current observation, reward, done, info for the active agent.

        Returns:
            Tuple of (observation, reward, done, info)

        Note:
            For multi-agent environments, observation is a dict of dicts with one dict per agent.
            For single-agent environments, observation is a dict with "messages" and "tools" keys.
        """
        raise NotImplementedError("Subclasses must implement last()")


class OpenAIEnvironment(Environment):
    """
    Base OpenAI environment that handles OpenAI message format.

    Handles actions with {"messages": [...]} format and extracts assistant messages.
    Subclasses should implement tool execution and reward calculation.

    Tracks generic metrics in info dict:
    - steps_taken: Number of steps in the episode (incremented each step)
    - tool_calls_count: Number of tool calls made (cumulative)
    """

    def __init__(self):
        """Initialize OpenAI environment with metric tracking."""
        self._step_count: int = 0
        self._tool_calls_count: int = 0
        # Agent management for multi-agent support
        self._agents: list[str] = []
        self._agent_selection: Optional[str] = None
        self._agent_idx: int = 0
        self._agent_states: dict[str, dict[str, Any]] = {}
        self._agent_system_prompts: dict[str, str] = {}

    def _web_search_tool_schema(self) -> dict[str, Any]:
        """
        Return the web search tool schema.

        Subclasses can override to customize the tool schema.
        """
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using DuckDuckGo to find information. Use this when you need to look up facts, current information, or verify details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the web",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Return list of tool schemas available at the current step.

        Default implementation returns web search tool. Subclasses can override
        to customize available tools (add, remove, or modify).
        """
        return [self._web_search_tool_schema()]

    def _web_search(self, query: str) -> str:
        """
        Search the web using DuckDuckGo and return results.

        Args:
            query: Search query string

        Returns:
            Search results as a string
        """
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))

            if not results:
                return "No search results found."

            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                body = result.get("body", "No description")
                href = result.get("href", "")
                formatted_results.append(f"{i}. {title}\n   {body}\n   URL: {href}\n")

            return "\n".join(formatted_results)
        except ImportError:
            return "Error: ddgs library not installed. Install with: pip install ddgs"
        except Exception as e:
            return f"Error during web search: {str(e)}"

    def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """
        Execute a tool and return the result.

        Default implementation handles web_search. Subclasses can override
        to add custom tools or override web search behavior.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Tool result as a string
        """
        if tool_name == "web_search":
            query = tool_args.get("query", "")
            if not query:
                return "Error: query parameter required for web_search"
            return self._web_search(query)
        return f"Error: Unknown tool {tool_name}"

    @abstractmethod
    def compute_reward(self, predicted_answer: str, reference_answer: Optional[str]) -> float:
        """
        Compute reward for a final answer.

        Args:
            predicted_answer: The agent's answer
            reference_answer: The correct answer (may be None)

        Returns:
            Reward value
        """
        ...

    def reset(self, *, seed: Optional[int] = None) -> Any:
        """
        Reset environment state and metrics.

        Subclasses should override this method and call super().reset(seed=seed)
        to reset metric counters, then return observation with tools.

        Returns:
            Dict with "messages" and "tools" keys (subclasses should implement)
            or dict of dicts for multi-agent environments
        """
        self._step_count = 0
        self._tool_calls_count = 0
        # Reset agent management
        self._agent_idx = 0
        if self._agents:
            self._agent_selection = self._agents[0]
        else:
            self._agent_selection = None

    def step(
        self, action: Any, agent_id: Optional[str] = None, info: Optional[dict[str, Any]] = None
    ) -> tuple[Any, float, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Dict with "messages" key containing list with assistant message
            agent_id: Optional agent ID (if None, uses _agent_selection)
            info: Optional info dict to update (subclass can pass reference_answer here)

        Returns:
            Tuple of (observation, reward, done, info)
            For multi-agent: observation is dict of dicts
            For single-agent: observation is dict with "messages" and "tools" keys
        """
        import json

        if info is None:
            info = {}

        # Use agent_id if provided, otherwise use _agent_selection
        current_agent_id = agent_id if agent_id is not None else self._agent_selection

        # Increment step count
        self._step_count += 1

        # Extract messages list from action
        messages = action["messages"]

        # Find assistant message in list
        assistant_message = None
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_message = msg
                break

        # Handle assistant message with tool_calls - execute ALL tool calls
        if assistant_message and assistant_message.get("tool_calls"):
            tool_calls = assistant_message["tool_calls"]
            # Count tool calls
            self._tool_calls_count += len(tool_calls)
            tool_messages = []

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                # Execute tool using subclass implementation
                tool_result = self.execute_tool(function_name, function_args)

                # Create tool message
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(tool_result),
                }
                tool_messages.append(tool_message)

            # Add metrics to info
            info["steps_taken"] = self._step_count
            info["tool_calls_count"] = self._tool_calls_count

            # Update agent state with tool messages (replace, not append)
            if current_agent_id and current_agent_id in self._agent_states:
                self._agent_states[current_agent_id]["messages"] = tool_messages

            # Return observation
            if self._agents and len(self._agents) > 1:
                # Multi-agent mode: return dict of dicts
                obs = {aid: self._agent_states.get(aid, {"messages": [], "tools": []}) for aid in self._agents}
            else:
                # Single-agent mode: return single dict
                obs = {"messages": tool_messages, "tools": self._get_tool_schemas()}
            return obs, 0.0, False, info

        # Handle final answer (assistant message with content, no tool_calls)
        if assistant_message and assistant_message.get("content"):
            pred = (assistant_message.get("content") or "").strip()
            info["predicted_answer"] = pred

            # Get reference answer from info (subclass should set this)
            ref = info.get("reference_answer")

            # Compute reward using subclass implementation
            reward = self.compute_reward(pred, ref)
            if ref:
                info["reference_answer"] = ref

            # Add metrics to info for final step
            info["steps_taken"] = self._step_count
            info["tool_calls_count"] = self._tool_calls_count

            done = True
            # Return observation
            if self._agents and len(self._agents) > 1:
                # Multi-agent mode: return dict of dicts
                obs = {aid: self._agent_states.get(aid, {"messages": [], "tools": []}) for aid in self._agents}
            else:
                # Single-agent mode: return empty dict
                obs = {"messages": [], "tools": []}
            return obs, reward, done, info

        raise ValueError(f"Assistant message must have either tool_calls or content: {assistant_message}")

    def agent_iter(self):
        """
        Generator that yields the current active agent ID.

        Yields:
            Current active agent ID (str)
        """
        while self._agent_selection is not None:
            yield self._agent_selection

    def last(self) -> tuple[Any, float, bool, dict[str, Any]]:
        """
        Return observation for current agent(s).

        Returns:
            Tuple of (observation, reward, done, info)
            For multi-agent: observation is dict of dicts {agent_id: {"messages": [...], "tools": [...]}}
            For single-agent: observation is dict with "messages" and "tools" keys
        """
        if not self._agents:
            raise RuntimeError("No agents configured. Call reset() first.")
        if len(self._agents) > 1:
            # Multi-agent mode: return dict of dicts
            obs_dict = {agent_id: self._agent_states.get(agent_id, {"messages": [], "tools": []}) for agent_id in self._agents}
            return obs_dict, 0.0, False, {}
        else:
            # Single-agent mode: return single dict
            agent_id = self._agents[0]
            obs = self._agent_states.get(agent_id, {"messages": [], "tools": []})
            return obs, 0.0, False, {}
