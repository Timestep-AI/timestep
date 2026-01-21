# Timestep

A unified RL-style evaluation framework for AI agents that unifies different eval families under one skeleton.

## Overview

Timestep provides a minimal, extensible framework for evaluating AI agents:

- **Environment**: Emits observations (with tools), accepts actions, returns next observation + rewards/info (with metrics)
- **Agent**: Maps observation + memory → action (extracts tools from observations)
- **Runner**: Executes episodes, records a trajectory log, extracts metrics from environment

## Architecture

```
┌─────────────┐
│   Runner    │
│  (Loop)     │
└──────┬──────┘
       │
       ├───▶ env.reset() → obs = {"messages": [...], "tools": [...]}
       │
       ├───▶ agent.act(obs) → action (uses tools from obs)
       │
       ├───▶ env.step(action) → obs = {"messages": [...], "tools": [...]}, reward, done, info
       │                                                                    (info contains metrics)
       │
       └───▶ Extract metrics from final step's info dict

┌─────────────┐         ┌─────────────┐
│ Environment │────────▶│   Agent     │
│             │         │             │
│ - reset()   │         │ - reset()   │
│   → obs with│         │ - act(obs)  │
│   tools     │         │   extracts  │
│ - step()    │         │   tools from│
│   → obs with│         │   obs       │
│   tools     │         │             │
│ - step()    │         │             │
│   returns   │         │             │
│   info with │         │             │
│   metrics   │         │             │
└─────────────┘         └─────────────┘
```

## Installation

```bash
pip install -e .
```

For GAIA example support:

```bash
pip install -e ".[gaia]"
```

Or with `uv`:

```bash
uv pip install -e ".[gaia]"
```

## Quick Start

### Basic Usage

```python
from timestep import Runner

# Create your environment (must implement timestep.Environment)
env = YourEnvironment()

# Create your agent (must implement timestep.Agent)
agent = YourAgent()

# Create runner (metrics are automatically extracted from environment)
runner = Runner(env=env, agent=agent)

# Run an episode
result = runner.run_episode(seed=42, max_steps=50)
print(f"Episode {result.episode_id}: {result.metrics}")

# Run multiple episodes
results = runner.run_many(seeds=range(10), max_steps=50)
```

### GAIA Example

See `examples/gaia/loop.py` for a complete example using the GAIA benchmark.

**Setup:**

1. Install GAIA dependencies:
   ```bash
   pip install -e ".[gaia]"
   # Or with uv:
   uv pip install -e ".[gaia]"
   ```

2. Accept terms on [GAIA Hugging Face dataset page](https://huggingface.co/datasets/gaia-benchmark/GAIA)

3. Set your Hugging Face token:
   ```bash
   export HF_TOKEN=your_token_here
   ```

4. Run the example:
   ```bash
   python examples/gaia/loop.py
   ```
   
   Or run as a module:
   ```bash
   python -m examples.gaia.loop
   ```
   
   Or with `uv`:
   ```bash
   uv run python examples/gaia/loop.py
   ```

## Core Concepts

### Environment Protocol

Implement the `Environment` protocol:

```python
from timestep import Environment
from typing import Any, Optional

class MyEnvironment:
    def reset(self, *, seed: Optional[int] = None) -> Any:
        # Return initial observation (may include "tools" field)
        # For OpenAI environments: {"messages": [...], "tools": [...]}
        return {"messages": [...], "tools": [...]}
    
    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        # Execute action, return (obs, reward, done, info)
        # info dict may contain metrics (e.g., "steps_taken", "tool_calls_count")
        # For OpenAI environments: obs = {"messages": [...], "tools": [...]}
        return next_obs, reward, done, info
```

### Agent Protocol

Implement the `Agent` protocol:

```python
from timestep import Agent

class MyAgent:
    def reset(self) -> None:
        # Reset agent state (e.g., clear memory)
        pass
    
    def act(self, observation: Any) -> Any:
        # Extract tools from observation if needed
        # tools = observation.get("tools", [])
        # Return action given observation
        return action
```

### Metrics

Metrics are computed by the environment and included in the `info` dict returned from `step()`.
The Runner automatically extracts metrics from the final step's info dict.

Common metrics provided by `OpenAIEnvironment`:
- `steps_taken`: Number of steps in the episode
- `tool_calls_count`: Number of tool calls made

Environment subclasses can add custom metrics to the info dict:

```python
def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
    # ... execute step ...
    info = {}
    
    # Add custom metrics when done
    if done:
        info["my_metric"] = compute_my_metric(...)
    
    return obs, reward, done, info
```

## Extension Points

### Custom Environments

Any environment implementing the `Environment` protocol can be used. Examples:
- Benchmarks (GAIA, WebArena, etc.)
- Simulated environments
- Real-world API wrappers

For OpenAI-based environments, extend `OpenAIEnvironment` which provides:
- Web search tool by default (can be overridden)
- Tool schema management via `_get_tool_schemas()`
- Generic metrics tracking (steps_taken, tool_calls_count)

### Custom Agents

Implement the `Agent` protocol with your own agent:
- LLM-based agents
- Tool-using agents
- Multi-modal agents

For OpenAI-based agents, extend `OpenAIAgent` which automatically extracts tools from observations.
See `examples/gaia/agent.py` for an example `GAIAAgent` implementation.

### Custom Metrics

Add custom metrics by including them in the environment's `info` dict:

```python
def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
    # ... execute step ...
    info = {}
    
    # Add metrics to info dict
    if done:
        info["robustness"] = compute_robustness(...)
        info["safety_score"] = check_safety(...)
        info["efficiency"] = compute_efficiency(...)
    
    return obs, reward, done, info
```

Common metric types:
- **Robustness**: Perturbation resistance
- **Safety**: Action/tool call violation detection
- **Efficiency**: Steps, tool calls, latency, cost
- **Generalization**: Cross-distribution performance

## Design Philosophy

- **Three core entities**: Agent, Environment, Loop (metrics are part of environment)
- **Trajectory logging**: Complete episode logs for post-hoc analysis
- **Protocol-based**: Flexible duck typing, no inheritance required
- **Extensible**: Easy to add new environments, agents, and metrics
- **Tool management**: Environment controls tool availability via observations
- **Metric computation**: Environment computes and provides metrics in info dict

## License

MIT License - see LICENSE file for details.
