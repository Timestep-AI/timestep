# Timestep (Python)

Python bindings for the Timestep AI Agents SDK. See the root `README.md` for the full story; this file highlights Python-specific setup.

## Install

```bash
pip install timestep
```

Or for development:

```bash
pip install -e .
pip install -e ".[dev]"  # For development dependencies
```

## Prerequisites

- **Python 3.11+**
- **OpenAI API key** (optional, for agents that use OpenAI or LLM-as-judge graders)

## Quick Start: Core Agent-Environment Loop

```python
from timestep import run_episode, agent_builtin_echo, DEFAULT_TOOLS
from timestep.core.types import Message

# Define initial messages
messages: list[Message] = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Calculate 5 + 3 using the calc tool."}
]

# Run a single episode
transcript, info = run_episode(
    initial_messages=messages,
    agent=agent_builtin_echo,
    tools=DEFAULT_TOOLS,
    tools_allowed=["calc"],
    limits={"max_steps": 10, "time_limit_s": 30},
    task_meta={"id": "demo"},
    seed=0,
)

print(f"Steps: {info.steps}, Tool calls: {info.tool_calls}")
print(f"Final message: {transcript[-1]['content']}")
```

## Quick Start: Evaluation Harness

```python
from timestep import run_suite, report, agent_builtin_echo, DEFAULT_TOOLS
from timestep import FinalContains, ForbiddenTools, LLMJudge
from pathlib import Path
import json

# Create tasks.jsonl
tasks = [
    {
        "id": "hello_01",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello to Mike."}
        ],
        "expected": {"final_contains": "Mike"},
        "limits": {"max_steps": 5}
    }
]

with open("tasks.jsonl", "w") as f:
    for task in tasks:
        f.write(json.dumps(task) + "\n")

# Run eval suite
run_suite(
    tasks_path=Path("tasks.jsonl"),
    outdir=Path("runs/demo"),
    agent=agent_builtin_echo,
    tools=DEFAULT_TOOLS,
    graders=[
        FinalContains(),  # Code-based grader
        ForbiddenTools(),  # Tool usage checker
        # LLMJudge(rubric="Is the response friendly?"),  # LLM-as-judge
    ],
    trials=3,
    seed=0,
    agent_timeout_s=120,
)

# Generate report
report(Path("runs/demo"))
```

## CLI Usage

```bash
# Install CLI
pip install timestep

# Run eval suite
timestep run --tasks tasks.jsonl --outdir runs/demo --agent builtin:echo

# Generate report
timestep report --outdir runs/demo
```

## Creating Your Own Agent Harness

```python
from timestep import AgentFn, Message, JSON

def my_agent(messages: list[Message], context: JSON) -> Message:
    """
    Agent harness function.
    
    Args:
        messages: Full conversation history (transcript)
        context: Context with tools_schema, task, seed, limits
    
    Returns:
        Assistant message (may include tool_calls and usage info)
    """
    # Your agent logic here
    # Use OpenAI library or other model provider
    return {
        "role": "assistant",
        "content": "Response here",
        "tool_calls": [...],  # Optional
        "usage": {  # Optional: token usage info
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
```

See `examples/agent_adapter_example.py` for a complete OpenAI integration example.

## Project Structure

```
python/timestep/
├── core/              # Core agent-environment loop
│   ├── agent.py       # Agent harness interface
│   ├── episode.py     # Episode runner
│   ├── tools.py        # Tool execution
│   └── types.py        # Core types
├── eval/               # Evaluation harness
│   ├── suite.py        # Suite runner
│   ├── graders.py      # All graders (code-based, LLM-as-judge, outcome)
│   └── cli.py          # CLI interface
└── utils/              # Utilities (JSONL, hashing, etc.)
```

## Testing

```bash
cd python
pytest tests/ -v
```

## Documentation

- Full docs: https://timestep-ai.github.io/timestep/
- Root README: `../README.md`
