# Getting Started

Timestep AI Agents SDK provides a core agent-environment loop and an evaluation harness. This guide will help you get started with both.

## Installation

### Python

```bash
pip install timestep
```

### TypeScript

```bash
npm install @timestep-ai/timestep
# or
pnpm add @timestep-ai/timestep
```

## Quick Start: Core Agent-Environment Loop

The core SDK lets you run the agent-environment loop (which orchestrates the agent harness) without evaluation. This is useful for building agents, testing them interactively, or integrating into your own systems.

### Python

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
    agent=agent_builtin_echo,  # Your agent harness
    tools=DEFAULT_TOOLS,
    tools_allowed=["calc"],
    limits={"max_steps": 10, "time_limit_s": 30},
    task_meta={"id": "demo"},
    seed=0,
)

print(f"Steps: {info.steps}, Tool calls: {info.tool_calls}")
print(f"Final message: {transcript[-1]['content']}")
```

### TypeScript

```typescript
import { runEpisode, agentBuiltinEcho, DEFAULT_TOOLS } from '@timestep-ai/timestep';
import type { Message } from '@timestep-ai/timestep';

const messages: Message[] = [
  { role: 'system', content: 'You are helpful.' },
  { role: 'user', content: 'Calculate 5 + 3 using the calc tool.' }
];

const [transcript, info] = await runEpisode(
  messages,
  agentBuiltinEcho,
  DEFAULT_TOOLS,
  ['calc'],
  { max_steps: 10, time_limit_s: 30 },
  { id: 'demo' },
  0
);

console.log(`Steps: ${info.steps}, Tool calls: ${info.tool_calls}`);
```

## Quick Start: Evaluation Harness

The evaluation harness builds on the core to run evaluation suites on multiple tasks.

### 1. Create Tasks

Create a `tasks.jsonl` file with your evaluation tasks:

```json
{"id":"hello_01","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Say hello to Mike."}],"expected":{"final_contains":"Mike"},"limits":{"max_steps":5}}
{"id":"calc_01","messages":[{"role":"system","content":"Use the calc tool."},{"role":"user","content":"Compute 19*7 using calc, answer with only the number."}],"tools_allowed":["calc"],"expected":{"final_regex":"^133$"},"limits":{"max_steps":10}}
```

### 2. Run Evaluation

**Python:**
```python
from timestep import run_suite, report, agent_builtin_echo, DEFAULT_TOOLS
from timestep import FinalContains, ForbiddenTools, LLMJudge
from pathlib import Path

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

report(Path("runs/demo"))
```

**TypeScript:**
```typescript
import { runSuite, report, agentBuiltinEcho, DEFAULT_TOOLS } from '@timestep-ai/timestep';
import { FinalContains, ForbiddenTools } from '@timestep-ai/timestep';

runSuite(
  'tasks.jsonl',
  'runs/demo',
  agentBuiltinEcho,
  DEFAULT_TOOLS,
  [new FinalContains(), new ForbiddenTools()],
  3,
  0,
  120
);

report('runs/demo');
```

### 3. View Results

Results are saved in `runs/demo/`:
- `results.jsonl`: Summary of all trials
- `trials/`: Detailed transcripts and grades per trial

## Creating Your Own Agent Harness

The agent harness is the `AgentFn` interface - a function that enables a model to act as an agent. The agent-environment loop orchestrates this harness:

```python
from timestep import AgentFn, Message, JSON

def my_agent(messages: list[Message], context: JSON) -> Message:
    """
    Agent harness function.
    
    Args:
        messages: Full conversation history (transcript)
        context: Context with tools_schema, task, seed, limits
    
    Returns:
        Assistant message (may include tool_calls)
    """
    # Your agent logic
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

## Using Different Grader Types

### Code-Based Graders

Fast, objective, reproducible:

```python
from timestep import FinalRegex, FinalContains, MaxToolCalls

graders = [
    FinalRegex(pattern="^133$"),
    FinalContains(substring="Mike"),
    MaxToolCalls(max_calls=5),
]
```

### LLM-as-Judge

Nuanced, handles subjective tasks:

```python
from timestep import LLMJudge

graders = [
    LLMJudge(
        rubric="Is the response helpful, accurate, and friendly?",
        model="gpt-4o-mini",
        temperature=0.0,
        grade_transcript=False  # True for full transcript, False for final message
    )
]
```

### Outcome Verification

Checks environment state, not just transcript:

```python
from timestep import OutcomeVerifier

def check_database_state(messages, tool_index, task):
    """Verify actual state in environment."""
    # Check database, file system, etc.
    return True  # or False

graders = [
    OutcomeVerifier(verifier_fn=check_database_state)
]
```

## Next Steps

- Learn about [Architecture](architecture.md) - core vs eval modules
- Explore [Task Format](architecture.md#task-format)
- Check out [Examples](../../python/examples/) - core usage and eval examples
