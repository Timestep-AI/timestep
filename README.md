# Timestep AI Agents SDK

A universal **agents SDK** built around the core **agent-environment loop** using OpenAI-style chat message protocol. The SDK provides a clean foundation for building agents, with an **evaluation harness** as one powerful use case.

## Core Concepts

Timestep is built on a simple, universal pattern:

- **Agent harness** (or scaffold): System that enables a model to act as an agent - the `AgentFn` interface that takes messages and context, returns assistant messages. The harness processes inputs, orchestrates tool calls, and returns results.
- **Agent-environment loop**: Core execution pattern implemented by `run_episode()` that orchestrates the agent harness, executes tools, and manages the conversation flow. The loop runs the harness in a multi-turn pattern until completion.
- **Evaluation harness**: Infrastructure that runs evaluation suites end-to-end - one use case of the core SDK
- **Transcript**: Complete record of an episode (all messages)
- **Outcome**: Final state in environment (separate from transcript)

## What Timestep gives you

### Core SDK
- **Universal protocol**: OpenAI chat message format - works with any agent framework
- **Simple agent interface**: Just a function `(messages, context) => assistant_message`
- **Tool execution**: Deterministic tool execution with automatic indexing
- **Cross-language parity**: Same API in Python and TypeScript

### Evaluation Harness
- **Built-in graders**: Code-based (regex, contains, JSON), LLM-as-judge, outcome verification
- **Token tracking**: Automatic tracking of input/output tokens and costs
- **JSONL task format**: Simple, human-readable task definitions
- **CLI interface**: Run eval suites and generate reports

## Prerequisites

- **Python 3.11+** or **Node.js 20+**
- **OpenAI API key** (optional, for agents that use OpenAI or LLM-as-judge graders)

## Quick Start

### Using the Core Agent-Environment Loop

The core SDK lets you run the agent-environment loop (which orchestrates the agent harness) without evaluation:

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
    agent=agent_builtin_echo,  # Your agent harness (AgentFn)
    tools=DEFAULT_TOOLS,
    tools_allowed=["calc"],
    limits={"max_steps": 10, "time_limit_s": 30},
    task_meta={"id": "demo"},
    seed=0,
)

print(f"Steps: {info.steps}, Tool calls: {info.tool_calls}")
print(f"Final message: {transcript[-1]['content']}")
```

### Using the Evaluation Harness

The evaluation harness builds on the core to run evaluation suites:

```python
from timestep import run_suite, report, agent_builtin_echo, DEFAULT_TOOLS
from timestep import FinalContains, ForbiddenTools, LLMJudge
from pathlib import Path

# Create tasks file (tasks.jsonl)
tasks = [
    {
        "id": "hello_01",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello to Mike in one sentence."}
        ],
        "expected": {"final_contains": "Mike"},
        "limits": {"max_steps": 5, "time_limit_s": 30}
    }
]

# Write tasks to JSONL
import json
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
        # LLMJudge(rubric="Is the response friendly and appropriate?"),  # LLM-as-judge
    ],
    trials=3,
    seed=0,
    agent_timeout_s=120,
)

# Generate report
report(Path("runs/demo"))
```

### TypeScript

```typescript
import { runEpisode, agentBuiltinEcho, DEFAULT_TOOLS } from '@timestep-ai/timestep';
import type { Message } from '@timestep-ai/timestep';

// Core usage
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

## Agent Harness Interface

The agent harness is the `AgentFn` interface - a function that enables a model to act as an agent. The agent-environment loop orchestrates this harness:

```python
from timestep import AgentFn, Message, JSON

def my_agent(messages: list[Message], context: JSON) -> Message:
    """
    Agent harness function.
    
    Args:
        messages: Full conversation history (transcript so far)
        context: Context dict with tools_schema, task, seed, limits
    
    Returns:
        Assistant message (may include tool_calls)
    """
    # Your agent logic here
    # Use OpenAI, Anthropic, or any other provider
    return {
        "role": "assistant",
        "content": "Hello!",
        "tool_calls": [...]  # Optional
    }
```

### Built-in Agent Harnesses

- `agent_builtin_echo`: Echoes the last user message (for testing)
- `agent_cmd_factory`: Wraps external command as agent harness (AgentFn)

## Tools

Tools are deterministic functions that take arguments and return results:

```python
from timestep import ToolFn, JSON

def my_tool(args: JSON) -> Any:
    """Your tool logic."""
    return {"result": "value"}
```

### Built-in Tools

- `tool_calc`: Calculates arithmetic expressions (demo only, uses eval - not production-safe)
- `tool_echo`: Echoes back arguments

## Evaluation Graders

Graders evaluate agent performance. Timestep supports multiple grader types:

### Code-Based Graders

- `FinalRegex`: Checks final assistant content matches regex
- `FinalContains`: Checks substring in final assistant content
- `FinalJSON`: Parses final assistant content as JSON and checks required keys
- `TranscriptContains`: Checks substring anywhere in transcript
- `TranscriptRegex`: Regex match anywhere in transcript
- `ForbiddenTools`: Fails if agent called tools not in allowlist
- `MaxToolCalls`: Fails if > N tool calls
- `MinToolCalls`: Fails if < N tool calls
- `ToolCallSequence`: Checks a tool name was called at least once
- `ToolCallOrder`: Verifies tool calls happened in expected sequence
- `ToolResultJSON`: Checks tool result JSON has required keys

### LLM-as-Judge Grader

Uses an LLM to grade based on a rubric:

```python
from timestep import LLMJudge

graders = [
    LLMJudge(
        rubric="Is the response helpful, accurate, and friendly?",
        model="gpt-4o-mini",
        temperature=0.0,
        grade_transcript=False  # True to grade full transcript, False for final message only
    )
]
```

### Outcome Verification

Checks environment state, not just the transcript:

```python
from timestep import OutcomeVerifier

def check_database_state(messages, tool_index, task):
    """Verify the outcome in the environment (e.g., database state)."""
    # Check actual state, not just what agent said
    return True  # or False

graders = [
    OutcomeVerifier(verifier_fn=check_database_state)
]
```

## Task Format (JSONL)

Each line in the tasks file is a JSON object representing one task:

```json
{
  "id": "calc_01",
  "messages": [
    {"role": "system", "content": "You must use the calc tool."},
    {"role": "user", "content": "Compute 19*7 using the calc tool, then answer with only the number."}
  ],
  "tools_allowed": ["calc"],
  "expected": {
    "final_regex": "^133$",
    "final_contains": "133"
  },
  "limits": {"max_steps": 10, "time_limit_s": 30}
}
```

### Task Fields

- `id` (optional): Unique task identifier. Auto-generated if missing.
- `messages`: List of OpenAI-style messages (system/user/assistant/tool).
- `tools_allowed` (optional): Allowlist of tool names the agent can use.
- `expected` (optional): Expected values for graders (e.g., `final_regex`, `final_contains`, `llm_judge_rubric`).
- `limits` (optional): Episode limits (`max_steps`, `time_limit_s`).

## Episode Info

The `EpisodeInfo` object tracks episode metadata:

```python
@dataclass
class EpisodeInfo:
    task_id: str
    trial: int
    seed: int
    steps: int
    tool_calls: int
    duration_s: float
    terminated_reason: str  # final_answer | max_steps | time_limit | error
    error: Optional[str] = None
    # Token tracking (if agent provides usage info)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
```

## Output Structure

Running an evaluation suite creates:

```
runs/demo/
├── run_meta.json          # Run metadata
├── results.jsonl          # One line per trial
└── trials/
    └── task_id/
        └── trial_XX/
            ├── transcript.json    # Full message transcript
            ├── tool_index.json    # Tool calls paired with results
            ├── grades.json        # Grader results
            └── info.json          # Episode info (steps, duration, tokens, etc.)
```

## CLI Usage

### Python

```bash
# Install
pip install timestep

# Run eval suite
timestep run \
  --tasks tasks.jsonl \
  --outdir runs/demo \
  --agent builtin:echo \
  --trials 3 \
  --graders FinalContains ForbiddenTools

# Generate report
timestep report --outdir runs/demo
```

### TypeScript

```bash
# Build first
cd typescript
pnpm build

# Run eval suite
node dist/eval/cli.js run \
  --tasks tasks.jsonl \
  --outdir runs/demo \
  --agent builtin:echo \
  --trials 3 \
  --graders FinalContains ForbiddenTools

# Generate report
node dist/eval/cli.js report --outdir runs/demo
```

## Architecture

Timestep is organized into two main modules:

- **`core/`**: Core agent-environment loop - orchestrates the agent harness, can be used independently
- **`eval/`**: Evaluation harness - builds on core to run evaluation suites

The core module provides:
- Agent harness interface (`AgentFn`) - enables a model to act as an agent
- Episode runner (`run_episode`) - orchestrates the agent harness in the agent-environment loop
- Tool execution and indexing
- Episode info tracking (including tokens)

The eval module provides:
- Suite runner (`run_suite`)
- Graders (code-based, LLM-as-judge, outcome verification)
- Reporting (`report`)

## Cross-Language Compatibility

Tasks created in Python work in TypeScript and vice versa. The JSONL format and result structures are identical.

## Examples

See `python/examples/` and `typescript/examples/` for complete examples:
- Core agent-environment loop usage
- Evaluation harness usage
- Custom agent harnesses
- Custom graders

## Documentation

- Full docs: https://timestep-ai.github.io/timestep/
- Python notes: `python/README.md`
- TypeScript notes: `typescript/README.md`

## License

MIT License - see `LICENSE`.
