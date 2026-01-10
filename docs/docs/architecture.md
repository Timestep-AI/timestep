# Architecture

Timestep AI Agents SDK is built around a simple, universal **agent-environment loop** using OpenAI chat message protocol. The SDK is organized into two main modules: **core** (the agent-environment loop) and **eval** (the evaluation harness).

## Core vs Eval

### Core Module (`timestep.core`)

The core module provides the foundation - the agent-environment loop. It can be used independently without evaluation:

- **Agent harness interface**: `AgentFn` - function that takes messages and context, returns assistant message
- **Episode runner**: `run_episode()` - executes the agent-environment loop
- **Tool execution**: Deterministic tool execution with automatic indexing
- **Episode info**: Tracks steps, tool calls, duration, tokens, costs

### Eval Module (`timestep.eval`)

The eval module builds on core to provide evaluation capabilities:

- **Suite runner**: `run_suite()` - runs evaluation suites on multiple tasks
- **Graders**: Code-based, LLM-as-judge, and outcome verification graders
- **Reporting**: `report()` - generates summary reports

## Core Concepts

### Agent Harness (or Scaffold)

An **agent harness** is a system that enables a model to act as an agent. In Timestep, this is the `AgentFn` interface:

```python
AgentFn = Callable[[List[Message], JSON], Message]
```

- **Input**: List of messages (conversation history/transcript) + context (tools schema, task metadata, etc.)
- **Output**: Single assistant message (may include `tool_calls`)

The agent harness processes inputs, orchestrates tool calls, and returns results. It can use any model provider (OpenAI, Anthropic, local models, etc.) - Timestep doesn't care, as long as it follows the interface.

### Agent-Environment Loop

The `run_episode()` function implements the canonical agent-environment loop, which orchestrates the agent harness:

1. The loop calls the agent harness with messages and context
2. Agent harness returns assistant message
3. If assistant has `tool_calls`:
   - Environment executes each tool call
   - Appends tool result messages to transcript
   - Loop continues (returns to step 1)
4. If assistant has no `tool_calls`:
   - Episode terminates (final answer)

This loop continues until:
- Agent harness returns final answer (no tool calls)
- Maximum steps reached
- Time limit exceeded
- Error occurs

The agent-environment loop orchestrates the agent harness, executing tools and managing the conversation flow. Together, the loop and harness form the complete agent system.

### Transcript

The **transcript** is the complete record of an episode - all messages exchanged between agent and environment. This includes:
- System/user messages (initial input)
- Assistant messages (agent responses)
- Tool messages (tool execution results)

The transcript is preserved in full for analysis and grading.

### Outcome

The **outcome** is the final state in the environment, separate from the transcript. For example:
- Transcript: Agent says "Your flight has been booked"
- Outcome: Actual reservation exists in the database

Timestep supports outcome verification through the `OutcomeVerifier` grader, which checks environment state rather than just the transcript.

### Tool Execution

Tools are deterministic functions:

```python
ToolFn = Callable[[JSON], Any]
```

- Tools receive arguments as JSON
- Return values are JSON-serialized
- Tool calls are automatically indexed and paired with results
- Tool execution is synchronous and deterministic

### Episode Info

The `EpisodeInfo` object tracks metadata about a completed episode:

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

## Evaluation Harness

The **evaluation harness** builds on the core agent-environment loop to provide evaluation capabilities.

### Graders

Graders evaluate agent performance. They consume:
- `messages`: Full transcript
- `tool_index`: List of tool calls paired with results
- `task`: Task JSON (for expected values, allowlists, etc.)
- `info`: EpisodeInfo

And return:
```python
{
    "name": str,
    "passed": bool,
    "score": float,  # 0.0 to 1.0
    "details": {...}
}
```

### Grader Types

1. **Code-based graders**: Fast, objective, reproducible
   - String matching (regex, contains)
   - JSON validation
   - Tool usage checks
   - Transcript analysis

2. **LLM-as-judge graders**: Nuanced, handles subjective tasks
   - Uses OpenAI to grade based on rubric
   - Can grade final message or full transcript
   - Configurable model and temperature

3. **Outcome verification**: Checks environment state
   - Takes verifier function
   - Checks actual state, not just transcript
   - Useful for verifying side effects (database changes, file writes, etc.)

## Task Format

Tasks are JSON objects in JSONL files:

```json
{
  "id": "task_01",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "tools_allowed": ["calc", "echo"],
  "expected": {
    "final_regex": "^\\d+$",
    "final_contains": "result",
    "llm_judge_rubric": "Is the response helpful and accurate?"
  },
  "limits": {
    "max_steps": 10,
    "time_limit_s": 30
  }
}
```

### Task Fields

- `id`: Unique identifier (auto-generated if missing)
- `messages`: List of OpenAI-style messages
- `tools_allowed`: Optional tool allowlist
- `expected`: Optional expected values for graders
- `limits`: Optional episode limits

## Message Protocol

Messages follow OpenAI chat completion format:

- **System/User/Assistant messages**:
  ```json
  {
    "role": "system" | "user" | "assistant",
    "content": "string"
  }
  ```

- **Assistant with tool calls**:
  ```json
  {
    "role": "assistant",
    "content": "string",
    "tool_calls": [
      {
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "calc",
          "arguments": "{\"expr\":\"2+2\"}"
        }
      }
    ],
    "usage": {  // Optional: token usage info
      "prompt_tokens": 10,
      "completion_tokens": 5,
      "total_tokens": 15
    }
  }
  ```

- **Tool result messages**:
  ```json
  {
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "{\"value\":4}"
  }
  ```

## Built-in Graders

### Code-Based

1. **FinalRegex**: Checks final assistant content matches regex
2. **FinalContains**: Checks substring in final assistant content
3. **FinalJSON**: Parses final assistant content as JSON and checks required keys
4. **TranscriptContains**: Checks substring anywhere in transcript
5. **TranscriptRegex**: Regex match anywhere in transcript
6. **ForbiddenTools**: Fails if agent called tools not in allowlist
7. **MaxToolCalls**: Fails if > N tool calls
8. **MinToolCalls**: Fails if < N tool calls
9. **ToolCallSequence**: Checks a tool name was called at least once
10. **ToolCallOrder**: Verifies tool calls happened in expected sequence
11. **ToolResultJSON**: Checks tool result JSON has required keys

### LLM-as-Judge

- **LLMJudge**: Uses OpenAI to grade based on rubric

### Outcome Verification

- **OutcomeVerifier**: Checks environment state via verifier function

### Creating Custom Graders

```python
from timestep import Grader

class MyGrader(Grader):
    name = "MyGrader"
    
    def grade(self, messages, tool_index, task, info):
        # Your grading logic
        return {
            "name": self.name,
            "passed": True,
            "score": 1.0,
            "details": {}
        }
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

## Cross-Language Parity

Python and TypeScript implementations:
- Use same function/class names
- Same parameter names and types
- Same task format
- Same result format
- Compatible task JSONL files
- Core and eval modules have identical APIs
