# Timestep (Python)

Python implementation of Timestep AI - MVP Agent System.

## Install

```bash
pip install timestep
```

Or from source:

```bash
cd python
pip install -e .
```

## Prerequisites

- `OPENAI_API_KEY` environment variable
- Python 3.11+

## Quick start

```python
import asyncio
from timestep import Agent, FileSession, run_agent

# Create an agent
agent = Agent(
    name="Assistant",
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    tools=[],
)

# Create a session
session = FileSession(
    agent_name=agent.name,
    conversation_id="my-conversation",
    agent_instructions=agent.instructions,
)

# Run the agent (result_processor defaults to default_result_processor)
async def main():
    messages = [{"role": "user", "content": "Hello!"}]
    result = await run_agent(agent, messages, session, stream=False)
    print(result["messages"])

asyncio.run(main())
```

## API Reference

### Agent

```python
from timestep import Agent

agent = Agent(
    name: str,                    # Agent name
    model: str,                   # OpenAI model (e.g., "gpt-4o")
    instructions: str,            # System instructions
    tools: List[Tool] = [],       # List of tool functions
    handoffs: List[Agent] = [],   # Agents to handoff to
    guardrails: List = [],        # InputGuardrail/OutputGuardrail instances
)
```

### FileSession

```python
from timestep import FileSession

session = FileSession(
    agent_name: str,              # Agent name
    conversation_id: str,         # Unique conversation ID
    agent_instructions: str = None,  # Optional system instructions
    storage_dir: str = "conversations",  # Storage directory
)
```

### Guardrails

```python
from timestep import InputGuardrail, OutputGuardrail, GuardrailResult, GuardrailInterrupt

# Input guardrail
async def my_input_guardrail(tool_name: str, args: dict) -> GuardrailResult:
    # Check/modify args before tool execution
    if some_condition:
        raise GuardrailInterrupt("Need approval", tool_name, args)
    return GuardrailResult.proceed()

# Output guardrail
async def my_output_guardrail(tool_name: str, args: dict, result: dict) -> GuardrailResult:
    # Check/modify result after tool execution
    return GuardrailResult.proceed()

input_guardrail = InputGuardrail(my_input_guardrail)
output_guardrail = OutputGuardrail(my_output_guardrail)
```

### Running Agents

```python
from timestep import run_agent, default_result_processor

# Run agent (returns async iterator of events)
events = run_agent(
    agent: Agent,
    messages: List[ChatMessage],
    session: Session,
    stream: bool = False,
)

# Process events
result = await default_result_processor(events)
```

## Examples

See the root `README.md` for a complete example with guardrails and handoffs.
