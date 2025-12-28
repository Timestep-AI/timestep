# Getting Started

This guide will help you get up and running with Timestep.

## Prerequisites

- `OPENAI_API_KEY` environment variable
- Python 3.11+

## Installation

Install Timestep using pip:

```bash
pip install timestep
```

Or from source:

```bash
cd python
pip install -e .
```

## Quick Start

Here's a simple example:

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

# Run the agent
async def main():
    messages = [{"role": "user", "content": "Hello!"}]
    result = await run_agent(agent, messages, session, stream=False)
    print(result["messages"])

asyncio.run(main())
```

## Next Steps

- Learn about [Guardrails](architecture.md#guardrails) for human-in-the-loop
- Explore [Handoffs](architecture.md#handoffs) for agent delegation
- See [Use Cases](use-cases.md) for more examples
