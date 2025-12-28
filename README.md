# Timestep AI

MVP Agent System with Human-in-the-Loop, Guardrails, Handoffs, and Sessions.

## What Timestep gives you

- **Agents with Human-in-the-Loop**: Request human approval during tool execution via guardrails
- **Guardrails**: Pre and post-execution validation and modification of tool inputs/outputs
- **Handoffs**: Agent-to-agent delegation via handoff tools
- **Sessions**: File-based conversation persistence
- **Custom Execution Loop**: Direct OpenAI API integration without SDK dependencies

## Prerequisites

- `OPENAI_API_KEY`
- Python 3.11+

## Quick start

### Python

```python
import asyncio
from timestep import (
    Agent,
    FileSession,
    InputGuardrail,
    OutputGuardrail,
    GuardrailInterrupt,
    run_agent,
    default_result_processor,
)

# Define a simple tool
async def get_weather(args: dict) -> dict:
    """Get weather information for a city."""
    city = args.get("city", "unknown")
    return {"result": f"The weather in {city} is sunny and 72°F"}

# Define input guardrail that requires approval for certain cities
async def city_approval_guardrail(tool_name: str, args: dict) -> "GuardrailResult":
    """Require approval for sensitive cities."""
    from timestep import GuardrailResult, GuardrailInterrupt
    
    city = args.get("city", "").lower()
    if city in ["berkeley", "san francisco"]:
        raise GuardrailInterrupt(
            prompt=f"Approval required to get weather for {city}. Approve? (y/n): ",
            tool_name=tool_name,
            args=args
        )
    return GuardrailResult.proceed()

# Define output guardrail
async def output_safety_guardrail(tool_name: str, args: dict, result: dict) -> "GuardrailResult":
    """Ensure safe output."""
    from timestep import GuardrailResult
    
    result_text = str(result.get("result", ""))
    if "password" in result_text.lower() or "api key" in result_text.lower():
        return GuardrailResult.block("Output contains sensitive information")
    return GuardrailResult.proceed()

# Create specialized weather agent
weather_agent = Agent(
    name="Weather Assistant",
    model="gpt-4o",
    instructions="You are a weather assistant. Use get_weather for all weather queries.",
    tools=[get_weather],
)

# Create main assistant with handoffs and guardrails
assistant = Agent(
    name="Personal Assistant",
    model="gpt-4o",
    instructions="You are a helpful personal assistant. For weather queries, use the transfer_to_weather_assistant tool.",
    tools=[],
    handoffs=[weather_agent],
    guardrails=[
        InputGuardrail(city_approval_guardrail),
        OutputGuardrail(output_safety_guardrail),
    ],
)

async def main():
    # Create session
    session = FileSession(
        agent_name=assistant.name,
        conversation_id="test-conversation",
        agent_instructions=assistant.instructions,
    )
    
    # Run agent (result_processor defaults to default_result_processor)
    messages = [{"role": "user", "content": "What's the weather in Berkeley?"}]
    
    result = await run_agent(assistant, messages, session, stream=False)
    
    print("Messages:", result["messages"])
    print("Tool calls:", result["tool_calls"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Concepts

### Agents

An `Agent` has:
- `name`: Agent identifier
- `model`: OpenAI model name (e.g., "gpt-4o")
- `instructions`: System instructions for the agent
- `tools`: List of callable tool functions
- `handoffs`: List of agents this agent can handoff to
- `guardrails`: List of InputGuardrail/OutputGuardrail instances

### Guardrails

Guardrails validate and modify tool execution:

- **InputGuardrail**: Runs before tool execution
  - Can block execution
  - Can modify input arguments
  - Can raise `GuardrailInterrupt` for human approval

- **OutputGuardrail**: Runs after tool execution
  - Can modify output results
  - Can block results

### Handoffs

Handoffs allow agents to delegate to other agents. When an agent has `handoffs=[other_agent]`, a tool `transfer_to_{other_agent_name}` is automatically created. The target agent manages its own session.

### Sessions

`FileSession` stores conversations as JSONL files:
- One file per agent+conversation
- Persists across runs
- Simple file-based storage (no database required)

## License

MIT License - see `LICENSE`.
