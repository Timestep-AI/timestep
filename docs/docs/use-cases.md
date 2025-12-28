# Use Cases

This document covers common patterns and use cases for Timestep MVP.

## Human-in-the-Loop with Guardrails

Request human approval during tool execution using guardrails.

```python
import asyncio
from timestep import (
    Agent,
    FileSession,
    InputGuardrail,
    GuardrailInterrupt,
    GuardrailResult,
    run_agent,
)

# Define a tool
async def send_email(args: dict) -> dict:
    """Send an email."""
    recipient = args.get("recipient", "")
    subject = args.get("subject", "")
    return {"result": f"Email sent to {recipient} with subject {subject}"}

# Define guardrail that requires approval for external emails
async def email_approval_guardrail(tool_name: str, args: dict) -> GuardrailResult:
    """Require approval for external email domains."""
    recipient = args.get("recipient", "")
    if "@" in recipient and not recipient.endswith("@company.com"):
        raise GuardrailInterrupt(
            prompt=f"Approval required to send email to external address {recipient}. Approve? (y/n): ",
            tool_name=tool_name,
            args=args
        )
    return GuardrailResult.proceed()

# Create agent with guardrail
agent = Agent(
    name="Email Assistant",
    model="gpt-4o",
    instructions="You are an email assistant. Use send_email to send emails.",
    tools=[send_email],
    guardrails=[InputGuardrail(email_approval_guardrail)],
)

async def main():
    session = FileSession(
        agent_name=agent.name,
        conversation_id="email-conversation",
        agent_instructions=agent.instructions,
    )
    
    messages = [{"role": "user", "content": "Send an email to alice@external.com with subject 'Hello'"}]
    result = await run_agent(agent, messages, session, stream=False)
    
    print("Result:", result)

asyncio.run(main())
```

## Agent Handoffs

Delegate tasks to specialized agents using handoffs.

```python
import asyncio
from timestep import Agent, FileSession, run_agent, default_result_processor

# Weather agent
async def get_weather(args: dict) -> dict:
    city = args.get("city", "unknown")
    return {"result": f"The weather in {city} is sunny and 72°F"}

weather_agent = Agent(
    name="Weather Assistant",
    model="gpt-4o",
    instructions="You are a weather assistant. Use get_weather for all weather queries.",
    tools=[get_weather],
)

# Main assistant with handoff
assistant = Agent(
    name="Personal Assistant",
    model="gpt-4o",
    instructions="You are a helpful assistant. For weather queries, use transfer_to_weather_assistant.",
    tools=[],
    handoffs=[weather_agent],
)

async def main():
    session = FileSession(
        agent_name=assistant.name,
        conversation_id="main-conversation",
        agent_instructions=assistant.instructions,
    )
    
    messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]
    result = await run_agent(assistant, messages, session, stream=False)
    
    print("Messages:", result["messages"])

asyncio.run(main())
```

## Output Guardrails

Modify or validate tool outputs before returning to the agent.

```python
import asyncio
from timestep import (
    Agent,
    FileSession,
    OutputGuardrail,
    GuardrailResult,
    run_agent,
)

# Tool that might return sensitive data
async def get_user_info(args: dict) -> dict:
    user_id = args.get("user_id", "")
    # Simulated sensitive data
    return {
        "result": f"User {user_id}: email=user@example.com, password=secret123"
    }

# Output guardrail to sanitize sensitive data
async def sanitize_output_guardrail(tool_name: str, args: dict, result: dict) -> GuardrailResult:
    """Remove sensitive information from output."""
    result_text = str(result.get("result", ""))
    if "password" in result_text.lower():
        # Remove password from output
        sanitized = result_text.replace("password=secret123", "password=***")
        return GuardrailResult.modify_result({"result": sanitized})
    return GuardrailResult.proceed()

agent = Agent(
    name="User Info Assistant",
    model="gpt-4o",
    instructions="You are a user information assistant.",
    tools=[get_user_info],
    guardrails=[OutputGuardrail(sanitize_output_guardrail)],
)

async def main():
    session = FileSession(
        agent_name=agent.name,
        conversation_id="user-info-conversation",
        agent_instructions=agent.instructions,
    )
    
    messages = [{"role": "user", "content": "Get info for user 123"}]
    result = await run_agent(agent, messages, session, stream=False)
    
    print("Result:", result)

asyncio.run(main())
```

## Session Persistence

Use FileSession to maintain conversation context across runs.

```python
import asyncio
from timestep import Agent, FileSession, run_agent, default_result_processor

agent = Agent(
    name="Assistant",
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    tools=[],
)

async def main():
    # Create session with persistent storage
    session = FileSession(
        agent_name=agent.name,
        conversation_id="my-conversation",
        agent_instructions=agent.instructions,
    )
    
    # First message
    messages1 = [{"role": "user", "content": "Hello, my name is Alice"}]
    result1 = await run_agent(agent, messages1, session, stream=False)
    print("First response:", result1["messages"])
    
    # Second message - session maintains context
    messages2 = [{"role": "user", "content": "What's my name?"}]
    result2 = await run_agent(agent, messages2, session, stream=False)
    print("Second response:", result2["messages"])
    # Agent remembers: "Alice"

asyncio.run(main())
```

## Streaming Responses

Process streaming responses for real-time updates.

```python
import asyncio
from timestep import Agent, FileSession, run_agent

agent = Agent(
    name="Assistant",
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    tools=[],
)

async def main():
    session = FileSession(
        agent_name=agent.name,
        conversation_id="streaming-conversation",
        agent_instructions=agent.instructions,
    )
    
    messages = [{"role": "user", "content": "Tell me a story"}]
    # For streaming, pass result_processor=None to get raw events
    events = run_agent(agent, messages, session, stream=True, result_processor=None)
    
    # Process streaming events
    async for event in events:
        if event.get("type") == "content_delta":
            print(event.get("content"), end="", flush=True)
        elif event.get("type") == "message":
            print("\n\nFinal message:", event.get("content"))

asyncio.run(main())
```

## Best Practices

1. **Environment Variables**: Always use environment variables for API keys
2. **Error Handling**: Implement proper error handling for tool execution
3. **Session Management**: Use FileSession for multi-turn conversations
4. **Guardrails**: Use guardrails for safety and human-in-the-loop patterns
5. **Handoffs**: Use handoffs to delegate to specialized agents
6. **Streaming**: Use streaming for better UX in interactive applications
