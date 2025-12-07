"""Example demonstrating DBOS workflows for durable agent execution."""

import asyncio
import os
from timestep import (
    run_agent_workflow,
    queue_agent_workflow,
    create_scheduled_agent_workflow,
    configure_dbos,
    ensure_dbos_launched,
    cleanup_dbos,
)
from timestep._vendored_imports import (
    Agent,
    OpenAIConversationsSession,
    ModelSettings,
    TResponseInputItem,
)


async def example_durable_workflow():
    """Example: Run an agent in a durable workflow."""
    print("=== Example 1: Durable Workflow ===")
    
    # Configure and launch DBOS
    await configure_dbos()
    await ensure_dbos_launched()
    
    # Create agent and session
    agent = Agent(
        instructions="You are a helpful assistant.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name="Assistant",
    )
    session = OpenAIConversationsSession()
    
    # Prepare input
    input_items: list[TResponseInputItem] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's 2+2?"}]}
    ]
    
    # Run in durable workflow
    result = await run_agent_workflow(
        agent=agent,
        input_items=input_items,
        session=session,
        stream=False,
        workflow_id="example-workflow-1"  # Idempotency key
    )
    
    print(f"Result: {result.output}")
    print()


async def example_queued_workflow():
    """Example: Enqueue agent runs with rate limiting."""
    print("\n=== Example 2: Queued Workflow ===")
    
    await ensure_dbos_launched()
    
    agent = Agent(
        instructions="You are a helpful assistant.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name="Assistant",
    )
    session = OpenAIConversationsSession()
    
    input_items: list[TResponseInputItem] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's 3+3?"}]}
    ]
    
    # Enqueue workflow (returns handle immediately)
    handle = await queue_agent_workflow(
        agent=agent,
        input_items=input_items,
        session=session,
        stream=False,
        priority=1,  # Higher priority
        deduplication_id="example-queue-1"  # Prevent duplicates
    )
    
    # Wait for result when ready
    result = await handle.get_result()
    print(f"Result: {result.output}")
    print()


async def example_scheduled_workflow():
    """Example: Schedule periodic agent runs."""
    print("\n=== Example 3: Scheduled Workflow ===")
    
    await ensure_dbos_launched()
    
    agent = Agent(
        instructions="You are a helpful assistant.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name="Assistant",
    )
    session = OpenAIConversationsSession()
    
    input_items: list[TResponseInputItem] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's 4+4?"}]}
    ]
    
    # Create scheduled workflow (runs every 5 minutes)
    # Note: In production, you'd typically run this in a long-lived process
    await create_scheduled_agent_workflow(
        crontab="*/5 * * * *",  # Every 5 minutes
        agent=agent,
        input_items=input_items,
        session=session,
        stream=False
    )
    
    print("Scheduled workflow created. It will run every 5 minutes.")
    print("Note: Keep the process running for scheduled workflows to execute.")
    print()


async def main():
    """Run all examples."""
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Examples may fail.")
    
    try:
        await example_durable_workflow()
        await example_queued_workflow()
        await example_scheduled_workflow()
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up DBOS resources
        await cleanup_dbos()


if __name__ == "__main__":
    asyncio.run(main())

