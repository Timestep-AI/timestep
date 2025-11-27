"""Tests for run_agent functionality with conversation items assertions."""

import pytest
import os
from pathlib import Path
from agents import Agent, OpenAIConversationsSession, ModelSettings, function_tool, Runner, TResponseInputItem
from openai import OpenAI
from timestep import run_agent, consume_result, RunStateStore

RECOMMENDED_PROMPT_PREFIX = "# System context\nYou are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.\n"

async def needs_approval_for_get_weather(ctx, args, call_id):
    """Require approval for Berkeley."""
    return "Berkeley" in args.get("city", "")

@function_tool
def get_weather(city: str) -> str:
    """returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

get_weather.needs_approval = needs_approval_for_get_weather

RUN_INPUTS: list[list[TResponseInputItem]] = [
    [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's 2+2?"}]}
    ],
    [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's the weather in Oakland?"}]}
    ],
    [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's three times that number you calculated earlier?"}]}
    ],
    [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's the weather in Berkeley?"}]}
    ]
]

def clean_items(items):
    """Remove IDs, status, call_id, annotations, and logprobs from conversation items and convert to dicts."""
    def to_dict(obj):
        if isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items() if k not in ('id', 'status', 'call_id', 'annotations', 'logprobs')}
        if isinstance(obj, list):
            return [to_dict(item) for item in obj]
        if hasattr(obj, 'model_dump'):
            return {k: to_dict(v) for k, v in obj.model_dump().items() if k not in ('id', 'status', 'call_id', 'annotations', 'logprobs')}
        return obj
    
    return [to_dict(item) for item in items]


async def run_agent_test(stream: bool = False):
    """Run the agent test and return conversation items."""
    weather_assistant_agent = Agent(
        instructions="You are a helpful AI assistant that can answer questions about weather. When asked about weather, you MUST use the get_weather tool to get accurate, real-time weather information.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name="Weather Assistant",
        tools=[get_weather],
    )

    personal_assistant_agent = Agent(
        handoffs=[weather_assistant_agent],
        instructions=f"{RECOMMENDED_PROMPT_PREFIX}You are an AI agent acting as a personal assistant.",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0),
        name="Personal Assistant",
    )

    session = OpenAIConversationsSession()

    # Get session ID for state file naming
    session_id = await session._get_session_id()
    if not session_id:
        raise ValueError("Failed to get session ID")

    state_file_path = Path(__file__).parent.parent.parent / "data" / f"agent_state_{session_id}.json"
    state_store = RunStateStore(str(state_file_path), personal_assistant_agent)

    for run_input in RUN_INPUTS:
        result = await run_agent(personal_assistant_agent, run_input, session, stream)
        result = await consume_result(result)

        # Handle interruptions
        if result.interruptions:
            # Save state
            state = result.to_state()
            await state_store.save(state)

            # Load and approve
            loaded_state = await state_store.load()
            for interruption in loaded_state.get_interruptions():
                loaded_state.approve(interruption)

            # Resume with state
            result = await run_agent(personal_assistant_agent, loaded_state, session, stream)
            result = await consume_result(result)

    # Clean up state file
    await state_store.clear()

    conversation_id = await session._get_session_id()
    if not conversation_id:
        raise ValueError("Session does not have a conversation ID")
    
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    client = OpenAI(api_key=openai_api_key)
    items_response = client.conversations.items.list(conversation_id, limit=100, order="asc")
    return items_response.data

EXPECTED_ITEMS = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "What's 2+2?"}]
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "2 + 2 = 4."}]
    },
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "What's the weather in Oakland?"}]
    },
    {
        "type": "function_call",
        "name": "transfer_to_weather_assistant",
        "arguments": "{}"
    },
    {
        "type": "function_call_output",
        "output": '{"assistant": "Weather Assistant"}'
    },
    {
        "type": "function_call",
        "name": "get_weather",
        "arguments": '{"city":"Oakland"}'
    },
    {
        "type": "function_call_output",
        "output": "The weather in Oakland is sunny"
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "The weather in Oakland is currently sunny. If you need more details like temperature or forecast, let me know!"}]
    },
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "What's three times that number you calculated earlier?"}]
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "12"}]
    },
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "What's the weather in Berkeley?"}]
    },
    {
        "type": "function_call",
        "name": "transfer_to_weather_assistant",
        "arguments": "{}"
    },
    {
        "type": "function_call_output",
        "output": '{"assistant": "Weather Assistant"}'
    },
    {
        "type": "function_call",
        "name": "get_weather",
        "arguments": '{"city":"Berkeley"}'
    },
    {
        "type": "function_call_output",
        "output": "The weather in Berkeley is sunny"
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "sunny"}]
    }
]

def assert_conversation_items(cleaned, expected):
    """Assert conversation items match expected structure."""
    assert len(cleaned) == len(expected), f"Expected {len(expected)} items, got {len(cleaned)}"
    for i, (cleaned_item, expected_item) in enumerate(zip(cleaned, expected)):
        # For assistant messages with output_text, check that actual text contains expected text
        if (cleaned_item.get("type") == "message" and 
            cleaned_item.get("role") == "assistant" and 
            expected_item.get("type") == "message" and 
            expected_item.get("role") == "assistant"):
            # Extract text from both actual and expected
            actual_text = " ".join([block.get("text", "") for block in cleaned_item.get("content", []) if block.get("type") == "output_text"])
            expected_text = " ".join([block.get("text", "") for block in expected_item.get("content", []) if block.get("type") == "output_text"])
            # Check that actual contains expected (case-insensitive for flexibility)
            assert expected_text.lower() in actual_text.lower(), f"Item {i} text mismatch: expected '{expected_text}' to be contained in '{actual_text}'"
            # Also check structure matches
            assert cleaned_item["type"] == expected_item["type"]
            assert cleaned_item["role"] == expected_item["role"]
        else:
            # For all other items, exact match
            assert cleaned_item == expected_item, f"Item {i} mismatch: {cleaned_item} != {expected_item}"

@pytest.mark.asyncio
async def test_run_agent_non_streaming():
    """Test non-streaming execution and assert conversation items."""
    items = await run_agent_test(stream=False)
    cleaned = clean_items(items)
    assert_conversation_items(cleaned, EXPECTED_ITEMS)

@pytest.mark.asyncio
async def test_run_agent_streaming():
    """Test streaming execution and assert conversation items."""
    items = await run_agent_test(stream=True)
    cleaned = clean_items(items)
    assert_conversation_items(cleaned, EXPECTED_ITEMS)
