"""Tests for run_agent functionality with conversation items assertions."""

import pytest
import os
from agents import Agent, OpenAIConversationsSession, ModelSettings, function_tool, Runner, TResponseInputItem
from openai import OpenAI
from timestep import run_agent

RECOMMENDED_PROMPT_PREFIX = "# System context\nYou are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.\n"

@function_tool
def get_weather(city: str) -> str:
    """returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

def clean_items(items):
    """Remove IDs, status, and call_id from conversation items and convert to dicts."""
    def to_dict(obj):
        if isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items() if k not in ('id', 'status', 'call_id')}
        if isinstance(obj, list):
            return [to_dict(item) for item in obj]
        if hasattr(obj, 'model_dump'):
            return {k: to_dict(v) for k, v in obj.model_dump().items() if k not in ('id', 'status', 'call_id')}
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

    run_input: list[TResponseInputItem] = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's the weather in Oakland?"}]}]
    await run_agent(personal_assistant_agent, run_input, session, stream)
    
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
        "content": [{"type": "output_text", "text": ""}]  # Text may vary
    }
]

def assert_conversation_items(cleaned, expected):
    """Assert conversation items match expected structure."""
    assert len(cleaned) == len(expected), f"Expected {len(expected)} items, got {len(cleaned)}"
    assert cleaned[0] == expected[0], f"First item mismatch: {cleaned[0]} != {expected[0]}"
    assert cleaned[1] == expected[1], f"Second item mismatch: {cleaned[1]} != {expected[1]}"
    assert cleaned[2] == expected[2], f"Third item mismatch: {cleaned[2]} != {expected[2]}"
    assert cleaned[3] == expected[3], f"Fourth item mismatch: {cleaned[3]} != {expected[3]}"
    assert cleaned[4] == expected[4], f"Fifth item mismatch: {cleaned[4]} != {expected[4]}"
    # Last message content may vary, just check structure
    assert cleaned[5]["type"] == expected[5]["type"]
    assert cleaned[5]["role"] == expected[5]["role"]
    assert "content" in cleaned[5]
    assert len(cleaned[5]["content"]) > 0

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
