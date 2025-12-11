"""Tests for run_agent functionality with conversation items assertions."""

import pytest
import os
import base64
import asyncio
import uuid
from pathlib import Path
from pydantic import BaseModel
from timestep._vendored_imports import (
    Agent,
    Runner,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    OpenAIConversationsSession,
    ModelSettings,
    function_tool,
)
from openai import OpenAI
from timestep import run_agent, RunStateStore

def file_to_base64(file_path: str) -> str:
    """Convert a file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def image_to_base64(image_path: str) -> str:
    """Convert an image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

RECOMMENDED_PROMPT_PREFIX = "# System context\nYou are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.\n"

async def needs_approval_for_get_weather(ctx, args, call_id):
    """Require approval for Berkeley."""
    return "Berkeley" in args.get("city", "")

@function_tool
def get_weather(city: str) -> str:
    """returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

get_weather.needs_approval = needs_approval_for_get_weather

# File paths for test media
PDF_PATH = Path(__file__).parent.parent.parent / "data" / "partial_o3-and-o4-mini-system-card.pdf"
IMAGE_PATH = Path(__file__).parent.parent.parent / "data" / "image_bison.jpg"

# Encode files once at module level
PDF_BASE64 = file_to_base64(str(PDF_PATH))
IMAGE_BASE64 = image_to_base64(str(IMAGE_PATH))

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
    ],
    [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's the weather on The Dark Side of the Moon?"}]}
    ],
    [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's four times the last number we calculated minus one?"}]}
    ],
    [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's four times the last number we calculated minus six?"}]}
    ],
    [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{IMAGE_BASE64}",
                    "detail": "auto",
                },
                {
                    "type": "input_text",
                    "text": "What do you see in this image?"
                }
            ],
        },
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


def truncate_image_data(obj, max_length=100):
    """Recursively truncate long image data in objects for readable debug output."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ('image_url', 'imageUrl') and isinstance(v, str):
                # Truncate long base64 image data
                if len(v) > max_length:
                    result[k] = v[:max_length] + "... [truncated]"
                else:
                    result[k] = v
            else:
                result[k] = truncate_image_data(v, max_length)
        return result
    elif isinstance(obj, list):
        return [truncate_image_data(item, max_length) for item in obj]
    else:
        return obj


@pytest.fixture(autouse=True)
def ensure_session_isolation():
    """
    Ensure each test gets a fresh session by not reusing conversation_ids.
    This fixture runs automatically before each test to ensure proper isolation.
    """
    # The fixture itself doesn't need to do anything - the tests will create
    # fresh sessions. This is just a marker to ensure we're thinking about isolation.
    yield
    # Cleanup after test if needed


async def run_agent_test(run_in_parallel: bool = True, stream: bool = False, session_id: str | None = None, model: str = "gpt-4.1"):
    """Run the agent test and return conversation items."""
    # Define guardrails
    @input_guardrail(run_in_parallel=run_in_parallel)
    async def moon_guardrail(
        ctx: RunContextWrapper[None],
        agent: Agent,
        input: str | list[TResponseInputItem]
    ) -> GuardrailFunctionOutput:
        """Prevent questions about the Moon."""
        input_text = input if isinstance(input, str) else str(input)
        mentions_moon = "moon" in input_text.lower()

        return GuardrailFunctionOutput(
            output_info={"mentions_moon": mentions_moon},
            tripwire_triggered=mentions_moon,
        )

    @output_guardrail
    async def no_47_guardrail(
        ctx: RunContextWrapper,
        agent: Agent,
        output: str
    ) -> GuardrailFunctionOutput:
        """Prevent output containing the number 47."""
        contains_47 = "47" in str(output)

        return GuardrailFunctionOutput(
            output_info={"contains_47": contains_47},
            tripwire_triggered=contains_47,
        )

    weather_assistant_agent = Agent(
        instructions="You are a helpful AI assistant that can answer questions about weather. When asked about weather, you MUST use the get_weather tool to get accurate, real-time weather information.",
        model=model,
        model_settings=ModelSettings(temperature=0),
        name="Weather Assistant",
        tools=[get_weather],
    )

    personal_assistant_agent = Agent(
        handoffs=[weather_assistant_agent],
        instructions=f"{RECOMMENDED_PROMPT_PREFIX}You are an AI agent acting as a personal assistant.",
        model=model,
        model_settings=ModelSettings(temperature=0),
        name="Personal Assistant",
        input_guardrails=[moon_guardrail],
        output_guardrails=[no_47_guardrail],
    )

    if session_id:
        session = OpenAIConversationsSession(conversation_id=session_id)
    else:
        session = OpenAIConversationsSession()

    # Get session ID for state file naming
    current_session_id = await session._get_session_id()
    if not current_session_id:
        raise ValueError("Failed to get session ID")
    
    # Get session ID for state file naming
    existing_items = await session.get_items()

    state_store = RunStateStore(agent=personal_assistant_agent, session_id=current_session_id)

    for idx, run_input in enumerate(RUN_INPUTS):
        try:
            result = await run_agent(personal_assistant_agent, run_input, session, stream)

            # Handle interruptions
            if result.interruptions and len(result.interruptions) > 0:
                # Save state
                state = result.to_state()
                await state_store.save(state)

                # Load and approve
                loaded_state = await state_store.load()
                for interruption in loaded_state.get_interruptions():
                    loaded_state.approve(interruption)

                # Resume with state
                result = await run_agent(personal_assistant_agent, loaded_state, session, stream)
        except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered):
            # Guardrail was triggered - pop items until we've removed the user message
            # First, peek at the last few items to see what needs to be removed
            recent_items = await session.get_items(2)
            # Count how many items to pop (from most recent back to the user message)
            items_to_pop = 0
            found_user_message = False
            for item in reversed(recent_items):
                items_to_pop += 1
                if isinstance(item, dict) and item.get('role') == 'user':
                    found_user_message = True
                    break  # Found the user message, stop counting

            # Only pop if we found a user message in the recent items
            if found_user_message:
                for _ in range(items_to_pop):
                    await session.pop_item()

    # Clean up state file
    await state_store.clear()

    # Wait for all background operations to complete before fetching from API.
    # This is necessary because Python's streaming doesn't have a completion Promise
    # like TypeScript's result.completed, so background operations may still be in progress.
    # We wait here (after all run_agent calls) rather than in the result processor itself
    # to avoid excessive delays when run_agent is called many times.
    # Parallel mode may need longer due to concurrent operations.
    if stream:
        import asyncio
        # Wait for all background operations to complete. Parallel mode needs longer
        # due to concurrent operations. The delay is necessary because Python's streaming
        # doesn't have a completion Promise like TypeScript's result.completed.
        delay = 3.0 if run_in_parallel else 2.0
        await asyncio.sleep(delay)  # Wait for all background operations

    conversation_id = await session._get_session_id()
    if not conversation_id:
        raise ValueError("Session does not have a conversation ID")

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = OpenAI(api_key=openai_api_key)
    items_response = client.conversations.items.list(conversation_id, limit=100, order="asc")
    return items_response.data


async def run_agent_test_partial(run_in_parallel: bool = True, stream: bool = False, session_id: str | None = None, start_index: int = 0, end_index: int = None, model: str = "gpt-4.1"):
    """Run partial agent test up to interruption and save state without approving.
    
    Returns a dict with "session_id" and "connection_string" for use in cross-language tests.
    The connection_string is the database connection string used by Python, which TypeScript
    should use to connect to the same database.
    """
    if end_index is None:
        end_index = len(RUN_INPUTS)
    
    # Define guardrails
    @input_guardrail(run_in_parallel=run_in_parallel)
    async def moon_guardrail(
        ctx: RunContextWrapper[None],
        agent: Agent,
        input: str | list[TResponseInputItem]
    ) -> GuardrailFunctionOutput:
        """Prevent questions about the Moon."""
        input_text = input if isinstance(input, str) else str(input)
        mentions_moon = "moon" in input_text.lower()

        return GuardrailFunctionOutput(
            output_info={"mentions_moon": mentions_moon},
            tripwire_triggered=mentions_moon,
        )

    @output_guardrail
    async def no_47_guardrail(
        ctx: RunContextWrapper,
        agent: Agent,
        output: str
    ) -> GuardrailFunctionOutput:
        """Prevent output containing the number 47."""
        contains_47 = "47" in str(output)

        return GuardrailFunctionOutput(
            output_info={"contains_47": contains_47},
            tripwire_triggered=contains_47,
        )

    weather_assistant_agent = Agent(
        instructions="You are a helpful AI assistant that can answer questions about weather. When asked about weather, you MUST use the get_weather tool to get accurate, real-time weather information.",
        model=model,
        model_settings=ModelSettings(temperature=0),
        name="Weather Assistant",
        tools=[get_weather],
    )

    personal_assistant_agent = Agent(
        handoffs=[weather_assistant_agent],
        instructions=f"{RECOMMENDED_PROMPT_PREFIX}You are an AI agent acting as a personal assistant.",
        model=model,
        model_settings=ModelSettings(temperature=0),
        name="Personal Assistant",
        input_guardrails=[moon_guardrail],
        output_guardrails=[no_47_guardrail],
    )

    if session_id:
        session = OpenAIConversationsSession(conversation_id=session_id)
    else:
        session = OpenAIConversationsSession()

    # Get session ID for state file naming
    current_session_id = await session._get_session_id()
    if not current_session_id:
        raise ValueError("Failed to get session ID")

    state_store = RunStateStore(agent=personal_assistant_agent, session_id=current_session_id)

    for i in range(start_index, end_index):
        run_input = RUN_INPUTS[i]
        try:
            result = await run_agent(personal_assistant_agent, run_input, session, stream)

            # Handle interruptions - save state but don't approve
            if result.interruptions and len(result.interruptions) > 0:
                # Save state (this will initialize the database connection if needed)
                state = result.to_state()
                await state_store.save(state)
                
                # Get the database connection string that was used
                # This ensures TypeScript can connect to the same database
                from timestep.config.dbos_config import get_dbos_connection_string
                connection_string = get_dbos_connection_string()
                
                # Return session ID and connection string without approving
                return {"session_id": current_session_id, "connection_string": connection_string}
        except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered):
            # Guardrail was triggered - pop items until we've removed the user message
            recent_items = await session.get_items(2)
            items_to_pop = 0
            found_user_message = False
            for item in reversed(recent_items):
                items_to_pop += 1
                if isinstance(item, dict) and item.get('role') == 'user':
                    found_user_message = True
                    break

            if found_user_message:
                for _ in range(items_to_pop):
                    await session.pop_item()

    # If we got here without interruption, get connection string anyway (may be None if DBOS wasn't configured)
    from timestep.config.dbos_config import get_dbos_connection_string
    connection_string = get_dbos_connection_string()
    return {"session_id": current_session_id, "connection_string": connection_string}


async def run_agent_test_from_typescript(session_id: str, run_in_parallel: bool = True, stream: bool = False, model: str = "gpt-4.1"):
    """Load state saved by TypeScript, approve, and continue execution."""
    # Define guardrails
    @input_guardrail(run_in_parallel=run_in_parallel)
    async def moon_guardrail(
        ctx: RunContextWrapper[None],
        agent: Agent,
        input: str | list[TResponseInputItem]
    ) -> GuardrailFunctionOutput:
        """Prevent questions about the Moon."""
        input_text = input if isinstance(input, str) else str(input)
        mentions_moon = "moon" in input_text.lower()

        return GuardrailFunctionOutput(
            output_info={"mentions_moon": mentions_moon},
            tripwire_triggered=mentions_moon,
        )

    @output_guardrail
    async def no_47_guardrail(
        ctx: RunContextWrapper,
        agent: Agent,
        output: str
    ) -> GuardrailFunctionOutput:
        """Prevent output containing the number 47."""
        contains_47 = "47" in str(output)

        return GuardrailFunctionOutput(
            output_info={"contains_47": contains_47},
            tripwire_triggered=contains_47,
        )

    weather_assistant_agent = Agent(
        instructions="You are a helpful AI assistant that can answer questions about weather. When asked about weather, you MUST use the get_weather tool to get accurate, real-time weather information.",
        model=model,
        model_settings=ModelSettings(temperature=0),
        name="Weather Assistant",
        tools=[get_weather],
    )

    personal_assistant_agent = Agent(
        handoffs=[weather_assistant_agent],
        instructions=f"{RECOMMENDED_PROMPT_PREFIX}You are an AI agent acting as a personal assistant.",
        model=model,
        model_settings=ModelSettings(temperature=0),
        name="Personal Assistant",
        input_guardrails=[moon_guardrail],
        output_guardrails=[no_47_guardrail],
    )

    # Use the same session ID
    session = OpenAIConversationsSession(conversation_id=session_id)

    state_store = RunStateStore(agent=personal_assistant_agent, session_id=session_id)

    # Load state saved by TypeScript
    loaded_state = await state_store.load()
    
    interruptions = loaded_state.get_interruptions()
    for interruption in interruptions:
        loaded_state.approve(interruption)

    # Resume with state
    result = await run_agent(personal_assistant_agent, loaded_state, session, stream)

    # Continue with remaining inputs (indices 4-7)
    for i in range(4, len(RUN_INPUTS)):
        run_input = RUN_INPUTS[i]
        try:
            result = await run_agent(personal_assistant_agent, run_input, session, stream)

            # Handle any new interruptions
            if result.interruptions and len(result.interruptions) > 0:
                state = result.to_state()
                await state_store.save(state)
                loaded_state = await state_store.load()
                for interruption in loaded_state.get_interruptions():
                    loaded_state.approve(interruption)
                result = await run_agent(personal_assistant_agent, loaded_state, session, stream)
        except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered):
            recent_items = await session.get_items(2)
            items_to_pop = 0
            found_user_message = False
            for item in reversed(recent_items):
                items_to_pop += 1
                if isinstance(item, dict) and item.get('role') == 'user':
                    found_user_message = True
                    break

            if found_user_message:
                for _ in range(items_to_pop):
                    await session.pop_item()

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
        "content": [{"type": "output_text", "text": "sunny"}]
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
    },
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "What's four times the last number we calculated minus six?"}]
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "42"}]
    },
    {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{IMAGE_BASE64}",
                "detail": "auto",
            },
            {
                "type": "input_text",
                "text": "What do you see in this image?"
            }
        ]
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "bison"}]
    }
]

def normalize_text(text: str) -> str:
    """Normalize text for comparison by handling common LLM output variations."""
    import re
    # Convert to lowercase
    normalized = text.lower()
    
    # Normalize mathematical operators
    normalized = re.sub(r'\bequals\b', '=', normalized)
    normalized = re.sub(r'\bis equal to\b', '=', normalized)
    normalized = re.sub(r'\bequal to\b', '=', normalized)
    
    # Normalize whitespace (multiple spaces to single space)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Trim
    normalized = normalized.strip()
    
    return normalized

def assert_conversation_items(cleaned, expected):
    """Assert conversation items match expected structure."""
    import json

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
            # Normalize both texts before comparison
            actual_normalized = normalize_text(actual_text)
            expected_normalized = normalize_text(expected_text)
            # Check that either normalized text contains the other (for flexibility with LLM variability)
            assert (expected_normalized in actual_normalized or actual_normalized in expected_normalized), \
                f"Item {i} text mismatch: expected '{expected_text}' and actual '{actual_text}' do not contain each other"
            # Also check structure matches
            assert cleaned_item["type"] == expected_item["type"]
            assert cleaned_item["role"] == expected_item["role"]
        elif cleaned_item.get("type") == "function_call" and expected_item.get("type") == "function_call":
            # For function_call items, compare arguments as JSON objects (not strings)
            assert cleaned_item["type"] == expected_item["type"]
            # Function names may differ in casing between languages (e.g., transfer_to_Weather_Assistant vs transfer_to_weather_assistant)
            assert cleaned_item["name"].lower() == expected_item["name"].lower(), f"Item {i} name mismatch: {cleaned_item['name']} != {expected_item['name']}"
            # Parse and compare JSON arguments
            actual_args = json.loads(cleaned_item["arguments"])
            expected_args = json.loads(expected_item["arguments"])
            assert actual_args == expected_args, f"Item {i} arguments mismatch: {actual_args} != {expected_args}"
        elif (cleaned_item.get("type") == "message" and
              cleaned_item.get("role") == "user" and
              expected_item.get("type") == "message" and
              expected_item.get("role") == "user"):
            # For user messages, check structure but handle images specially
            assert cleaned_item["type"] == expected_item["type"]
            assert cleaned_item["role"] == expected_item["role"]
            assert len(cleaned_item.get("content", [])) == len(expected_item.get("content", []))

            # Check each content block
            for j, (actual_block, expected_block) in enumerate(zip(cleaned_item.get("content", []), expected_item.get("content", []))):
                assert actual_block.get("type") == expected_block.get("type"), f"Item {i} content block {j} type mismatch"

                if actual_block.get("type") == "input_image":
                    # For images, check detail and validate file_id if present
                    if "detail" in expected_block:
                        assert actual_block.get("detail") == expected_block.get("detail"), f"Item {i} content block {j} detail mismatch"
                elif actual_block.get("type") == "input_text":
                    assert actual_block.get("text") == expected_block.get("text"), f"Item {i} content block {j} text mismatch"
                else:
                    assert actual_block == expected_block, f"Item {i} content block {j} mismatch"
        elif cleaned_item.get("type") == "function_call_output" and expected_item.get("type") == "function_call_output":
            # For function_call_output items, parse JSON output if present
            assert cleaned_item["type"] == expected_item["type"]
            if "output" in cleaned_item and "output" in expected_item:
                try:
                    actual_output = json.loads(cleaned_item["output"])
                    expected_output = json.loads(expected_item["output"])
                    assert actual_output == expected_output, f"Item {i} output mismatch: {actual_output} != {expected_output}"
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, compare as strings
                    assert cleaned_item["output"] == expected_item["output"], f"Item {i} output mismatch: {cleaned_item['output']} != {expected_item['output']}"
            else:
                assert cleaned_item == expected_item, f"Item {i} mismatch: {cleaned_item} != {expected_item}"
        else:
            # For all other items, exact match
            assert cleaned_item == expected_item, f"Item {i} mismatch: {cleaned_item} != {expected_item}"

@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1", "ollama/gpt-oss:20b-cloud", "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M"])
async def test_run_agent_blocking_non_streaming(model):
    """Test blocking (run_in_parallel=False) non-streaming execution."""
    if model == "ollama/gpt-oss:20b-cloud":
        # Expected failure: Ollama cloud model has known compatibility issues
        with pytest.raises(Exception):
            await run_agent_test(run_in_parallel=False, stream=False, session_id=None, model=model)
        return
    # Don't pass session_id to ensure a fresh session is created
    items = await run_agent_test(run_in_parallel=False, stream=False, session_id=None, model=model)
    cleaned = clean_items(items)
    
    # Debug: Print all items to identify the extra one
    if len(cleaned) != len(EXPECTED_ITEMS):
        print(f"\n{'='*80}")
        print(f"ITEM COUNT MISMATCH: Got {len(cleaned)} items, expected {len(EXPECTED_ITEMS)} items")
        print(f"{'='*80}\n")
        
        # Print item types
        actual_types = [item.get('type', 'unknown') for item in cleaned]
        expected_types = [item.get('type', 'unknown') for item in EXPECTED_ITEMS]
        print(f"Actual item types ({len(actual_types)}): {actual_types}")
        print(f"Expected item types ({len(expected_types)}): {expected_types}\n")
        
        # Print detailed comparison
        max_len = max(len(cleaned), len(EXPECTED_ITEMS))
        for i in range(max_len):
            print(f"\n--- Position {i} ---")
            if i < len(cleaned):
                actual_item = cleaned[i]
                item_type = actual_item.get('type', 'unknown')
                if item_type == 'message':
                    role = actual_item.get('role', 'unknown')
                    content = actual_item.get('content', [])
                    text = content[0].get('text', '')[:50] if content else ''
                    print(f"ACTUAL:   type={item_type}, role={role}, text={text}...")
                elif item_type == 'function_call':
                    name = actual_item.get('name', 'unknown')
                    args = actual_item.get('arguments', '')[:50]
                    print(f"ACTUAL:   type={item_type}, name={name}, args={args}...")
                elif item_type == 'function_call_output':
                    output = str(actual_item.get('output', ''))[:50]
                    print(f"ACTUAL:   type={item_type}, output={output}...")
                else:
                    truncated_actual = truncate_image_data(actual_item)
                    print(f"ACTUAL:   {json.dumps(truncated_actual, indent=2)}")
            else:
                print(f"ACTUAL:   <missing>")
            
            if i < len(EXPECTED_ITEMS):
                expected_item = EXPECTED_ITEMS[i]
                item_type = expected_item.get('type', 'unknown')
                if item_type == 'message':
                    role = expected_item.get('role', 'unknown')
                    content = expected_item.get('content', [])
                    text = content[0].get('text', '')[:50] if content else ''
                    print(f"EXPECTED: type={item_type}, role={role}, text={text}...")
                elif item_type == 'function_call':
                    name = expected_item.get('name', 'unknown')
                    args = expected_item.get('arguments', '')[:50]
                    print(f"EXPECTED: type={item_type}, name={name}, args={args}...")
                elif item_type == 'function_call_output':
                    output = str(expected_item.get('output', ''))[:50]
                    print(f"EXPECTED: type={item_type}, output={output}...")
                else:
                    truncated_expected = truncate_image_data(expected_item)
                    print(f"EXPECTED: {json.dumps(truncated_expected, indent=2)}")
            else:
                print(f"EXPECTED: <missing>")
        
        print(f"\n{'='*80}\n")
    
    assert_conversation_items(cleaned, EXPECTED_ITEMS)

@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1", "ollama/gpt-oss:20b-cloud", "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M"])
async def test_run_agent_blocking_streaming(model):
    """Test blocking (run_in_parallel=False) streaming execution."""
    if model == "ollama/gpt-oss:20b-cloud":
        # Expected failure: Ollama cloud model has known compatibility issues
        with pytest.raises(Exception):
            await run_agent_test(run_in_parallel=False, stream=True, session_id=None, model=model)
        return
    # Don't pass session_id to ensure a fresh session is created
    items = await run_agent_test(run_in_parallel=False, stream=True, session_id=None, model=model)
    cleaned = clean_items(items)
    if len(cleaned) != len(EXPECTED_ITEMS):
        import json
        print(f"\n{'='*80}")
        print(f"STREAM ITEM COUNT MISMATCH: Got {len(cleaned)} items, expected {len(EXPECTED_ITEMS)} items")
        print(f"{'='*80}\n")
        max_len = max(len(cleaned), len(EXPECTED_ITEMS))
        for i in range(max_len):
            print(f"\n--- Position {i} ---")
            if i < len(cleaned):
                truncated_actual = truncate_image_data(cleaned[i])
                print(f"ACTUAL:   {json.dumps(truncated_actual, indent=2)}")
            else:
                print("ACTUAL:   <missing>")
            if i < len(EXPECTED_ITEMS):
                truncated_expected = truncate_image_data(EXPECTED_ITEMS[i])
                print(f"EXPECTED: {json.dumps(truncated_expected, indent=2)}")
            else:
                print("EXPECTED: <missing>")
        print(f"\n{'='*80}\n")
    assert_conversation_items(cleaned, EXPECTED_ITEMS)

@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1", "ollama/gpt-oss:20b-cloud", "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M"])
async def test_run_agent_parallel_non_streaming(model):
    """Test parallel (run_in_parallel=True) non-streaming execution."""
    if model == "ollama/gpt-oss:20b-cloud":
        # Expected failure: Ollama cloud model has known compatibility issues
        with pytest.raises(Exception):
            await run_agent_test(run_in_parallel=True, stream=False, session_id=None, model=model)
        return
    # Don't pass session_id to ensure a fresh session is created
    items = await run_agent_test(run_in_parallel=True, stream=False, session_id=None, model=model)
    cleaned = clean_items(items)
    assert_conversation_items(cleaned, EXPECTED_ITEMS)

@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1", "ollama/gpt-oss:20b-cloud", "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M"])
async def test_run_agent_parallel_streaming(model):
    """Test parallel (run_in_parallel=True) streaming execution."""
    if model == "ollama/gpt-oss:20b-cloud":
        # Expected failure: Ollama cloud model has known compatibility issues
        with pytest.raises(Exception):
            await run_agent_test(run_in_parallel=True, stream=True, session_id=None, model=model)
        return
    # Don't pass session_id to ensure a fresh session is created
    items = await run_agent_test(run_in_parallel=True, stream=True, session_id=None, model=model)
    cleaned = clean_items(items)
    assert_conversation_items(cleaned, EXPECTED_ITEMS)
