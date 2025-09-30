import os
import json
from typing import AsyncGenerator
from ag_ui.core import (
    RunAgentInput,
    BaseEvent,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    TextMessageStartEvent,
    TextMessageChunkEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallResultEvent,
    ToolCallEndEvent,
)
from agents import Agent, Runner, function_tool, RunConfig
from multi_provider import MultiProvider, MultiProviderMap
from ollama_model_provider import OllamaModelProvider
from agents import OpenAIProvider


class TimestepAgent:
    """AG-UI compatible agent that wraps the OpenAI agents framework."""

    def __init__(self, model_id: str = None, openai_use_responses: bool = False):
        self.model_id = model_id or os.getenv("MODEL_ID") or "ollama/gpt-oss:120b-cloud"
        self.openai_use_responses = openai_use_responses

    async def run(self, input: RunAgentInput) -> AsyncGenerator[BaseEvent, None]:
        """
        Run the agent and yield AG-UI events.

        Args:
            input: RunAgentInput containing threadId, runId, and messages

        Yields:
            BaseEvent objects conforming to AG-UI protocol
        """
        try:
            # Emit RUN_STARTED event
            yield RunStartedEvent(thread_id=input.thread_id, run_id=input.run_id)

            # Run the agent and emit events
            async for event in self._run_agent(input):
                yield event

            # Emit RUN_FINISHED event
            yield RunFinishedEvent(thread_id=input.thread_id, run_id=input.run_id)

        except Exception as error:
            # Emit RUN_ERROR event
            yield RunErrorEvent(message=str(error))
            raise

    async def _run_agent(self, input: RunAgentInput) -> AsyncGenerator[BaseEvent, None]:
        """Internal method to run the agent and map events to AG-UI protocol."""

        # Set up model providers
        model_provider_map = MultiProviderMap()

        model_provider_map.add_provider(
            "ollama",
            OllamaModelProvider(api_key=os.getenv("OLLAMA_API_KEY")),
        )

        model_provider_map.add_provider(
            "anthropic",
            OpenAIProvider(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url="https://api.anthropic.com/v1/",
                use_responses=False,
            ),
        )

        model_provider = MultiProvider(
            provider_map=model_provider_map,
            openai_use_responses=self.openai_use_responses,
        )

        run_config = RunConfig(
            model_provider=model_provider,
            trace_include_sensitive_data=True,
            tracing_disabled=False,
        )

        # Create a simple tool for demonstration (no approval for AG-UI integration)
        @function_tool
        def get_weather(city: str) -> str:
            """Get the weather for a given city"""
            return f"The weather in {city} is sunny."

        weather_agent = Agent(
            model=self.model_id,
            name="Weather agent",
            instructions="You provide weather information.",
            handoff_description="Handles weather-related queries",
            tools=[get_weather],
        )

        agent = Agent(
            model=self.model_id,
            name="Main agent",
            instructions="You are a general assistant. For weather questions, call the weather agent tool with a short input string and then answer.",
            handoffs=[weather_agent],
            tools=[],
        )

        runner = Runner()

        # Get the last user message as input
        messages = input.messages or []
        user_messages = [m for m in messages if m.role == "user"]
        last_user_message = user_messages[-1] if user_messages else None
        user_input = last_user_message.content if last_user_message else ""

        if not user_input:
            # No user input, just return without doing anything
            return

        result = Runner.run_streamed(agent, user_input, run_config=run_config)

        current_message_id = None
        message_started = False

        async for event in result.stream_events():
            # Check for message output events that contain text
            if event.type == "run_item_stream_event":
                item = event.item

                if item.type == "message_output_item":
                    # Get text content from message
                    from agents import ItemHelpers

                    text_content = ItemHelpers.text_message_output(item)
                    if text_content:
                        # Initialize message ID and send TEXT_MESSAGE_START
                        if not current_message_id:
                            current_message_id = getattr(item, "id", str(id(item)))
                            yield TextMessageStartEvent(message_id=current_message_id)
                            message_started = True

                        # Send TEXT_MESSAGE_CHUNK
                        yield TextMessageChunkEvent(
                            message_id=current_message_id, delta=text_content
                        )

                # Handle tool call events
                elif item.type == "tool_call_item":
                    # Get tool call information from raw_item
                    raw_item = item.raw_item
                    tool_call_id = raw_item.call_id
                    tool_call_name = raw_item.name
                    tool_args = raw_item.arguments

                    yield ToolCallStartEvent(
                        tool_call_id=tool_call_id, tool_call_name=tool_call_name
                    )

                    # Convert args to JSON string (handle both dict and string)
                    if isinstance(tool_args, dict):
                        args_str = json.dumps(tool_args)
                    elif isinstance(tool_args, str):
                        args_str = tool_args
                    else:
                        args_str = str(tool_args)

                    yield ToolCallArgsEvent(tool_call_id=tool_call_id, delta=args_str)

                elif item.type == "tool_call_output_item":
                    # Get tool output information
                    raw_item = item.raw_item
                    tool_call_id = (
                        raw_item["call_id"]
                        if isinstance(raw_item, dict)
                        else raw_item.call_id
                    )
                    result_text = item.output
                    message_id = str(id(item))

                    yield ToolCallResultEvent(
                        message_id=message_id,
                        tool_call_id=tool_call_id,
                        content=result_text,
                    )

                    yield ToolCallEndEvent(tool_call_id=tool_call_id)

        # Send TEXT_MESSAGE_END if we started a message
        if message_started and current_message_id:
            yield TextMessageEndEvent(message_id=current_message_id)
