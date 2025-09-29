import json
import asyncio
import aiofiles
import time
import os
from typing import Dict, Any, List
from agents import Agent, Runner, function_tool, ModelProvider, RunConfig, ItemHelpers
from agents import OpenAIResponsesModel
from multi_provider import MultiProvider, MultiProviderMap
from ollama_provider import OllamaModelProvider

async def main(model_id: str, openai_use_responses: bool = False):
    
    # Define a tool that requires approval for certain inputs
    @function_tool
    def get_weather(city: str) -> str:
        """Get the weather for a given city"""
        return f"The weather in {city} is sunny."

    weather_agent = Agent(
        name='Weather agent',
        instructions='You provide weather information.',
        handoff_description='Handles weather-related queries',
        model=model_id,
        tools=[get_weather]
    )

    main_agent = Agent(
        name='Main agent',
        instructions='You are a general assistant. For weather questions, call the weather agent tool with a short input string and then answer.',
        model=model_id,
        handoffs=[weather_agent],
        tools=[]
    )

    # Create model provider map
    model_provider_map = MultiProviderMap()
    
    model_provider_map.add_provider("ollama", OllamaModelProvider(
        api_key=os.getenv("OLLAMA_API_KEY"),
    ))
    
    # Use MultiProvider for model selection
    model_provider = MultiProvider(
        provider_map=model_provider_map,
        openai_use_responses=openai_use_responses
    )
    
    run_config = RunConfig(
        model_provider=model_provider,
        trace_include_sensitive_data=True,
        tracing_disabled=False
    )

    runner = Runner()
    
    result = Runner.run_streamed(
        main_agent,
        'What is the weather and temperature in San Francisco and Oakland? Use available tools as needed.',
        run_config=run_config
    )

    # Create filename in data folder (without timestamp)
    model_name = main_agent.model.replace(':', '_')  # Replace colon with underscore for filename
    model_name_for_file = model_name.replace('/', '_')

    # Only include openai_use_responses flag for OpenAI models (no slash in model name)
    is_openai_model = '/' not in main_agent.model
    filename = f"data/{model_name_for_file}.{openai_use_responses}.jsonl" if is_openai_model else f"data/{model_name_for_file}.jsonl"
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Create a single file stream for writing JSONL
    async with aiofiles.open(filename, 'w') as file_stream:
        print("=== Run starting ===")
        event_count = 0
        async for event in result.stream_events():
            # Deep-serialize events to JSON-compatible structures
            def deep_serialize(obj):
                # primitives
                if obj is None or isinstance(obj, (bool, int, float, str)):
                    return obj
                # lists/tuples
                if isinstance(obj, (list, tuple)):
                    return [deep_serialize(x) for x in obj]
                # dicts
                if isinstance(obj, dict):
                    return {k: deep_serialize(v) for k, v in obj.items()}
                # pydantic model-like (openai types expose model_dump)
                model_dump = getattr(obj, 'model_dump', None)
                if callable(model_dump):
                    try:
                        return deep_serialize(model_dump())
                    except Exception:
                        pass
                # generic objects with __dict__
                if hasattr(obj, '__dict__'):
                    try:
                        return {k: deep_serialize(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
                    except Exception:
                        return str(obj)
                # fallback
                return str(obj)

            event_dict = {
                'type': getattr(event, 'type', None),
                'timestamp': getattr(event, 'timestamp', None),
            }
            if hasattr(event, 'data'):
                event_dict['data'] = deep_serialize(getattr(event, 'data'))
            # Optional standard fields occasionally present on events
            if hasattr(event, 'new_agent'):
                event_dict['new_agent'] = deep_serialize(getattr(event, 'new_agent'))
            if hasattr(event, 'item'):
                event_dict['item'] = deep_serialize(getattr(event, 'item'))
            if hasattr(event, 'response'):
                event_dict['response'] = deep_serialize(getattr(event, 'response'))
            await file_stream.write(json.dumps(event_dict) + '\n')

            # Handle different types of stream events with improved formatting
            if event.type == "raw_response_event":
                continue  # Ignore raw response events
            elif event.type == "agent_updated_stream_event":
                print(f"🔄 Agent updated: {event.new_agent.name}")
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print("🔧 Tool was called")
                elif event.item.type == "tool_call_output_item":
                    print(f"✅ Tool output: {event.item.output}")
                elif event.item.type == "message_output_item":
                    print(f"💬 Message output:\n{ItemHelpers.text_message_output(event.item)}")
                else:
                    pass  # Ignore other event types
            else:
                # Uncomment the line below to see all events
                # print(f"📝 Event: {event.type}")
                pass

    print('\n\nDone')

if __name__ == "__main__":
    async def run_all_models():
        # Run with gpt-5 (openai_use_responses=False - default)
        print("=== Running with gpt-5 (openai_use_responses=False) ===")
        await main("gpt-5")

        # Run with gpt-5 (openai_use_responses=True)
        print("\n=== Running with gpt-5 (openai_use_responses=True) ===")
        await main("gpt-5", openai_use_responses=True)

        # Then run with smollm2:1.7b
        print("\n=== Running with ollama/smollm2:1.7b ===")
        await main("ollama/smollm2:1.7b")

        # Finally run with gpt-oss:120b-cloud
        print("\n=== Running with ollama/gpt-oss:120b-cloud ===")
        await main("ollama/gpt-oss:120b-cloud")

    asyncio.run(run_all_models())
