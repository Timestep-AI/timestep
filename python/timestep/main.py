"""Main interactive client matching TypeScript version."""

import asyncio
import os
import sys
from agents import Agent, Runner, RunConfig
from agents import OpenAIProvider
from multi_provider import MultiProvider, MultiProviderMap
from ollama_model_provider import OllamaModelProvider
from mcp_server_proxy import fetch_mcp_tools


async def confirm(question: str) -> bool:
    """Prompt user for yes/no confirmation."""
    while True:
        answer = input(f"{question} (y/n): ").strip().lower()
        if answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False


async def run_test_client(
    user_input: str,
    model_id: str,
    openai_use_responses: bool = False
):
    """Run the test client with user input."""
    is_first_run = True
    if is_first_run:
        print('[MCP] Loading tools...')

    # Configure approval policies
    require_approval = {
        "never": {"toolNames": ["search_codex_code", "fetch_codex_documentation"]},
        "always": {"toolNames": ["get_weather", "fetch_generic_url_content"]},
    }

    # Fetch built-in tools for weather agent
    weather_tools = await fetch_mcp_tools(None, True, require_approval)

    # Create weather agent with built-in tools
    weather_agent = Agent(
        model=model_id,
        name='Weather agent',
        instructions='You provide weather information.',
        handoff_description='Handles weather-related queries',
        tools=weather_tools,
    )

    # Fetch remote MCP tools from the codex server
    mcp_tools = await fetch_mcp_tools('https://gitmcp.io/timestep-ai/timestep', False, require_approval)

    if is_first_run:
        print(f'[MCP] Loaded {len(weather_tools) + len(mcp_tools)} tools\n')

    # Create main agent with remote MCP tools and weather handoff
    agent = Agent(
        model=model_id,
        name='Main Assistant',
        instructions='You are a helpful assistant. For questions about the openai/codex repository, use the MCP tools. For weather questions, hand off to the weather agent.',
        tools=mcp_tools,
        handoffs=[weather_agent],
    )

    # Create model provider map
    model_provider_map = MultiProviderMap()

    model_provider_map.add_provider("ollama", OllamaModelProvider(
        api_key=os.getenv("OLLAMA_API_KEY"),
    ))

    # Add Anthropic provider using OpenAI interface
    model_provider_map.add_provider("anthropic", OpenAIProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://api.anthropic.com/v1/",
        use_responses=False
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

    # Run the agent
    result = Runner.run_streamed(
        agent,
        user_input,
        run_config=run_config
    )

    # Process the stream
    async for event in result.stream_events():
        # Handle different types of stream events
        if event.type == "agent_updated_stream_event":
            # AgentUpdatedStreamEvent indicates agent handoff
            if hasattr(event, 'new_agent'):
                print(f"✅ Handoff completed → {event.new_agent.name}")
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                # Try to get tool name from raw_item first, then fall back to name attribute
                tool_name = 'Unknown'
                if hasattr(event.item, 'raw_item') and hasattr(event.item.raw_item, 'name'):
                    tool_name = event.item.raw_item.name
                elif hasattr(event.item, 'name'):
                    tool_name = event.item.name

                print(f"🔧 Tool called: {tool_name}")
                if hasattr(event.item, 'arguments'):
                    print(f"   Arguments: {event.item.arguments}")
            elif event.item.type == "tool_call_output_item":
                # Try to get tool name from raw_item first, then fall back to name attribute
                tool_name = 'Unknown'
                if hasattr(event.item, 'raw_item') and hasattr(event.item.raw_item, 'name'):
                    tool_name = event.item.raw_item.name
                elif hasattr(event.item, 'name'):
                    tool_name = event.item.name

                print(f"✅ Tool output from {tool_name}:")
                print(f"   Result: {event.item.output}")
            elif event.item.type == "message_output_item":
                # Extract text from message output
                # Try raw_item.content first, then fall back to content
                content = None
                if hasattr(event.item, 'raw_item') and hasattr(event.item.raw_item, 'content'):
                    content = event.item.raw_item.content
                elif hasattr(event.item, 'content'):
                    content = event.item.content

                if content:
                    if isinstance(content, list):
                        for content_block in content:
                            if hasattr(content_block, 'text'):
                                print(content_block.text, end='')
                            elif hasattr(content_block, 'output_text'):
                                print(content_block.output_text, end='')
                    elif isinstance(content, str):
                        print(content, end='')

    print('\n')


async def main():
    """Main entry point."""
    model_id = os.getenv("MODEL_ID")
    openai_use_responses = (os.getenv("OPENAI_USE_RESPONSES") or "false").lower() == "true"

    if not model_id:
        print("Missing required env var MODEL_ID")
        print("Usage: MODEL_ID=\"gpt-5|anthropic/claude-sonnet-4-5|ollama/smollm2:1.7b|ollama/gpt-oss:120b-cloud\" uv run main.py")
        sys.exit(1)

    # Main loop for handling chat
    while True:
        # Prompt for user input
        try:
            current_input = input("You: ").strip()
            if not current_input:
                break
        except (EOFError, KeyboardInterrupt):
            print("\n\nConversation completed.")
            break

        print()

        try:
            await run_test_client(current_input, model_id, openai_use_responses)
        except Exception as error:
            print(f"\n❌ Error during run: {error}")

        # Ask if user wants to continue
        try:
            continue_chat = await confirm('Do you want to continue the conversation?')
            if not continue_chat:
                break
        except (EOFError, KeyboardInterrupt):
            print("\n\nConversation completed.")
            break

    print("\n\nConversation completed.")


if __name__ == "__main__":
    asyncio.run(main())
