#!/usr/bin/env python3
"""Example of creating a streaming agent harness using OpenAI's streaming API."""

import asyncio
from timestep import stream_episode, create_openai_streaming_agent, DEFAULT_TOOLS
from timestep.core.types import Message


async def main():
    """Example of using a streaming agent with stream_episode."""
    # Create streaming agent (requires OpenAI API key)
    try:
        agent = create_openai_streaming_agent()
    except ImportError:
        print("OpenAI not available. Install with: pip install openai")
        print("Setting OPENAI_API_KEY environment variable is required.")
        return
    
    # Define initial messages
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 5, saying each number on a new line."}
    ]
    
    print("Streaming agent response (chunks arrive in real-time):\n")
    
    # Stream the episode - chunks will arrive as the agent generates them
    async for event in stream_episode(
        initial_messages=messages,
        agent=agent,  # Streaming agent
        tools=DEFAULT_TOOLS,
        limits={"max_steps": 5, "time_limit_s": 30},
        task_meta={"id": "streaming_demo"},
        seed=0,
    ):
        event_type = event.get("type")
        
        if event_type == "TextMessageContent":
            # Content chunks arrive in real-time (like OpenAI streaming)
            print(event["delta"], end="", flush=True)
        elif event_type == "ToolCallChunk":
            print(f"\n[Tool call chunk: {event['chunk']}]")
        elif event_type == "TextMessageEnd":
            print(f"\n\n[Message complete]")
        elif event_type == "RunFinished":
            info = event["result"]["episodeInfo"]
            print(f"\n\nEpisode complete!")
            print(f"Steps: {info['steps']}, Duration: {info['duration_s']}s")


if __name__ == "__main__":
    asyncio.run(main())
