#!/usr/bin/env python3
"""Example of using stream_episode with FastAPI for real-time agent execution."""

import asyncio
from timestep import stream_episode, agent_builtin_echo, DEFAULT_TOOLS
from timestep.core.types import Message


async def main():
    """Example of consuming the streaming episode."""
    # Define initial messages
    messages: list[Message] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Calculate 5 + 3 using the calc tool."}
    ]
    
    # Stream the episode
    async for event in stream_episode(
        initial_messages=messages,
        agent=agent_builtin_echo,  # Can also use streaming agents
        tools=DEFAULT_TOOLS,
        tools_allowed=["calc"],
        limits={"max_steps": 10, "time_limit_s": 30},
        task_meta={"id": "demo"},
        seed=0,
    ):
        event_type = event.get("type")
        
        if event_type == "step_start":
            print(f"Step {event['step']} started")
        elif event_type == "content_delta":
            # Stream content chunks in real-time
            print(f"Content chunk: {event['delta']}", end="", flush=True)
        elif event_type == "tool_call_delta":
            print(f"\nTool call chunk: {event['delta']}")
        elif event_type == "agent_response_complete":
            print(f"\nAgent response complete: {event['message'].get('content', '')[:50]}...")
        elif event_type == "tool_call_start":
            print(f"Tool call started: {event['tool_call'].get('function', {}).get('name', 'unknown')}")
        elif event_type == "tool_call_result":
            print(f"Tool call result: {event['result']}")
        elif event_type == "step_complete":
            print(f"Step {event['step']} completed")
        elif event_type == "episode_complete":
            info = event["info"]
            print(f"\nEpisode complete!")
            print(f"Steps: {info.steps}, Tool calls: {info.tool_calls}")
            print(f"Duration: {info.duration_s}s")
            if info.total_tokens > 0:
                print(f"Tokens: {info.total_tokens} (input: {info.input_tokens}, output: {info.output_tokens})")


# Example FastAPI integration:
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/agent/stream")
async def stream_agent(request: dict):
    async def generate():
        async for event in stream_episode(
            initial_messages=request["messages"],
            agent=agent_builtin_echo,
            tools=DEFAULT_TOOLS,
            tools_allowed=request.get("tools_allowed"),
            limits=request.get("limits", {}),
            task_meta=request.get("task_meta", {}),
            seed=request.get("seed", 0),
        ):
            # Format as Server-Sent Events
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
"""

if __name__ == "__main__":
    asyncio.run(main())
