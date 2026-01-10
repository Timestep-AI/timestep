#!/usr/bin/env python3
"""Example usage of Timestep core agent-environment loop (without evaluation)."""

from timestep import run_episode, agent_builtin_echo, DEFAULT_TOOLS
from timestep.core.types import Message


def main():
    """Run a single episode using the core agent-environment loop."""
    # Define initial messages
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Calculate 5 + 3 using the calc tool, then tell me the result."}
    ]
    
    # Run a single episode
    transcript, info = run_episode(
        initial_messages=messages,
        agent=agent_builtin_echo,  # Your agent harness
        tools=DEFAULT_TOOLS,
        tools_allowed=["calc"],
        limits={"max_steps": 10, "time_limit_s": 30},
        task_meta={"id": "demo"},
        seed=0,
    )
    
    # Print results
    print("Episode completed!")
    print(f"Steps: {info.steps}")
    print(f"Tool calls: {info.tool_calls}")
    print(f"Duration: {info.duration_s:.2f}s")
    print(f"Terminated reason: {info.terminated_reason}")
    if info.input_tokens > 0:
        print(f"Tokens: {info.total_tokens} (input: {info.input_tokens}, output: {info.output_tokens})")
    
    print("\nTranscript:")
    for i, msg in enumerate(transcript):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"{i+1}. {role}: {content[:100]}")  # Truncate long content


if __name__ == "__main__":
    main()
