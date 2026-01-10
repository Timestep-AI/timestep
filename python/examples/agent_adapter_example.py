#!/usr/bin/env python3
"""Example agent adapter using OpenAI library."""

import json
import sys
from openai import OpenAI

# Agent adapter that reads from stdin and writes to stdout
def main():
    """Agent adapter main function."""
    # Read payload from stdin
    payload = json.load(sys.stdin)
    messages = payload.get("messages", [])
    tools_schema = payload.get("tools", [])
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools_schema if tools_schema else None,
    )
    
    # Extract assistant message
    message = response.choices[0].message
    
    # Format as OpenAI-style assistant message
    assistant_msg = {
        "role": "assistant",
        "content": message.content or "",
    }
    
    # Add tool calls if present
    if message.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in message.tool_calls
        ]
    
    # Write to stdout
    print(json.dumps(assistant_msg))


if __name__ == "__main__":
    main()
