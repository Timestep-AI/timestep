#!/usr/bin/env python3
"""Example of creating a custom agent harness using OpenAI."""

from timestep import AgentFn, Message, JSON, run_episode, DEFAULT_TOOLS


def create_openai_agent(api_key: str = None) -> AgentFn:
    """
    Creates an agent harness function that uses OpenAI.
    
    This is a simple example - in production you'd want better error handling,
    retry logic, streaming support, etc.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    def agent(messages: list[Message], context: JSON) -> Message:
        """
        Agent harness that calls OpenAI API.
        
        Args:
            messages: Full conversation history (transcript)
            context: Context with tools_schema, task, seed, limits
        
        Returns:
            Assistant message (may include tool_calls and usage info)
        """
        # Get tools schema from context
        tools = context.get("tools_schema", [])
        
        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
        )
        
        # Extract assistant message
        choice = response.choices[0]
        message = choice.message
        
        # Build assistant message in Timestep format
        assistant_msg: Message = {
            "role": "assistant",
            "content": message.content or "",
        }
        
        # Add tool calls if present
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]
        
        # Add usage info for token tracking
        if response.usage:
            assistant_msg["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return assistant_msg
    
    return agent


def main():
    """Example usage of custom agent harness."""
    # Create agent harness
    # In production, get API key from environment: os.getenv("OPENAI_API_KEY")
    try:
        agent = create_openai_agent()
    except ImportError:
        print("OpenAI not available, using builtin echo agent instead")
        from timestep import agent_builtin_echo
        agent = agent_builtin_echo
    
    # Run a single episode
    messages: list[Message] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Calculate 5 + 3 using the calc tool."}
    ]
    
    transcript, info = run_episode(
        initial_messages=messages,
        agent=agent,
        tools=DEFAULT_TOOLS,
        tools_allowed=["calc"],
        limits={"max_steps": 10, "time_limit_s": 30},
        task_meta={"id": "demo"},
        seed=0,
    )
    
    print(f"Episode completed in {info.steps} steps")
    print(f"Tool calls: {info.tool_calls}")
    if info.total_tokens > 0:
        print(f"Tokens used: {info.total_tokens} (input: {info.input_tokens}, output: {info.output_tokens})")
    print(f"Final message: {transcript[-1].get('content', '')}")


if __name__ == "__main__":
    main()
