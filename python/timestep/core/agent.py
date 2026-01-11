"""Agent harness interface and adapters."""

import asyncio
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from .types import AgentFn, JSON, Message, StreamingAgentFn


def agent_builtin_echo(messages: List[Message], context: JSON) -> Message:
    """Builtin agent harness that finishes immediately by echoing the last user message."""
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = str(m.get("content", "") or "")
            break
    return {"role": "assistant", "content": f"Echo: {last_user}"}


def _agent_cmd_factory(agent_cmd: str, timeout_s: int = 120):
    """
    Internal function for CLI use only - creates an agent harness that shells out to a command.
    Not exported from the package.
    """
    import json
    import shlex
    import subprocess
    import time
    from .types import AgentFn
    
    args = shlex.split(agent_cmd)

    def _agent(messages: List[Message], context: JSON) -> Message:
        payload = {
            "messages": messages,
            "tools": context.get("tools_schema", []),
            "task": context.get("task", {}),
            "seed": context.get("seed"),
            "limits": context.get("limits", {}),
        }
        t0 = time.time()
        try:
            proc = subprocess.run(
                args,
                input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"role": "assistant", "content": "", "tool_calls": [], "error": "agent_timeout"}

        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()

        if not stdout:
            return {"role": "assistant", "content": "", "tool_calls": [], "error": "agent_empty_stdout", "stderr": stderr}

        try:
            msg = json.loads(stdout)
        except json.JSONDecodeError:
            msg = {"role": "assistant", "content": stdout}

        if stderr:
            msg.setdefault("_agent_stderr", stderr)
        msg.setdefault("_agent_latency_s", round(time.time() - t0, 4))
        return msg

    return _agent


def create_agent(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout_s: int = 120,
) -> AgentFn:
    """
    Creates an agent harness function that uses OpenAI-compatible streaming API.
    
    This is the single agent function for the Timestep library. It uses streaming internally
    but returns a synchronous AgentFn compatible with run_episode().
    Supports any OpenAI-compatible API (OpenAI, Anthropic via proxy, local models, etc.)
    via the base_url parameter.
    
    Args:
        api_key: API key (defaults to OPENAI_API_KEY env var)
        base_url: Base URL for API (defaults to OPENAI_BASE_URL env var or OpenAI default)
        model: Model name to use
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout_s: Request timeout in seconds
    
    Returns:
        AgentFn that can be used with run_episode() and run_suite()
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    # Get API key and base_url from environment if not provided
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    
    # Create client
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    
    async def _streaming_agent(messages: List[Message], context: JSON) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming agent harness that calls OpenAI-compatible streaming API.
        
        Args:
            messages: Full conversation history (transcript)
            context: Context with tools_schema, task, seed, limits
        
        Yields:
            Chunks in Timestep format
        """
        # Get tools schema from context
        tools = context.get("tools_schema", [])
        
        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"
        
        try:
            # Call API with streaming
            stream = client.chat.completions.create(**request_params, timeout=timeout_s)
            
            # Track accumulated message state
            accumulated_content = ""
            accumulated_tool_calls = {}  # tool_call_id -> tool_call dict
            
            for chunk in stream:
                # Check for usage information (OpenAI provides this in the final chunk)
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = chunk.usage
                    yield {
                        "type": "usage",
                        "usage": {
                            "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                            "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                            "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                        }
                    }
                
                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Content delta
                if delta.content:
                    accumulated_content += delta.content
                    yield {"type": "content", "delta": delta.content}
                
                # Tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        tc_id = tc_delta.id
                        if tc_id not in accumulated_tool_calls:
                            accumulated_tool_calls[tc_id] = {
                                "id": tc_id,
                                "type": tc_delta.type or "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        
                        tc = accumulated_tool_calls[tc_id]
                        
                        # Function name delta
                        if tc_delta.function and tc_delta.function.name:
                            tc["function"]["name"] = tc_delta.function.name
                        
                        # Function arguments delta
                        if tc_delta.function and tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments
                            yield {
                                "type": "tool_call",
                                "delta": {
                                    "id": tc_id,
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": tc_delta.function.arguments,
                                    },
                                },
                            }
            
            # Signal completion
            yield {"type": "done"}
            
        except Exception as e:
            yield {"type": "error", "error": str(e)}
    
    # Wrap the streaming agent to make it synchronous
    return _streaming_to_sync_agent(_streaming_agent)


def _streaming_to_sync_agent(streaming_agent: StreamingAgentFn) -> AgentFn:
    """
    Convert a streaming agent to a synchronous agent by consuming the stream.
    
    This allows streaming agents to work with run_episode() which expects AgentFn.
    """
    def _sync_agent(messages: List[Message], context: JSON) -> Message:
        """Synchronous wrapper that consumes the streaming agent's output."""
        # Create a new event loop or get the running one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async generator
        async def _consume_stream():
            accumulated_content = ""
            accumulated_tool_calls = []
            current_tool_call = None
            usage_info = None
            
            async for chunk in streaming_agent(messages, context):
                if chunk.get("type") == "content":
                    accumulated_content += chunk.get("delta", "")
                elif chunk.get("type") == "tool_call":
                    delta = chunk.get("delta", {})
                    tc_id = delta.get("id")
                    func_delta = delta.get("function", {})
                    
                    # Find or create tool call
                    if current_tool_call is None or current_tool_call.get("id") != tc_id:
                        if current_tool_call:
                            accumulated_tool_calls.append(current_tool_call)
                        current_tool_call = {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": func_delta.get("name", ""),
                                "arguments": func_delta.get("arguments", ""),
                            }
                        }
                    else:
                        # Append to existing tool call arguments
                        current_tool_call["function"]["arguments"] += func_delta.get("arguments", "")
                elif chunk.get("type") == "usage":
                    # Capture usage information
                    usage_info = chunk.get("usage", {})
                elif chunk.get("type") == "done":
                    break
                elif chunk.get("type") == "error":
                    return {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [],
                        "error": chunk.get("error", "unknown_error")
                    }
            
            # Add final tool call if exists
            if current_tool_call:
                accumulated_tool_calls.append(current_tool_call)
            
            # Build final message
            msg: Message = {
                "role": "assistant",
                "content": accumulated_content,
            }
            
            if accumulated_tool_calls:
                msg["tool_calls"] = accumulated_tool_calls
            
            # Include usage information if available
            if usage_info:
                msg["usage"] = usage_info
            
            return msg
        
        # Run the async function
        # Note: This will fail if called from an async context with a running loop
        # For run_episode() this is fine since it's synchronous
        if loop.is_running():
            raise RuntimeError("Cannot use create_agent() from within an async context with a running event loop. Use stream_episode() instead.")
        return loop.run_until_complete(_consume_stream())
    
    return _sync_agent
