"""Agent harness interface and adapters."""

import json
import shlex
import subprocess
import time
from typing import Any, AsyncIterator, Dict, List

from .types import AgentFn, JSON, Message, StreamingAgentFn
from ..utils.messages import is_assistant_message


def agent_builtin_echo(messages: List[Message], context: JSON) -> Message:
    """Builtin agent harness that finishes immediately by echoing the last user message."""
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = str(m.get("content", "") or "")
            break
    return {"role": "assistant", "content": f"Echo: {last_user}"}


def agent_cmd_factory(agent_cmd: str, timeout_s: int = 120) -> AgentFn:
    """
    Creates an agent harness function that shells out to `agent_cmd`.
    
    Protocol:
      - send JSON to stdin: {"messages":[...], "tools":[...], "task":{...}, "seed":..., "limits":...}
      - expect stdout JSON: assistant message dict
    """
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
            # Treat empty output as a final assistant message (failure is visible via graders)
            return {"role": "assistant", "content": "", "tool_calls": [], "error": "agent_empty_stdout", "stderr": stderr}

        try:
            msg = json.loads(stdout)
        except json.JSONDecodeError:
            # If agent prints plain text, treat as assistant content
            msg = {"role": "assistant", "content": stdout}

        # Attach diagnostics if present
        if stderr:
            msg.setdefault("_agent_stderr", stderr)
        msg.setdefault("_agent_latency_s", round(time.time() - t0, 4))
        return msg

    return _agent


def create_openai_streaming_agent(api_key: str = None) -> StreamingAgentFn:
    """
    Creates a streaming agent harness function that uses OpenAI's streaming API.
    
    Yields chunks in Timestep format:
    - {type: "content", delta: str} - content chunk
    - {type: "tool_call", delta: {...}} - tool call chunk (partial)
    - {type: "done"} - agent response complete
    - {type: "error", error: str} - error occurred
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    async def _streaming_agent(messages: List[Message], context: JSON) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming agent harness that calls OpenAI streaming API.
        
        Args:
            messages: Full conversation history (transcript)
            context: Context with tools_schema, task, seed, limits
        
        Yields:
            Chunks in Timestep format
        """
        # Get tools schema from context
        tools = context.get("tools_schema", [])
        
        try:
            # Call OpenAI with streaming
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                stream=True,
            )
            
            # Track accumulated message state
            accumulated_content = ""
            accumulated_tool_calls = {}  # tool_call_id -> tool_call dict
            
            for chunk in stream:
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
    
    return _streaming_agent
