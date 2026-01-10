"""Episode runner - the core agent-environment loop."""

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from .types import AgentFn, JSON, Message, StreamingAgentFn, ToolFn
from .tools import build_tools_schema
from ..utils.messages import is_assistant_message
from ..utils.io import now


@dataclass
class EpisodeInfo:
    """Information about a completed episode."""
    task_id: str
    trial: int
    seed: int
    steps: int
    tool_calls: int
    duration_s: float
    terminated_reason: str  # final_answer | max_steps | time_limit | error
    error: Optional[str] = None
    # Token and cost tracking (optional, populated if agent provides usage info)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


def _extract_usage_from_message(msg: Message) -> Dict[str, Any]:
    """Extract token usage from agent message if present."""
    usage = msg.get("usage") or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


async def _run_episode_stream(
    initial_messages: List[Message],
    agent: Union[AgentFn, StreamingAgentFn],
    tools: Dict[str, ToolFn],
    tools_allowed: Optional[List[str]] = None,
    limits: Optional[JSON] = None,
    task_meta: Optional[JSON] = None,
    seed: int = 0,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Internal implementation that yields events as the episode progresses.
    Supports both streaming and non-streaming agents.
    """
    task_id = str((task_meta or {}).get("id", "unknown"))
    max_steps = int((limits or {}).get("max_steps", 30))
    time_limit_s = float((limits or {}).get("time_limit_s", 120))

    messages = list(initial_messages)
    tool_allow = set(tools_allowed) if tools_allowed is not None else None

    t0 = now()
    steps = 0
    tool_calls = 0
    terminated_reason = "error"
    err: Optional[str] = None
    
    # Token tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    # Provide schema to agent via context (optional)
    tools_schema = build_tools_schema(tools, tools_allowed)

    for step in range(max_steps):
        if now() - t0 > time_limit_s:
            terminated_reason = "time_limit"
            break

        yield {"type": "step_start", "step": step + 1}

        context = {
            "tools_schema": tools_schema,
            "task": task_meta or {},
            "seed": seed,
            "limits": limits or {},
        }

        # Handle streaming vs non-streaming agents
        assistant_msg: Message = {"role": "assistant", "content": ""}
        
        # Call agent - check if it returns an async iterator (streaming) or a Message (non-streaming)
        result = agent(messages, context)
        
        # Check if result is an async iterator (streaming agent)
        if hasattr(result, "__aiter__"):
            # It's a streaming agent - consume the stream
            accumulated_content = ""
            accumulated_tool_calls: Dict[str, Dict[str, Any]] = {}
            tool_call_ids: List[str] = []
            
            async for chunk in result:  # type: ignore
                chunk_type = chunk.get("type", "")
                
                if chunk_type == "content":
                    delta = chunk.get("delta", "")
                    accumulated_content += delta
                    yield {"type": "content_delta", "delta": delta, "step": step + 1}
                elif chunk_type == "tool_call":
                    delta = chunk.get("delta", {})
                    tc_id = delta.get("id", "")
                    if tc_id and tc_id not in accumulated_tool_calls:
                        accumulated_tool_calls[tc_id] = {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                        tool_call_ids.append(tc_id)
                    
                    if tc_id in accumulated_tool_calls:
                        tc = accumulated_tool_calls[tc_id]
                        fn_delta = delta.get("function", {})
                        if "name" in fn_delta:
                            tc["function"]["name"] = fn_delta["name"]
                        if "arguments" in fn_delta:
                            tc["function"]["arguments"] += fn_delta["arguments"]
                    
                    yield {"type": "tool_call_delta", "delta": delta, "step": step + 1}
                elif chunk_type == "done":
                    break
                elif chunk_type == "error":
                    err = chunk.get("error", "unknown_error")
                    terminated_reason = "error"
                    yield {"type": "agent_error", "error": err, "step": step + 1}
                    break
            
            # Build complete message from accumulated state
            assistant_msg["content"] = accumulated_content
            if accumulated_tool_calls:
                assistant_msg["tool_calls"] = [accumulated_tool_calls[tc_id] for tc_id in tool_call_ids]
        else:
            # Non-streaming agent - result is a Message (or Promise<Message>)
            if hasattr(result, "__await__"):
                # It's a Promise, await it
                assistant_msg = await result  # type: ignore
            else:
                # It's a regular Message
                assistant_msg = result  # type: ignore

        if not is_assistant_message(assistant_msg):
            terminated_reason = "error"
            err = "agent_returned_non_assistant_message"
            messages.append({"role": "assistant", "content": "", "_error": err})
            yield {"type": "agent_error", "error": err, "step": step + 1}
            break

        # Extract token usage if available
        usage = _extract_usage_from_message(assistant_msg)
        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]
        total_tokens += usage["total_tokens"]

        # Normalize basic fields
        assistant_msg.setdefault("content", "")
        messages.append(assistant_msg)
        steps += 1

        yield {"type": "agent_response_complete", "message": assistant_msg, "step": step + 1}

        tcs = assistant_msg.get("tool_calls") or []
        if tcs:
            # Execute tool calls and append tool messages
            for tc in tcs:
                tool_calls += 1
                tc_id = str(tc.get("id", ""))
                fn = tc.get("function") or {}
                name = str(fn.get("name", ""))
                arg_str = fn.get("arguments", "{}")

                yield {"type": "tool_call_start", "tool_call": tc, "step": step + 1}

                # Enforce allowlist
                if tool_allow is not None and name not in tool_allow:
                    result = {"error": f"forbidden_tool:{name}"}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    yield {"type": "tool_call_result", "tool_call_id": tc_id, "result": result, "step": step + 1}
                    continue

                # Unknown tool
                if name not in tools:
                    result = {"error": f"unknown_tool:{name}"}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    yield {"type": "tool_call_result", "tool_call_id": tc_id, "result": result, "step": step + 1}
                    continue

                # Parse arguments
                try:
                    args = json.loads(arg_str) if isinstance(arg_str, str) else (arg_str or {})
                    if not isinstance(args, dict):
                        args = {"_non_dict_args": args}
                except Exception:
                    result = {"error": "invalid_tool_arguments_json"}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    yield {"type": "tool_call_result", "tool_call_id": tc_id, "result": result, "step": step + 1}
                    continue

                # Invoke tool
                try:
                    res = tools[name](args)
                except Exception as e:
                    res = {"error": repr(e)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": json.dumps(res, ensure_ascii=False),
                })
                yield {"type": "tool_call_result", "tool_call_id": tc_id, "result": res, "step": step + 1}

            # Continue loop (not done)
            yield {"type": "step_complete", "step": step + 1, "messages": messages.copy()}
            continue

        # No tool calls => final answer => done
        terminated_reason = "final_answer"
        yield {"type": "step_complete", "step": step + 1, "messages": messages.copy()}
        break
    else:
        terminated_reason = "max_steps"

    duration = now() - t0
    info = EpisodeInfo(
        task_id=task_id,
        trial=int((task_meta or {}).get("_trial", 0)),
        seed=seed,
        steps=steps,
        tool_calls=tool_calls,
        duration_s=round(duration, 4),
        terminated_reason=terminated_reason,
        error=err,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        cost_usd=0.0,
    )
    yield {"type": "episode_complete", "transcript": messages, "info": info}


def run_episode(
    initial_messages: List[Message],
    agent: AgentFn,
    tools: Dict[str, ToolFn],
    tools_allowed: Optional[List[str]] = None,
    limits: Optional[JSON] = None,
    task_meta: Optional[JSON] = None,
    seed: int = 0,
) -> Tuple[List[Message], EpisodeInfo]:
    """
    Orchestrates the agent harness in the canonical agent-environment loop:
      - Loop calls agent harness with messages and context
      - Agent harness returns assistant message
      - If assistant has tool_calls: environment executes them and appends tool messages, loop continues
      - Else: assistant is final; done
    
    This is the core execution pattern that orchestrates the agent harness. The evaluation harness builds on top of this.
    
    Note: This function only works with synchronous agents (AgentFn). For streaming support, use `stream_episode()` instead.
    """
    task_id = str((task_meta or {}).get("id", "unknown"))
    max_steps = int((limits or {}).get("max_steps", 30))
    time_limit_s = float((limits or {}).get("time_limit_s", 120))

    messages = list(initial_messages)
    tool_allow = set(tools_allowed) if tools_allowed is not None else None

    t0 = now()
    steps = 0
    tool_calls = 0
    terminated_reason = "error"
    err: Optional[str] = None
    
    # Token tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    # Provide schema to agent via context (optional)
    tools_schema = build_tools_schema(tools, tools_allowed)

    for _ in range(max_steps):
        if now() - t0 > time_limit_s:
            terminated_reason = "time_limit"
            break

        context = {
            "tools_schema": tools_schema,
            "task": task_meta or {},
            "seed": seed,
            "limits": limits or {},
        }

        assistant_msg = agent(messages, context)
        if not is_assistant_message(assistant_msg):
            terminated_reason = "error"
            err = "agent_returned_non_assistant_message"
            messages.append({"role": "assistant", "content": "", "_error": err})
            break

        # Extract token usage if available
        usage = _extract_usage_from_message(assistant_msg)
        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]
        total_tokens += usage["total_tokens"]

        # Normalize basic fields
        assistant_msg.setdefault("content", "")
        messages.append(assistant_msg)
        steps += 1

        tcs = assistant_msg.get("tool_calls") or []
        if tcs:
            # Execute tool calls and append tool messages
            for tc in tcs:
                tool_calls += 1
                tc_id = str(tc.get("id", ""))
                fn = tc.get("function") or {}
                name = str(fn.get("name", ""))
                arg_str = fn.get("arguments", "{}")

                # Enforce allowlist
                if tool_allow is not None and name not in tool_allow:
                    result = {"error": f"forbidden_tool:{name}"}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    continue

                # Unknown tool
                if name not in tools:
                    result = {"error": f"unknown_tool:{name}"}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    continue

                # Parse arguments
                try:
                    args = json.loads(arg_str) if isinstance(arg_str, str) else (arg_str or {})
                    if not isinstance(args, dict):
                        args = {"_non_dict_args": args}
                except Exception:
                    result = {"error": "invalid_tool_arguments_json"}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    continue

                # Invoke tool
                try:
                    res = tools[name](args)
                except Exception as e:
                    res = {"error": repr(e)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": json.dumps(res, ensure_ascii=False),
                })

            # Continue loop (not done)
            continue

        # No tool calls => final answer => done
        terminated_reason = "final_answer"
        break
    else:
        terminated_reason = "max_steps"

    duration = now() - t0
    info = EpisodeInfo(
        task_id=task_id,
        trial=int((task_meta or {}).get("_trial", 0)),
        seed=seed,
        steps=steps,
        tool_calls=tool_calls,
        duration_s=round(duration, 4),
        terminated_reason=terminated_reason,
        error=err,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        cost_usd=0.0,
    )
    return messages, info


async def stream_episode(
    initial_messages: List[Message],
    agent: Union[AgentFn, StreamingAgentFn],
    tools: Dict[str, ToolFn],
    tools_allowed: Optional[List[str]] = None,
    limits: Optional[JSON] = None,
    task_meta: Optional[JSON] = None,
    seed: int = 0,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Streaming version of `run_episode()` that yields events and chunks in real-time.
    
    Supports both streaming agents (StreamingAgentFn) and non-streaming agents (AgentFn).
    
    Yields:
    - Chunk events (from streaming agents):
      - `{type: "content_delta", delta: str, step: int}` - content chunk
      - `{type: "tool_call_delta", delta: {...}, step: int}` - tool call chunk
    - Control events:
      - `{type: "step_start", step: int}`
      - `{type: "agent_response_complete", message: Message, step: int}`
      - `{type: "tool_call_start", tool_call: dict, step: int}`
      - `{type: "tool_call_result", tool_call_id: str, result: any, step: int}`
      - `{type: "step_complete", step: int, messages: List[Message]}`
      - `{type: "episode_complete", transcript: List[Message], info: EpisodeInfo}`
    """
    async for event in _run_episode_stream(
        initial_messages, agent, tools, tools_allowed, limits, task_meta, seed
    ):
        yield event
