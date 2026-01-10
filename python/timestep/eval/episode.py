"""Episode runner - the core agent-environment loop."""

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agent import AgentFn
from .tools import ToolFn, build_tools_schema
from ..utils.messages import is_assistant_message
from ..utils.io import now

JSON = Dict[str, Any]
Message = Dict[str, Any]


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


def run_episode(
    initial_messages: List[Message],
    agent: AgentFn,
    tools: Dict[str, ToolFn],
    tools_allowed: Optional[List[str]],
    limits: JSON,
    task_meta: JSON,
    seed: int,
) -> Tuple[List[Message], EpisodeInfo]:
    """
    Runs the canonical loop:
      - agent returns assistant message
      - if assistant has tool_calls: env executes them and appends tool messages
      - else: assistant is final; done
    """
    task_id = str(task_meta.get("id", "unknown"))
    max_steps = int((limits or {}).get("max_steps", 30))
    time_limit_s = float((limits or {}).get("time_limit_s", 120))

    messages = list(initial_messages)
    tool_allow = set(tools_allowed) if tools_allowed is not None else None

    t0 = now()
    steps = 0
    tool_calls = 0
    terminated_reason = "error"
    err: Optional[str] = None

    # Provide schema to agent via context (optional)
    tools_schema = build_tools_schema(tools, tools_allowed)

    for _ in range(max_steps):
        if now() - t0 > time_limit_s:
            terminated_reason = "time_limit"
            break

        context = {
            "tools_schema": tools_schema,
            "task": task_meta,
            "seed": seed,
            "limits": limits,
        }

        assistant_msg = agent(messages, context)
        if not is_assistant_message(assistant_msg):
            terminated_reason = "error"
            err = "agent_returned_non_assistant_message"
            # Append an error assistant message for observability
            messages.append({"role": "assistant", "content": "", "_error": err})
            break

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
        trial=int(task_meta.get("_trial", 0)),
        seed=seed,
        steps=steps,
        tool_calls=tool_calls,
        duration_s=round(duration, 4),
        terminated_reason=terminated_reason,
        error=err,
    )
    return messages, info
