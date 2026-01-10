"""Agent function interface and adapters."""

import json
import shlex
import subprocess
import time
from typing import Any, Callable, Dict, List

from ..utils.messages import is_assistant_message

JSON = Dict[str, Any]
Message = Dict[str, Any]
AgentFn = Callable[[List[Message], JSON], Message]  # (messages, context) -> assistant message


def agent_builtin_echo(messages: List[Message], context: JSON) -> Message:
    """Builtin agent that finishes immediately by echoing the last user message."""
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = str(m.get("content", "") or "")
            break
    return {"role": "assistant", "content": f"Echo: {last_user}"}


def agent_cmd_factory(agent_cmd: str, timeout_s: int = 120) -> AgentFn:
    """
    Creates an agent function that shells out to `agent_cmd`.
    
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
