"""Tool execution and indexing."""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .types import JSON, Message, ToolFn
from ..utils.messages import is_assistant_message, is_tool_message


def tool_calc(args: JSON) -> Any:
    """
    Demo tool: calculates arithmetic expression in args["expr"].
    WARNING: For production, do NOT use eval. Use a safe expression parser.
    """
    expr = str(args.get("expr", ""))
    # Extremely restricted eval (still not perfect for production)
    val = eval(expr, {"__builtins__": {}}, {})
    return {"expr": expr, "value": val}


def tool_echo(args: JSON) -> Any:
    """Demo tool: echoes back arguments."""
    return {"echo": args}


DEFAULT_TOOLS: Dict[str, ToolFn] = {
    "calc": tool_calc,
    "echo": tool_echo,
}


def build_tools_schema(tools: Dict[str, ToolFn], allowed: Optional[List[str]]) -> List[JSON]:
    """
    Builds a minimal OpenAI-style tools schema list.
    
    NOTE: This is a minimal schema for interoperability; you can extend it.
    """
    names = sorted(tools.keys())
    if allowed is not None:
        allowed_set = set(allowed)
        names = [n for n in names if n in allowed_set]

    # Minimal function schema; arguments left open (free-form JSON) by default.
    schema = []
    for name in names:
        schema.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool '{name}'",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            }
        })
    return schema


@dataclass
class ToolCallRecord:
    """Record of a tool call paired with its result."""
    tool_call_id: str
    name: str
    arguments_raw: Any
    arguments: JSON
    result_raw: str
    result: Any
    error: Optional[str] = None


def index_tool_calls(messages: List[Message]) -> List[ToolCallRecord]:
    """
    Pairs assistant tool calls with subsequent tool messages by tool_call_id.
    
    Returns a list of ToolCallRecord in chronological order of the tool calls.
    """
    # Map tool_call_id -> (name, arguments_raw)
    calls: Dict[str, Tuple[str, Any]] = {}
    ordered_ids: List[str] = []

    for m in messages:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                tc_id = str(tc.get("id", ""))
                fn = tc.get("function") or {}
                name = str(fn.get("name", ""))
                args_raw = fn.get("arguments", "{}")
                if tc_id:
                    calls[tc_id] = (name, args_raw)
                    ordered_ids.append(tc_id)

    # Pair with tool results
    results: Dict[str, Tuple[str, Any]] = {}  # id -> (raw_content, parsed)
    for m in messages:
        if m.get("role") == "tool":
            tc_id = str(m.get("tool_call_id", ""))
            raw = str(m.get("content", "") or "")
            parsed: Any = None
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw
            if tc_id:
                results[tc_id] = (raw, parsed)

    out: List[ToolCallRecord] = []
    for tc_id in ordered_ids:
        name, args_raw = calls.get(tc_id, ("", "{}"))
        args_parsed: JSON = {}
        err: Optional[str] = None
        try:
            args_parsed = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            if not isinstance(args_parsed, dict):
                args_parsed = {"_non_dict_args": args_parsed}
        except Exception:
            err = "invalid_tool_arguments_json"
            args_parsed = {}

        res_raw, res_parsed = results.get(tc_id, ("", None))
        out.append(ToolCallRecord(
            tool_call_id=tc_id,
            name=name,
            arguments_raw=args_raw,
            arguments=args_parsed,
            result_raw=res_raw,
            result=res_parsed,
            error=err,
        ))
    return out
