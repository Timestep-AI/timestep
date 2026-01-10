"""Built-in graders for evaluation."""

import json
import re
from typing import Any, Dict, List, Optional

from .episode import EpisodeInfo
from .tools import ToolCallRecord
from ..utils.messages import Message, last_assistant_content
from ..utils.io import clamp01

JSON = Dict[str, Any]


class Grader:
    """
    A grader consumes:
      - messages: full transcript
      - tool_index: list[ToolCallRecord]
      - task: task JSON (for expected values, allowlists, etc.)
      - info: EpisodeInfo

    And returns a result dict:
      {"name": str, "passed": bool, "score": float 0..1, "details": {...}}
    """
    name: str = "Grader"

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        raise NotImplementedError


class FinalRegex(Grader):
    name = "FinalRegex"

    def __init__(self, pattern: Optional[str] = None, from_expected_key: str = "final_regex"):
        self.pattern = pattern
        self.from_expected_key = from_expected_key

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        pat = self.pattern or (task.get("expected", {}) or {}).get(self.from_expected_key)
        if not pat:
            return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}
        text = last_assistant_content(messages)
        ok = re.search(pat, text, flags=re.MULTILINE) is not None
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"pattern": pat}}


class FinalContains(Grader):
    name = "FinalContains"

    def __init__(self, substring: Optional[str] = None, from_expected_key: str = "final_contains"):
        self.substring = substring
        self.from_expected_key = from_expected_key

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        sub = self.substring or (task.get("expected", {}) or {}).get(self.from_expected_key)
        if not sub:
            return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}
        text = last_assistant_content(messages)
        ok = str(sub) in text
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"substring": sub}}


class FinalJSON(Grader):
    name = "FinalJSON"

    def __init__(self, required_keys: Optional[List[str]] = None, from_expected_key: str = "final_json_required_keys"):
        self.required_keys = required_keys
        self.from_expected_key = from_expected_key

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        keys = self.required_keys or (task.get("expected", {}) or {}).get(self.from_expected_key)
        if not keys:
            return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}
        text = last_assistant_content(messages)
        try:
            obj = json.loads(text)
        except Exception as e:
            return {"name": self.name, "passed": False, "score": 0.0, "details": {"error": "invalid_json", "exception": repr(e)}}
        missing = [k for k in keys if k not in obj]
        ok = not missing
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"missing": missing}}


class ForbiddenTools(Grader):
    name = "ForbiddenTools"

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        allowed = task.get("tools_allowed")
        if allowed is None:
            return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}
        allowed_set = set(allowed)
        used = [r.name for r in tool_index]
        forbidden = [n for n in used if n not in allowed_set]
        ok = not forbidden
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"forbidden": forbidden, "used": used}}


class MaxToolCalls(Grader):
    name = "MaxToolCalls"

    def __init__(self, max_calls: int = 999999, from_limits_key: str = "max_tool_calls"):
        self.max_calls = max_calls
        self.from_limits_key = from_limits_key

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        lim = (task.get("limits", {}) or {}).get(self.from_limits_key)
        max_calls = int(lim) if lim is not None else self.max_calls
        calls = len(tool_index)
        ok = calls <= max_calls
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"calls": calls, "max_calls": max_calls}}


class ToolCallSequence(Grader):
    name = "ToolCallSequence"

    def __init__(self, must_call: Optional[str] = None, from_expected_key: str = "must_call_tool"):
        self.must_call = must_call
        self.from_expected_key = from_expected_key

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        must = self.must_call or (task.get("expected", {}) or {}).get(self.from_expected_key)
        if not must:
            return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}
        used = [r.name for r in tool_index]
        ok = str(must) in used
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"must_call": must, "used": used}}


class ToolResultJSON(Grader):
    name = "ToolResultJSON"

    def __init__(self, tool_name: Optional[str] = None, required_keys: Optional[List[str]] = None):
        self.tool_name = tool_name
        self.required_keys = required_keys

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        # Configure from task.expected if not provided:
        exp = task.get("expected", {}) or {}
        tool = self.tool_name or exp.get("tool_result_name")
        keys = self.required_keys or exp.get("tool_result_required_keys")

        if not tool or not keys:
            return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}

        # Find last record for that tool
        recs = [r for r in tool_index if r.name == tool]
        if not recs:
            return {"name": self.name, "passed": False, "score": 0.0, "details": {"error": "tool_not_called", "tool": tool}}

        last = recs[-1]
        if isinstance(last.result, str):
            try:
                obj = json.loads(last.result)
            except Exception:
                return {"name": self.name, "passed": False, "score": 0.0, "details": {"error": "tool_result_not_json"}}
        else:
            obj = last.result

        if not isinstance(obj, dict):
            return {"name": self.name, "passed": False, "score": 0.0, "details": {"error": "tool_result_not_object"}}

        missing = [k for k in keys if k not in obj]
        ok = not missing
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"tool": tool, "missing": missing}}


BUILTIN_GRADERS = {
    "FinalRegex": FinalRegex,
    "FinalContains": FinalContains,
    "FinalJSON": FinalJSON,
    "ForbiddenTools": ForbiddenTools,
    "MaxToolCalls": MaxToolCalls,
    "ToolCallSequence": ToolCallSequence,
    "ToolResultJSON": ToolResultJSON,
}


def parse_grader_spec(spec: str) -> Grader:
    """
    Simple CLI grader spec format:
      - "FinalRegex" (uses task.expected.final_regex)
      - "FinalRegex:^133$" (explicit regex)
      - "FinalContains:Mike"
      - "MaxToolCalls:5"
      - "ToolCallSequence:calc"
      - "ToolResultJSON:calc,value" (tool=calc, required_keys=[value])
    """
    if ":" not in spec:
        cls = BUILTIN_GRADERS.get(spec)
        if not cls:
            raise SystemExit(f"Unknown grader '{spec}'. Available: {', '.join(BUILTIN_GRADERS)}")
        return cls()  # type: ignore

    name, arg = spec.split(":", 1)
    cls = BUILTIN_GRADERS.get(name)
    if not cls:
        raise SystemExit(f"Unknown grader '{name}'. Available: {', '.join(BUILTIN_GRADERS)}")

    if name == "FinalRegex":
        return FinalRegex(pattern=arg)
    if name == "FinalContains":
        return FinalContains(substring=arg)
    if name == "MaxToolCalls":
        return MaxToolCalls(max_calls=int(arg))
    if name == "ToolCallSequence":
        return ToolCallSequence(must_call=arg)
    if name == "ToolResultJSON":
        parts = [p.strip() for p in arg.split(",") if p.strip()]
        tool = parts[0] if parts else None
        keys = parts[1:] if len(parts) > 1 else None
        return ToolResultJSON(tool_name=tool, required_keys=keys)
    if name == "FinalJSON":
        keys = [k.strip() for k in arg.split(",") if k.strip()]
        return FinalJSON(required_keys=keys)

    # Default: no-arg init
    return cls()  # type: ignore


def aggregate_grades(grades: List[JSON]) -> JSON:
    """Aggregate multiple grade results into a single result."""
    if not grades:
        return {"passed": False, "score": 0.0}
    mean = sum(float(g.get("score", 0.0)) for g in grades) / len(grades)
    passed = all(bool(g.get("passed")) for g in grades)
    return {"passed": passed, "score": clamp01(mean)}
