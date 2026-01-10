"""Built-in graders for evaluation."""

import json
import re
from typing import Any, Callable, Dict, List, Optional

from ..core.episode import EpisodeInfo
from ..core.tools import ToolCallRecord
from ..core.types import JSON, Message
from ..utils.messages import last_assistant_content
from ..utils.io import clamp01


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


# Code-based graders

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


class TranscriptContains(Grader):
    name = "TranscriptContains"

    def __init__(self, substring: Optional[str] = None, from_expected_key: str = "transcript_contains"):
        self.substring = substring
        self.from_expected_key = from_expected_key

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        sub = self.substring or (task.get("expected", {}) or {}).get(self.from_expected_key)
        if not sub:
            return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}
        # Check all messages, not just final
        transcript_text = " ".join(str(m.get("content", "")) for m in messages)
        ok = str(sub) in transcript_text
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"substring": sub}}


class TranscriptRegex(Grader):
    name = "TranscriptRegex"

    def __init__(self, pattern: Optional[str] = None, from_expected_key: str = "transcript_regex"):
        self.pattern = pattern
        self.from_expected_key = from_expected_key

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        pat = self.pattern or (task.get("expected", {}) or {}).get(self.from_expected_key)
        if not pat:
            return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}
        # Check all messages, not just final
        transcript_text = " ".join(str(m.get("content", "")) for m in messages)
        ok = re.search(pat, transcript_text, flags=re.MULTILINE) is not None
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"pattern": pat}}


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


class MinToolCalls(Grader):
    name = "MinToolCalls"

    def __init__(self, min_calls: int = 0, from_limits_key: str = "min_tool_calls"):
        self.min_calls = min_calls
        self.from_limits_key = from_limits_key

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        lim = (task.get("limits", {}) or {}).get(self.from_limits_key)
        min_calls = int(lim) if lim is not None else self.min_calls
        calls = len(tool_index)
        ok = calls >= min_calls
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"calls": calls, "min_calls": min_calls}}


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


class ToolCallOrder(Grader):
    name = "ToolCallOrder"

    def __init__(self, expected_sequence: Optional[List[str]] = None, from_expected_key: str = "tool_call_order"):
        self.expected_sequence = expected_sequence
        self.from_expected_key = from_expected_key

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        expected = self.expected_sequence or (task.get("expected", {}) or {}).get(self.from_expected_key)
        if not expected:
            return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}
        actual = [r.name for r in tool_index]
        # Check if expected sequence appears in actual sequence (allowing extra calls)
        expected_idx = 0
        for tool_name in actual:
            if expected_idx < len(expected) and tool_name == expected[expected_idx]:
                expected_idx += 1
        ok = expected_idx == len(expected)
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {"expected": expected, "actual": actual}}


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


# Outcome verification grader

class OutcomeVerifier(Grader):
    name = "OutcomeVerifier"

    def __init__(self, verifier_fn: Optional[Callable[[List[Message], List[ToolCallRecord], JSON], bool]] = None):
        """
        Outcome verifier that checks environment state, not just final message.
        
        Args:
            verifier_fn: Function that takes (transcript, tool_index, task) and returns bool
                        If None, uses task.expected.outcome_verifier
        """
        self.verifier_fn = verifier_fn

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        verifier = self.verifier_fn
        if verifier is None:
            # Try to get from task.expected
            verifier_data = (task.get("expected", {}) or {}).get("outcome_verifier")
            if verifier_data is None:
                return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}
            # If it's a string, try to import it
            if isinstance(verifier_data, str):
                # For now, require verifier_fn to be passed directly
                return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True, "note": "outcome_verifier must be passed as function"}}
            # Otherwise assume it's a callable (though JSON can't serialize functions)
            verifier = verifier_data
        
        try:
            ok = verifier(messages, tool_index, task)
        except Exception as e:
            return {"name": self.name, "passed": False, "score": 0.0, "details": {"error": "verifier_exception", "exception": repr(e)}}
        
        return {"name": self.name, "passed": ok, "score": 1.0 if ok else 0.0, "details": {}}


# LLM-as-judge grader

class LLMJudge(Grader):
    name = "LLMJudge"

    def __init__(
        self,
        rubric: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        grade_transcript: bool = False,
        from_expected_key: str = "llm_judge_rubric",
    ):
        """
        LLM-as-judge grader that uses OpenAI to grade based on a rubric.
        
        Args:
            rubric: Grading criteria/rubric for the LLM judge
            model: OpenAI model to use for judging
            temperature: Temperature for the judge model
            grade_transcript: If True, grades full transcript; if False, grades only final message
            from_expected_key: Key in task.expected to get rubric from
        """
        self.rubric = rubric
        self.model = model
        self.temperature = temperature
        self.grade_transcript = grade_transcript
        self.from_expected_key = from_expected_key

    def grade(self, messages: List[Message], tool_index: List[ToolCallRecord], task: JSON, info: EpisodeInfo) -> JSON:
        rubric = self.rubric or (task.get("expected", {}) or {}).get(self.from_expected_key)
        if not rubric:
            return {"name": self.name, "passed": True, "score": 1.0, "details": {"skipped": True}}

        try:
            from openai import OpenAI
            client = OpenAI()
        except ImportError:
            return {"name": self.name, "passed": False, "score": 0.0, "details": {"error": "openai_not_installed"}}

        # Prepare content to grade
        if self.grade_transcript:
            # Grade full transcript
            transcript_text = "\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                for m in messages
                if m.get("role") in ("user", "assistant")
            )
            content_to_grade = f"Transcript:\n{transcript_text}"
        else:
            # Grade only final assistant message
            content_to_grade = f"Final assistant message: {last_assistant_content(messages)}"

        # Create judge prompt
        judge_prompt = f"""You are evaluating an AI agent's performance. Here is the rubric:

{rubric}

Here is what to evaluate:

{content_to_grade}

Respond with a JSON object with:
- "passed": boolean (true if the agent meets the criteria)
- "score": float between 0.0 and 1.0
- "reasoning": string explaining your judgment
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            passed = bool(result.get("passed", False))
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
            return {
                "name": self.name,
                "passed": passed,
                "score": clamp01(score),
                "details": {"reasoning": reasoning, "model": self.model},
            }
        except Exception as e:
            return {"name": self.name, "passed": False, "score": 0.0, "details": {"error": "llm_judge_failed", "exception": repr(e)}}


BUILTIN_GRADERS = {
    # Code-based
    "FinalRegex": FinalRegex,
    "FinalContains": FinalContains,
    "FinalJSON": FinalJSON,
    "TranscriptContains": TranscriptContains,
    "TranscriptRegex": TranscriptRegex,
    "ForbiddenTools": ForbiddenTools,
    "MaxToolCalls": MaxToolCalls,
    "MinToolCalls": MinToolCalls,
    "ToolCallSequence": ToolCallSequence,
    "ToolCallOrder": ToolCallOrder,
    "ToolResultJSON": ToolResultJSON,
    # Outcome verification
    "OutcomeVerifier": OutcomeVerifier,
    # LLM-as-judge
    "LLMJudge": LLMJudge,
}


def parse_grader_spec(spec: str) -> Grader:
    """
    Simple CLI grader spec format:
      - "FinalRegex" (uses task.expected.final_regex)
      - "FinalRegex:^133$" (explicit regex)
      - "FinalContains:Mike"
      - "MaxToolCalls:5"
      - "MinToolCalls:2"
      - "ToolCallSequence:calc"
      - "ToolCallOrder:calc,echo" (expected sequence)
      - "ToolResultJSON:calc,value" (tool=calc, required_keys=[value])
      - "LLMJudge" (uses task.expected.llm_judge_rubric)
      - "LLMJudge:gpt-4o-mini:0.0" (model:temperature)
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
    if name == "TranscriptContains":
        return TranscriptContains(substring=arg)
    if name == "TranscriptRegex":
        return TranscriptRegex(pattern=arg)
    if name == "MaxToolCalls":
        return MaxToolCalls(max_calls=int(arg))
    if name == "MinToolCalls":
        return MinToolCalls(min_calls=int(arg))
    if name == "ToolCallSequence":
        return ToolCallSequence(must_call=arg)
    if name == "ToolCallOrder":
        sequence = [s.strip() for s in arg.split(",") if s.strip()]
        return ToolCallOrder(expected_sequence=sequence)
    if name == "ToolResultJSON":
        parts = [p.strip() for p in arg.split(",") if p.strip()]
        tool = parts[0] if parts else None
        keys = parts[1:] if len(parts) > 1 else None
        return ToolResultJSON(tool_name=tool, required_keys=keys)
    if name == "FinalJSON":
        keys = [k.strip() for k in arg.split(",") if k.strip()]
        return FinalJSON(required_keys=keys)
    if name == "LLMJudge":
        # Format: model:temperature or just model
        parts = arg.split(":")
        model = parts[0] if parts else "gpt-4o-mini"
        temp = float(parts[1]) if len(parts) > 1 else 0.0
        return LLMJudge(model=model, temperature=temp)

    # Default: no-arg init
    return cls()  # type: ignore


def aggregate_grades(grades: List[JSON]) -> JSON:
    """Aggregate multiple grade results into a single result."""
    if not grades:
        return {"passed": False, "score": 0.0}
    mean = sum(float(g.get("score", 0.0)) for g in grades) / len(grades)
    passed = all(bool(g.get("passed")) for g in grades)
    return {"passed": passed, "score": clamp01(mean)}
