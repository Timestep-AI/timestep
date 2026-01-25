"""Trace-based evaluations - verify handoff flows and tool calls using OpenTelemetry traces."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from timestep.evals.base import Eval, EvalResult


def load_trace(trace_file: str) -> List[Dict[str, Any]]:
    """Load OpenTelemetry trace from JSONL file.
    
    Args:
        trace_file: Path to trace file (JSONL format)
        
    Returns:
        List of span objects
    """
    spans = []
    with open(trace_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    span = json.loads(line)
                    spans.append(span)
                except json.JSONDecodeError:
                    continue
    return spans


def find_span(
    spans: List[Dict[str, Any]],
    span_name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Find a span by name and optional attributes.
    
    Args:
        spans: List of span objects
        span_name: Name of the span to find
        attributes: Optional attributes to match
        
    Returns:
        Span object if found, None otherwise
    """
    for span in spans:
        if span.get("name") == span_name:
            if attributes:
                span_attrs = span.get("attributes", {})
                if all(span_attrs.get(k) == v for k, v in attributes.items()):
                    return span
            else:
                return span
    return None


def find_spans(
    spans: List[Dict[str, Any]],
    span_name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Find all spans matching name and optional attributes.
    
    Args:
        spans: List of span objects
        span_name: Name of the span to find
        attributes: Optional attributes to match
        
    Returns:
        List of matching span objects
    """
    results = []
    for span in spans:
        if span.get("name") == span_name:
            if attributes:
                span_attrs = span.get("attributes", {})
                if all(span_attrs.get(k) == v for k, v in attributes.items()):
                    results.append(span)
            else:
                results.append(span)
    return results


class TraceEval(Eval):
    """Base class for trace-based evaluations."""
    
    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.spans = load_trace(trace_file)
    
    def _find_span(
        self,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a span by name and optional attributes."""
        return find_span(self.spans, span_name, attributes)
    
    def _find_spans(
        self,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Find all spans matching name and optional attributes."""
        return find_spans(self.spans, span_name, attributes)
    
    async def evaluate(self) -> EvalResult:
        """Run the evaluation."""
        raise NotImplementedError("Subclasses must implement evaluate()")


async def verify_handoff(
    trace_file: str,
    from_agent: str,
    to_agent: str,
) -> bool:
    """Verify handoff from one agent to another.
    
    Args:
        trace_file: Path to trace file
        from_agent: Source agent ID
        to_agent: Target agent ID
        
    Returns:
        True if handoff verified, False otherwise
    """
    spans = load_trace(trace_file)
    
    # Find handoff tool call
    handoff_span = find_span(spans, "execute_tool", {"tool_name": "handoff"})
    if not handoff_span:
        return False
    
    # Check tool arguments
    tool_arguments = handoff_span.get("attributes", {}).get("tool_arguments", "{}")
    if isinstance(tool_arguments, str):
        try:
            tool_arguments = json.loads(tool_arguments)
        except json.JSONDecodeError:
            # If not valid JSON, check if it's a string representation
            if to_agent in tool_arguments:
                return True
            return False
    
    # Verify agent_uri contains target agent
    agent_uri = tool_arguments.get("agent_uri", "")
    if to_agent not in str(agent_uri):
        return False
    
    return True


async def verify_tool_call(
    trace_file: str,
    agent_id: str,
    tool_name: str,
    **kwargs,
) -> bool:
    """Verify specific tool call with arguments.
    
    Args:
        trace_file: Path to trace file
        agent_id: Agent ID (for verification)
        tool_name: Name of tool to verify
        **kwargs: Tool arguments to verify
        
    Returns:
        True if tool call verified, False otherwise
    """
    spans = load_trace(trace_file)
    
    # Find tool execution span
    tool_span = find_span(spans, "execute_tool", {"tool_name": tool_name})
    if not tool_span:
        return False
    
    # Check agent_id if provided
    span_attrs = tool_span.get("attributes", {})
    if agent_id and span_attrs.get("agent_id") != agent_id:
        return False
    
    # Check tool arguments
    if kwargs:
        tool_arguments = span_attrs.get("tool_arguments", "{}")
        if isinstance(tool_arguments, str):
            try:
                tool_arguments = json.loads(tool_arguments)
            except json.JSONDecodeError:
                # If not valid JSON, try string matching
                for key, value in kwargs.items():
                    if f'"{key}": "{value}"' not in tool_arguments and f'"{key}":{value}' not in tool_arguments:
                        return False
                return True
        
        for key, value in kwargs.items():
            if tool_arguments.get(key) != value:
                return False
    
    return True
