"""Unified observability and evaluations module.

This module provides trace-based evaluation capabilities using OpenTelemetry
Test and GenAI semantic conventions. Evaluations are based on spans (for prompts)
and traces (for full conversations/episodes/rollouts).
"""

from timestep.observability.types import (
    Dataset,
    EvalItem,
    EvalSpec,
    Criterion,
    GraderConfig,
    EvalRun,
    EvalCaseResult,
    CriterionResult,
    EvalAggregation,
)

__all__ = [
    "Dataset",
    "EvalItem",
    "EvalSpec",
    "Criterion",
    "GraderConfig",
    "EvalRun",
    "EvalCaseResult",
    "CriterionResult",
    "EvalAggregation",
]
