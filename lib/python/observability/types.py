"""Observability and evaluations data types."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class Dataset:
    """Collection of eval items."""
    id: str
    version: str  # Semantic version or hash
    items: List['EvalItem']
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EvalItem:
    """Single test case input."""
    id: str
    input: str  # User message
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSpec:
    """Evaluation specification (criteria, graders, agent config)."""
    id: str
    dataset_id: str
    dataset_version: str
    agent_id: str
    agent_version: str  # Commit SHA or semantic version
    agent_config_hash: str  # Hash of model, temperature, top_p, etc.
    criteria: List['Criterion']
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Criterion:
    """Single evaluation criterion."""
    id: str
    name: str
    description: str
    grader: 'GraderConfig'
    weight: float = 1.0


@dataclass
class GraderConfig:
    """Configuration for a grader (LLM judge, regex, etc.)."""
    type: str  # "llm_judge", "regex", "exact_match", "custom"
    model: Optional[str] = None  # For LLM judges
    prompt_template: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRun:
    """A single evaluation run (batch execution)."""
    id: str
    spec_id: str
    status: str  # "running", "completed", "failed"
    started_at: datetime
    completed_at: Optional[datetime] = None
    trace_id: str = ""  # Root trace ID for this run


@dataclass
class EvalCaseResult:
    """Result for a single eval case."""
    run_id: str
    case_id: str
    item_id: str
    item_index: int
    trace_id: str  # Trace ID for this case
    status: str  # "completed", "failed", "timeout"
    agent_output: str
    criterion_results: List['CriterionResult']
    started_at: datetime
    completed_at: datetime


@dataclass
class CriterionResult:
    """Result for a single criterion."""
    criterion_id: str
    score: float  # 0.0-1.0
    passed: bool
    reason: Optional[str] = None
    judge_model: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class EvalAggregation:
    """Aggregated results for an eval run."""
    run_id: str
    total_cases: int
    passed_cases: int
    total_score: float  # Weighted average
    criterion_scores: Dict[str, float]  # Per-criterion averages
    created_at: datetime = field(default_factory=datetime.now)
