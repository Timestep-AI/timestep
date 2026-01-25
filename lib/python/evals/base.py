"""Base eval classes and interfaces."""

from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Result of an evaluation."""
    passed: bool
    score: float  # 0.0 to 1.0
    details: str
    metadata: Optional[Dict[str, Any]] = None


class Eval:
    """Base class for evaluations."""
    
    async def evaluate(self, *args, **kwargs) -> EvalResult:
        """Run the evaluation.
        
        Returns:
            EvalResult with pass/fail, score, and details
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
