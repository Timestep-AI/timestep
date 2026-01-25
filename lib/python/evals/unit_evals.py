"""Unit test style evaluations with assertions."""

from typing import Any, Dict, List, Optional

from timestep.evals.base import Eval, EvalResult


class UnitEvaluator(Eval):
    """Unit test style evaluator with assertions."""
    
    def __init__(self):
        self.assertions: List[Dict[str, Any]] = []
    
    def assert_contains(self, text: str, substring: str) -> bool:
        """Assert that text contains substring.
        
        Args:
            text: Text to check
            substring: Substring to find
            
        Returns:
            True if assertion passes
        """
        result = substring in text
        self.assertions.append({
            "type": "contains",
            "text": text,
            "substring": substring,
            "passed": result,
        })
        return result
    
    def assert_not_contains(self, text: str, substring: str) -> bool:
        """Assert that text does not contain substring.
        
        Args:
            text: Text to check
            substring: Substring to avoid
            
        Returns:
            True if assertion passes
        """
        result = substring not in text
        self.assertions.append({
            "type": "not_contains",
            "text": text,
            "substring": substring,
            "passed": result,
        })
        return result
    
    def assert_equals(self, actual: Any, expected: Any) -> bool:
        """Assert that actual equals expected.
        
        Args:
            actual: Actual value
            expected: Expected value
            
        Returns:
            True if assertion passes
        """
        result = actual == expected
        self.assertions.append({
            "type": "equals",
            "actual": actual,
            "expected": expected,
            "passed": result,
        })
        return result
    
    def assert_tool_called(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_name: str,
        **kwargs,
    ) -> bool:
        """Assert that a specific tool was called with given arguments.
        
        Args:
            tool_calls: List of tool calls
            tool_name: Name of tool to check
            **kwargs: Tool arguments to verify
            
        Returns:
            True if assertion passes
        """
        for tool_call in tool_calls:
            if tool_call.get("name") == tool_name:
                if kwargs:
                    arguments = tool_call.get("arguments", {})
                    if all(arguments.get(k) == v for k, v in kwargs.items()):
                        self.assertions.append({
                            "type": "tool_called",
                            "tool_name": tool_name,
                            "arguments": kwargs,
                            "passed": True,
                        })
                        return True
                else:
                    self.assertions.append({
                        "type": "tool_called",
                        "tool_name": tool_name,
                        "passed": True,
                    })
                    return True
        
        self.assertions.append({
            "type": "tool_called",
            "tool_name": tool_name,
            "arguments": kwargs,
            "passed": False,
        })
        return False
    
    async def evaluate(self) -> EvalResult:
        """Run all assertions and return result.
        
        Returns:
            EvalResult with pass/fail, score, and details
        """
        passed_count = sum(1 for a in self.assertions if a.get("passed", False))
        total_count = len(self.assertions)
        
        passed = passed_count == total_count
        score = passed_count / total_count if total_count > 0 else 0.0
        
        details = f"{passed_count}/{total_count} assertions passed"
        if not passed:
            failed = [a for a in self.assertions if not a.get("passed", False)]
            details += f"\nFailed assertions: {failed}"
        
        return EvalResult(
            passed=passed,
            score=score,
            details=details,
            metadata={"assertions": self.assertions},
        )
