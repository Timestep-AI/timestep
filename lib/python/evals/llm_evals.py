"""LLM-based evaluations."""

from typing import Any, Dict, Optional
from openai import OpenAI

from timestep.evals.base import Eval, EvalResult


class LLMEvaluator(Eval):
    """LLM-based evaluator that uses an LLM to evaluate agent responses."""
    
    def __init__(
        self,
        criteria: str,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.criteria = criteria
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    async def evaluate(
        self,
        agent_response: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """Evaluate agent response using LLM.
        
        Args:
            agent_response: The agent's response to evaluate
            expected_output: Optional expected output for comparison
            context: Optional context information
            
        Returns:
            EvalResult with pass/fail, score, and details
        """
        # Build evaluation prompt
        prompt = f"""Evaluate the following agent response based on these criteria:

Criteria: {self.criteria}

Agent Response:
{agent_response}
"""
        
        if expected_output:
            prompt += f"\nExpected Output:\n{expected_output}\n"
        
        if context:
            prompt += f"\nContext:\n{context}\n"
        
        prompt += """
Please provide:
1. A pass/fail judgment (PASS or FAIL)
2. A score from 0.0 to 1.0
3. Detailed reasoning

Format your response as:
PASS/FAIL
Score: X.X
Reasoning: [detailed explanation]
"""
        
        # Call LLM for evaluation
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of AI agent responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        
        evaluation_text = response.choices[0].message.content or ""
        
        # Parse response
        passed = "PASS" in evaluation_text.upper()
        score = 0.0
        details = evaluation_text
        
        # Extract score if present
        if "Score:" in evaluation_text:
            try:
                score_line = [line for line in evaluation_text.split("\n") if "Score:" in line][0]
                score_str = score_line.split("Score:")[1].strip().split()[0]
                score = float(score_str)
            except (ValueError, IndexError):
                score = 1.0 if passed else 0.0
        
        return EvalResult(
            passed=passed,
            score=score,
            details=details,
        )
