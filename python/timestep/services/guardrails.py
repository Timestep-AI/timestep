"""Guardrail system for tool execution."""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from timestep.utils.types import ChatMessage


# ============================================================================
# Guardrail System
# ============================================================================

class GuardrailError(Exception):
    """Base exception for guardrail failures."""
    pass


class GuardrailInterrupt(GuardrailError):
    """Raised when guardrail requires interruption (e.g., human input) to proceed."""
    def __init__(self, prompt: str, tool_name: str, args: Dict[str, Any]):
        self.prompt = prompt
        self.tool_name = tool_name
        self.args = args
        super().__init__(f"Guardrail interrupt for {tool_name}: {prompt}")


@dataclass
class GuardrailResult:
    """Result from a guardrail check."""
    proceed: bool  # Whether to continue execution
    modified_args: Optional[Dict[str, Any]] = None  # Modified input args (pre-guardrail)
    modified_result: Optional[Dict[str, Any]] = None  # Modified output result (post-guardrail)
    reason: Optional[str] = None  # Reason for blocking/modifying
    
    @classmethod
    def block(cls, reason: str) -> "GuardrailResult":
        """Block execution."""
        return cls(proceed=False, reason=reason)
    
    @classmethod
    def proceed(cls) -> "GuardrailResult":
        """Allow execution to proceed unchanged."""
        return cls(proceed=True)
    
    @classmethod
    def modify_args(cls, new_args: Dict[str, Any]) -> "GuardrailResult":
        """Modify input arguments before execution."""
        return cls(proceed=True, modified_args=new_args)
    
    @classmethod
    def modify_result(cls, new_result: Dict[str, Any]) -> "GuardrailResult":
        """Modify output result after execution."""
        return cls(proceed=True, modified_result=new_result)


# Guardrail function signature (async function)
Guardrail = Callable[
    [str, Dict[str, Any], Literal["pre", "post"], Optional[Dict[str, Any]]],
    Awaitable[GuardrailResult]
]


class InputGuardrail:
    """
    Input guardrail for tool execution.
    
    Applies before tool execution to validate/modify inputs.
    """
    def __init__(self, handler: Callable[[str, Dict[str, Any]], Awaitable[GuardrailResult]]):
        """
        Initialize input guardrail.
        
        Args:
            handler: Async function (tool_name, args) -> GuardrailResult
        """
        self.handler = handler
    
    async def check(self, tool_name: str, args: Dict[str, Any]) -> GuardrailResult:
        """Check input guardrail."""
        return await self.handler(tool_name, args)


class OutputGuardrail:
    """
    Output guardrail for tool execution.
    
    Applies after tool execution to validate/modify outputs.
    """
    def __init__(self, handler: Callable[[str, Dict[str, Any], Dict[str, Any]], Awaitable[GuardrailResult]]):
        """
        Initialize output guardrail.
        
        Args:
            handler: Async function (tool_name, args, result) -> GuardrailResult
        """
        self.handler = handler
    
    async def check(self, tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> GuardrailResult:
        """Check output guardrail."""
        return await self.handler(tool_name, args, result)


class _GuardrailWrapper:
    """
    Internal helper that wraps tool handlers with guardrails.
    
    Applies pre and post guardrails to tool handlers before execution.
    """
    
    def __init__(
        self,
        pre_guardrails: Optional[List[Guardrail]] = None,
        post_guardrails: Optional[List[Guardrail]] = None,
    ):
        """
        Initialize the guardrail wrapper.
        
        Args:
            pre_guardrails: List of guardrails to apply before tool execution
            post_guardrails: List of guardrails to apply after tool execution
        """
        self.pre_guardrails = pre_guardrails or []
        self.post_guardrails = post_guardrails or []
    
    def __call__(
        self,
        tool_handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        tool_name: str,
    ) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
        """
        Wrap a tool handler with guardrails.
        
        Args:
            tool_handler: The tool handler function to wrap
            tool_name: Name of the tool (needed for guardrail context)
        
        Returns:
            Wrapped tool handler function with guardrails applied
        """
        async def guarded_handler(args: Dict[str, Any]) -> Dict[str, Any]:
            return await with_guardrails(
                tool_handler,
                tool_name=tool_name,
                args=args,
                pre_guardrails=self.pre_guardrails,
                post_guardrails=self.post_guardrails,
            )
        return guarded_handler


async def request_approval(prompt: str) -> bool:
    """Request approval from user. Blocks until answered."""
    event_loop = asyncio.get_event_loop()
    while True:
        answer = await event_loop.run_in_executor(None, input, prompt)
        answer = answer.strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


async def with_guardrails(
    tool_handler: Callable[[Dict[str, Any]], Any],
    tool_name: str,
    args: Dict[str, Any],
    pre_guardrails: Optional[List[Guardrail]] = None,
    post_guardrails: Optional[List[Guardrail]] = None,
) -> Dict[str, Any]:
    """
    Execute a tool handler with optional pre and post guardrails.
    
    Pre-guardrails can:
    - Block execution (return GuardrailResult.block())
    - Modify input args (return GuardrailResult.modify_args())
    - Raise GuardrailInterrupt to require human input
    
    Post-guardrails can:
    - Modify output result (return GuardrailResult.modify_result())
    - Block (though tool already executed)
    
    Args:
        tool_handler: The tool handler function to wrap
        tool_name: Name of the tool
        args: Input arguments for the tool
        pre_guardrails: List of guardrail functions to run before execution
        post_guardrails: List of guardrail functions to run after execution
    
    Returns:
        Tool result (possibly modified by post-guardrails)
    
    Raises:
        GuardrailInterrupt: If a guardrail requires interruption
    """
    # Execute pre-guardrails
    if pre_guardrails:
        for guardrail in pre_guardrails:
            result = await guardrail(tool_name, args, "pre")
            if not result.proceed:
                raise GuardrailError(f"Guardrail blocked execution: {result.reason}")
            if result.modified_args is not None:
                args = result.modified_args
    
    # Execute tool with (possibly modified) args
    tool_result = await tool_handler(args)
    
    # Execute post-guardrails
    if post_guardrails:
        for guardrail in post_guardrails:
            result = await guardrail(tool_name, args, "post", tool_result)
            if not result.proceed:
                # Tool already executed, but we can still block the result
                raise GuardrailError(f"Guardrail blocked result: {result.reason}")
            if result.modified_result is not None:
                tool_result = result.modified_result
    
    return tool_result




