"""Environment manages state, tools, guardrails, and handoffs."""

from __future__ import annotations

import inspect
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from timestep.utils.constants import EVENT_ERROR, EVENT_MESSAGE, ROLE_SYSTEM, ROLE_TOOL, ROLE_USER
from timestep.utils.exceptions import AgentConfigError, HandoffError, ToolExecutionError
from timestep.services.guardrails import GuardrailError, GuardrailInterrupt, InputGuardrail, OutputGuardrail, request_approval, with_guardrails
from timestep.stores.session import FileSession, Session
from timestep.utils.types import AgentConfig, ChatMessage, Tool
from timestep.utils.helpers import create_tool_message, parse_tool_call_arguments

logger = logging.getLogger(__name__)


def convert_handoffs_to_tools(agent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert agent handoffs to OpenAI tool format.
    
    Args:
        agent: Agent with handoffs configuration
        
    Returns:
        List of tool definitions for handoffs
    """
    tools = []
    for handoff_agent in agent.get("handoffs", []):
        tool_name = f"transfer_to_{handoff_agent['name'].lower().replace(' ', '_')}"
        tool_description = f"Handoff to the {handoff_agent['name']} agent"
        
        tool_def = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to send to the target agent"
                        },
                        "conversation_id": {
                            "type": "string",
                            "description": "Optional conversation ID for the target agent's session"
                        }
                    },
                    "required": ["message"]
                }
            }
        }
        tools.append(tool_def)
    return tools


def convert_tool_to_openai_format(tool: Tool) -> Dict[str, Any]:
    """Convert a tool function to OpenAI tool format.
    
    Args:
        tool: Tool function (callable)
        
    Returns:
        OpenAI tool format dict
    """
    # Try to extract tool metadata from function attributes
    tool_name = getattr(tool, "__name__", "tool")
    tool_description = getattr(tool, "__doc__", "") or getattr(tool, "description", "")
    
    # Try to get parameters schema
    params_schema = getattr(tool, "parameters", None) or getattr(tool, "params_json_schema", None)
    if params_schema is None:
        # Default empty schema
        params_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": params_schema
        }
    }


class Environment:
    """Environment manages state, tools, guardrails, and handoffs."""
    
    def __init__(
        self,
        agent: Dict[str, Any],  # Agent config
        session: Session,
        api_key: Optional[str] = None
    ):
        """Initialize environment.
        
        Args:
            agent: Agent configuration
            session: Session for conversation history
            api_key: OpenAI API key for handoffs (optional)
            
        Raises:
            AgentConfigError: If agent configuration is invalid
        """
        # Validate agent config
        Environment.validate_agent_config(agent)
        
        self.agent = agent
        self.session = session
        self.api_key = api_key
        
        # Register tools
        self._tool_registry: Dict[str, Tool] = {}
        self._handoff_registry: Dict[str, Dict[str, Any]] = {}
        self._openai_tools = self._register_tools()
    
    @staticmethod
    def validate_agent_config(agent: Dict[str, Any]) -> None:
        """Validate agent configuration.
        
        Args:
            agent: Agent configuration dictionary
            
        Raises:
            AgentConfigError: If configuration is invalid
        """
        if not isinstance(agent, dict):
            raise AgentConfigError("Agent configuration must be a dictionary")
        
        # Check required fields
        required_fields = ["name", "model", "instructions"]
        for field in required_fields:
            if field not in agent:
                raise AgentConfigError(f"Agent configuration missing required field: '{field}'")
            if not isinstance(agent[field], str) or not agent[field].strip():
                raise AgentConfigError(f"Agent configuration field '{field}' must be a non-empty string")
        
        # Validate optional fields if present
        if "tools" in agent and not isinstance(agent["tools"], list):
            raise AgentConfigError("Agent configuration field 'tools' must be a list")
        
        if "handoffs" in agent:
            if not isinstance(agent["handoffs"], list):
                raise AgentConfigError("Agent configuration field 'handoffs' must be a list")
            # Recursively validate handoff agents
            for i, handoff_agent in enumerate(agent["handoffs"]):
                try:
                    Environment.validate_agent_config(handoff_agent)
                except AgentConfigError as e:
                    raise AgentConfigError(f"Invalid handoff agent at index {i}: {e}") from e
        
        if "guardrails" in agent and not isinstance(agent["guardrails"], list):
            raise AgentConfigError("Agent configuration field 'guardrails' must be a list")
    
    def _validate_agent_config(self, agent: Dict[str, Any]) -> None:
        """Validate agent configuration (instance method wrapper).
        
        Args:
            agent: Agent configuration dictionary
            
        Raises:
            AgentConfigError: If configuration is invalid
        """
        Environment.validate_agent_config(agent)
    
    def _register_tools(self) -> List[Dict[str, Any]]:
        """Register agent tools and convert to OpenAI format.
        
        Returns:
            List of OpenAI tool definitions
        """
        openai_tools = []
        
        # Register regular tools
        for tool in self.agent.get("tools", []):
            tool_name = getattr(tool, "__name__", f"tool_{len(self._tool_registry)}")
            self._tool_registry[tool_name] = tool
            openai_tool = convert_tool_to_openai_format(tool)
            openai_tools.append(openai_tool)
        
        # Register handoff tools
        handoff_tools = convert_handoffs_to_tools(self.agent)
        for handoff_tool in handoff_tools:
            tool_name = handoff_tool["function"]["name"]
            # Find the corresponding handoff agent
            for handoff_agent in self.agent.get("handoffs", []):
                expected_name = f"transfer_to_{handoff_agent['name'].lower().replace(' ', '_')}"
                if tool_name == expected_name:
                    self._handoff_registry[tool_name] = handoff_agent
                    break
            openai_tools.append(handoff_tool)
        
        return openai_tools
    
    async def reset(self, initial_messages: List[ChatMessage] | None = None) -> None:
        """Reset the environment by clearing the session and optionally adding initial messages.
        
        Args:
            initial_messages: Optional initial messages to add after clearing
        """
        await self.session.clear_session()
        if initial_messages:
            await self.session.add_items(initial_messages)
    
    async def get_observation(self) -> List[ChatMessage]:
        """Get current observation (messages from session).
        
        Returns:
            List of conversation messages
        """
        return await self.session.get_items()
    
    async def step(self, action: ChatMessage) -> Dict[str, Any]:
        """Execute action (assistant message) and update state.
        
        Args:
            action: Assistant message with optional tool calls
            
        Returns:
            Dict with step results (tool_results, done flag, etc.)
        """
        # Add assistant message to session
        await self.session.add_items([action])
        
        # Check for tool calls
        tool_calls = action.get("tool_calls", [])
        if not tool_calls:
            # No tool calls, we're done
            return {"done": True, "tool_results": []}
        
        # Execute tools
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = parse_tool_call_arguments(tool_call)
            
            tool_result = await self._execute_tool(tool_name, tool_args)
            
            # Add tool result to session
            tool_message = create_tool_message(tool_result, tool_call["id"])
            await self.session.add_items([tool_message])
            
            tool_results.append({
                "tool": tool_name,
                "result": tool_result,
                "tool_call_id": tool_call["id"]
            })
        
        return {"done": False, "tool_results": tool_results}
    
    async def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool with guardrails.
        
        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments
            
        Returns:
            Tool execution result
        """
        # Check if it's a handoff tool
        if tool_name in self._handoff_registry:
            return await self._execute_handoff(tool_name, args)
        
        # Regular tool
        if tool_name not in self._tool_registry:
            raise ToolExecutionError(tool_name, f"Tool '{tool_name}' not found in registry")
        
        tool = self._tool_registry[tool_name]
        
        # Get guardrails for this agent
        pre_guardrails = []
        post_guardrails = []
        for guardrail in self.agent.get("guardrails", []):
            if isinstance(guardrail, InputGuardrail):
                # Create wrapper that matches Guardrail signature
                def make_pre_check(g):
                    async def pre_check(tool_name: str, args: Dict[str, Any], phase: str, result: Optional[Dict[str, Any]] = None):
                        return await g.check(tool_name, args)
                    return pre_check
                pre_guardrails.append(make_pre_check(guardrail))
            elif isinstance(guardrail, OutputGuardrail):
                # Create wrapper that matches Guardrail signature
                def make_post_check(g):
                    async def post_check(tool_name: str, args: Dict[str, Any], phase: str, result: Optional[Dict[str, Any]] = None):
                        return await g.check(tool_name, args, result or {})
                    return post_check
                post_guardrails.append(make_post_check(guardrail))
        
        # Wrap tool handler
        async def tool_handler(tool_args: Dict[str, Any]) -> Dict[str, Any]:
            # Handle both sync and async tools
            if inspect.iscoroutinefunction(tool):
                result = await tool(tool_args)
            else:
                result = tool(tool_args)
            # Ensure result is a dict
            if not isinstance(result, dict):
                return {"result": result}
            return result
        
        try:
            result = await with_guardrails(
                tool_handler,
                tool_name=tool_name,
                args=args,
                pre_guardrails=pre_guardrails if pre_guardrails else None,
                post_guardrails=post_guardrails if post_guardrails else None,
            )
            return result
        except GuardrailInterrupt as e:
            # Request human approval
            prompt = f"{e.prompt}\nApprove? (y/n): "
            approved = await request_approval(prompt)
            if approved:
                # Re-execute with approval
                result = await tool_handler(args)
                if not isinstance(result, dict):
                    return {"result": result}
                return result
            else:
                raise GuardrailError(f"Tool {tool_name} execution blocked by user")
    
    async def _execute_handoff(
        self,
        tool_name: str,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a handoff to another agent.
        
        Args:
            tool_name: Name of the handoff tool
            args: Handoff arguments (message, conversation_id)
            
        Returns:
            Result from target agent
        """
        target_agent = self._handoff_registry[tool_name]
        message = args.get("message", "")
        conversation_id = args.get("conversation_id")
        
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Create session for target agent
        target_session = FileSession(
            agent_name=target_agent["name"],
            conversation_id=conversation_id,
            agent_instructions=target_agent.get("instructions", "")
        )
        
        # Add user message to target session
        await target_session.add_items([{"role": ROLE_USER, "content": message}])
        
        try:
            # Execute target agent using nested loop
            # Import here to avoid circular dependency
            from .agent import Agent, ExecutionLoop
            
            # Create environment first to get tools
            target_environment = Environment(target_agent, target_session, api_key=self.api_key)
            target_agent_instance = Agent(
                model=target_agent["model"],
                tools=target_environment.get_tools(),
                api_key=self.api_key
            )
            target_loop = ExecutionLoop(target_agent_instance, target_environment)
            
            result_messages = []
            async for event in target_loop.run():
                if event.get("type") == EVENT_MESSAGE:
                    result_messages.append(event.get("content", ""))
                elif event.get("type") == EVENT_ERROR:
                    raise HandoffError(
                        target_agent.get("name", "unknown"),
                        event.get("error", "Unknown error"),
                    )
            
            # Return result
            result_text = "\n".join(result_messages) if result_messages else ""
            return {"result": result_text, "conversation_id": conversation_id}
        except HandoffError:
            raise
        except Exception as e:
            logger.exception("Unexpected error during handoff to %s", target_agent.get("name", "unknown"))
            raise HandoffError(
                target_agent.get("name", "unknown"),
                f"Unexpected error: {str(e)}",
                original_error=e
            )
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get OpenAI tool definitions.
        
        Returns:
            List of OpenAI tool definitions
        """
        return self._openai_tools

