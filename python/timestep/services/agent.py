"""Custom agent execution loop without OpenAI Agents SDK."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from timestep.utils.constants import (
    DEFAULT_MAX_ITERATIONS,
    EVENT_CONTENT_DELTA,
    EVENT_ERROR,
    EVENT_MESSAGE,
    EVENT_TOOL_CALL,
    EVENT_TOOL_ERROR,
    EVENT_TOOL_RESULT,
    ROLE_ASSISTANT,
    ROLE_USER,
)
from timestep.services.environment import Environment
from timestep.utils.exceptions import AgentExecutionError
from timestep.stores.session import Session
from timestep.utils.types import ChatMessage
from timestep.utils.helpers import (
    convert_messages_to_openai_format,
    create_tool_message,
    extract_tool_name_from_call,
    parse_tool_call_arguments,
)

logger = logging.getLogger(__name__)


class Agent:
    """Agent decision maker - handles OpenAI API calls."""
    
    def __init__(
        self,
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        api_key: Optional[str] = None
    ):
        """Initialize agent decision maker.
        
        Args:
            model: OpenAI model name
            tools: Default list of OpenAI tool definitions. Can be overridden in act().
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self.tools = tools or []
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def act(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[ChatMessage]:
        """Get agent action (LLM response) given messages and tools.
        
        Args:
            messages: Conversation messages
            tools: Optional list of OpenAI tool definitions. If None, uses self.tools.
            
        Returns:
            AsyncIterator of message chunks (incremental content updates, then final complete message)
        """
        # Use provided tools or fall back to default tools
        tools_to_use = tools if tools is not None else self.tools
        
        # Convert messages to OpenAI format
        openai_messages = convert_messages_to_openai_format(messages)
        
        # Call OpenAI API (always streaming)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=tools_to_use if tools_to_use else None,
            stream=True
        )
        
        # Return async iterator for streaming
        # Yields incremental content updates, then final complete message
        async def stream_messages():
            assistant_message: ChatMessage = {"role": ROLE_ASSISTANT, "content": "", "tool_calls": []}
            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content = delta.content
                        assistant_message["content"] += content
                        # Yield incremental content
                        yield {"content": content}
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call.index is not None:
                                while len(assistant_message["tool_calls"]) <= tool_call.index:
                                    assistant_message["tool_calls"].append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                tc = assistant_message["tool_calls"][tool_call.index]
                                if tool_call.id:
                                    tc["id"] = tool_call.id
                                if tool_call.function:
                                    if tool_call.function.name:
                                        tc["function"]["name"] = tool_call.function.name
                                    if tool_call.function.arguments:
                                        tc["function"]["arguments"] += tool_call.function.arguments
            # Yield final complete message
            yield assistant_message
        
        return stream_messages()
    
    async def invoke(
        self,
        agent_config: Dict[str, Any],
        session: Session,
        messages: List[ChatMessage],
        event_queue: Any,  # A2A EventQueue
    ) -> str:
        """Run full execution loop and stream events to EventQueue.
        
        Args:
            agent_config: Agent configuration dictionary
            session: Session for conversation persistence
            messages: Initial messages
            event_queue: A2A EventQueue for streaming events
            
        Returns:
            Final message text
        """
        from a2a.utils import new_agent_text_message
        
        # Create environment
        api_key = None
        if hasattr(self.client, '_api_key'):
            api_key = self.client._api_key
        environment = Environment(agent_config, session, api_key=api_key)
        
        # Update agent tools from environment
        self.tools = environment.get_tools()
        
        # Reset environment with initial messages
        # If messages is empty, check if session has existing messages
        # If so, don't reset (continuing conversation)
        existing_messages = await session.get_items()
        if messages or not existing_messages:
            await environment.reset(initial_messages=messages)
        
        # Create execution loop
        loop = ExecutionLoop(self, environment)
        
        # Run loop and stream events to EventQueue
        final_message = ""
        async for event in loop.run():
            event_type = event.get("type", "")
            if event_type == EVENT_CONTENT_DELTA:
                # Stream content delta to EventQueue
                content = event.get("content", "")
                await event_queue.enqueue_event(new_agent_text_message(content))
            elif event_type == EVENT_MESSAGE:
                # Final message
                final_message = event.get("content", "")
                # Don't enqueue again if we've already streamed the content
                # The final message is just for return value
            elif event_type == EVENT_ERROR:
                # Enqueue error
                error_msg = event.get("error", "")
                await event_queue.enqueue_event(new_agent_text_message(f"Error: {error_msg}"))
        
        return final_message


class ExecutionLoop:
    """Orchestrates agent-environment interaction."""
    
    def __init__(
        self,
        agent_decision_maker: Agent,
        environment: Environment,
        max_iterations: int = DEFAULT_MAX_ITERATIONS
    ):
        """Initialize execution loop.
        
        Args:
            agent_decision_maker: Agent decision maker instance
            environment: Environment instance
            max_iterations: Maximum number of iterations
        """
        self.agent_decision_maker = agent_decision_maker
        self.environment = environment
        self.max_iterations = max_iterations
    
    async def run(self) -> AsyncIterator[Dict[str, Any]]:
        """Run the agent-environment loop.
        
        Yields:
            Events during execution (message, tool_call, etc.)
        """
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1
            
            try:
                # Get observation from environment
                messages = await self.environment.get_observation()
                
                # Get action from agent (always streaming)
                tools = self.environment.get_tools()
                action = await self.agent_decision_maker.act(
                    messages=messages,
                    tools=tools,
                )
                
                # Handle streaming response
                assistant_message: ChatMessage = {"role": ROLE_ASSISTANT, "content": "", "tool_calls": []}
                async for chunk in action:
                    # Chunk can be either incremental content update or final message
                    if "content" in chunk and "role" not in chunk:
                        # Incremental content update
                        content = chunk.get("content", "")
                        assistant_message["content"] += content
                        yield {"type": EVENT_CONTENT_DELTA, "content": content}
                    else:
                        # Final complete message
                        assistant_message.update(chunk)
                
                # Step environment with complete message
                step_result = await self.environment.step(assistant_message)
                
                # Emit tool call events
                if assistant_message.get("tool_calls"):
                    for tool_call in assistant_message["tool_calls"]:
                        tool_name = extract_tool_name_from_call(tool_call)
                        tool_args = parse_tool_call_arguments(tool_call)
                        yield {"type": EVENT_TOOL_CALL, "tool": tool_name, "args": tool_args}
                    
                    # Emit tool result events
                    for tool_result in step_result["tool_results"]:
                        if "error" in tool_result:
                            yield {"type": EVENT_TOOL_ERROR, "tool": tool_result["tool"], "error": tool_result["error"]}
                        else:
                            yield {"type": EVENT_TOOL_RESULT, "tool": tool_result["tool"], "result": tool_result["result"]}
                else:
                    # Final message, we're done
                    yield {"type": EVENT_MESSAGE, "content": assistant_message.get("content", "")}
                    break
                        
            except AgentExecutionError as e:
                logger.error("Agent execution error: %s", e)
                yield {"type": EVENT_ERROR, "error": str(e)}
                break
            except Exception as e:
                logger.exception("Unexpected error in execution loop")
                yield {"type": EVENT_ERROR, "error": f"Unexpected error: {str(e)}"}
                break



