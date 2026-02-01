"""Loop class - Orchestrates agent-environment interaction loop until completion."""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

from a2a.client import ClientFactory
from a2a.client.client import ClientConfig
from a2a.types import (
    Message,
    Part,
    DataPart,
    Role,
)
from a2a.client.helpers import create_text_message_object
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
from mcp.types import TextContent
import mcp.types as mcp_types
from mcp.shared.context import RequestContext

from timestep.utils.message_helpers import (
    TOOL_CALLS_KEY,
    TOOL_RESULTS_KEY,
    convert_mcp_tool_to_openai,
)
from timestep.utils.event_helpers import (
    extract_event_data,
    extract_task_from_tuple,
)

# OpenTelemetry imports for trace context propagation in sampling callback
try:
    from opentelemetry import propagate, context
except ImportError:
    # If OpenTelemetry is not available, propagate and context will be None
    propagate = None
    context = None

# Optional httpx for creating an instrumented client in the sampling callback
# so A2A handoff requests carry the current trace context (Traceparent header)
httpx: Any = None
try:
    import httpx as _httpx_runtime
    httpx = _httpx_runtime
except ImportError:
    pass


class Loop:
    """Loop orchestrates agent-environment interaction until completion (full rollout/episode).
    
    This component provides a /v1/responses endpoint (OpenAI-compatible) that:
    1. Fetches system prompt and tools from Environment (MCP) before sending to Agent
    2. Handles tool execution via MCP
    3. Continues agent loop until final response (complete rollout/episode)
    4. Supports both streaming and non-streaming modes
    
    Loop is a pure client component - it communicates with Agent and Environment via protocols,
    not direct class references.
    """
    
    def __init__(
        self,
        agent_base_url: str,
        environment_uri: str,
        agent_id: str,
        model_name: str,
    ):
        """Initialize Loop.
        
        Args:
            agent_base_url: Base URL for the agent (e.g., "http://localhost:9999") - for A2A client
            environment_uri: MCP environment URI - for MCP client
            agent_id: Agent ID for fetching system prompt from Environment
            model_name: Model name for Responses API format
        """
        self.agent_base_url = agent_base_url
        self.environment_uri = environment_uri
        self.agent_id = agent_id
        self.model_name = model_name
        # Always create sampling callback (handoff tool is always in Environment)
        self.sampling_callback = self._create_sampling_callback()
        
        # Create FastAPI app for this component
        self.app = FastAPI()
        
        # Register the /v1/responses endpoint
        # FastAPI instrumentation will automatically create spans for this endpoint
        @self.app.post("/v1/responses")
        async def handle_responses(request: Request):
            """Handle /v1/responses endpoint - automatically executes tool calls."""
            body = await request.json()
            stream = body.get("stream", False)
            
            if stream:
                return StreamingResponse(
                    self._handle_responses_streaming(body),
                    media_type="text/event-stream"
                )
            else:
                return await self._handle_responses_non_streaming(body)
    
    @property
    def fastapi_app(self) -> FastAPI:
        """Get the FastAPI app with /v1/responses route."""
        return self.app
    
    async def _fetch_system_prompt_and_tools(self, instructions: Optional[str] = None) -> tuple[str, List[Dict[str, Any]]]:
        """Fetch system prompt and tools from Environment via MCP.
        
        Args:
            instructions: Optional instructions to pass to system prompt
            
        Returns:
            Tuple of (system_prompt, tools)
        """
        async with streamable_http_client(self.environment_uri) as (read, write, _):
            async with ClientSession(read, write) as mcp_session:
                await mcp_session.initialize()
                
                # Get system prompt
                prompt_args = {"agent_name": self.agent_id}
                if instructions:
                    prompt_args["instructions"] = instructions
                
                system_prompt_result = await mcp_session.get_prompt("system_prompt", prompt_args)
                if not system_prompt_result.messages or not system_prompt_result.messages[0].content.text:
                    raise ValueError(f"System prompt 'system_prompt' returned empty result from environment at {self.environment_uri}")
                system_prompt = system_prompt_result.messages[0].content.text
                
                # Get tools
                tools_result = await mcp_session.list_tools()
                tools = [
                    convert_mcp_tool_to_openai(tool)
                    for tool in tools_result.tools
                ]
                
                return system_prompt, tools
    
    def _get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Get attribute from object (no dict fallback - A2A SDK returns objects)."""
        return getattr(obj, attr, default)
    
    def _extract_state_value(self, status: Any) -> Optional[str]:
        """Extract state value from status object."""
        if not status:
            return None
        state = self._get_attr(status, 'state')
        if not state:
            return None
        return self._get_attr(state, 'value')
    
    def _extract_text_from_parts(self, parts: List[Any]) -> str:
        """Extract all text from message parts."""
        text = ""
        for part in parts:
            part_data = part.root if hasattr(part, 'root') else part
            if self._get_attr(part_data, 'kind') == 'text':
                part_text = self._get_attr(part_data, 'text', '')
                if part_text:
                    text += part_text
        return text
    
    def _extract_message_from_status(self, status: Any) -> Optional[Any]:
        """Extract message from status object."""
        if not status:
            return None
        return self._get_attr(status, 'message')
    
    def _extract_task_info(self, event: Any) -> tuple[Optional[str], Optional[str], Any]:
        """Extract task_id, context_id, and task object from event.
        
        Returns:
            Tuple of (task_id, context_id, task_object)
        """
        event_data = extract_event_data(event)
        task_obj = extract_task_from_tuple(event)
        
        if not task_obj:
            raise ValueError("Task object not found in event - extract_task_from_tuple returned None")
        
        task_id = self._get_attr(event_data, 'task_id') or self._get_attr(task_obj, 'id')
        context_id = self._get_attr(event_data, 'context_id') or self._get_attr(task_obj, 'context_id')
        
        return task_id, context_id, task_obj
    
    def _extract_tool_calls_from_status(self, status: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from a status object's message DataPart."""
        if not status:
            return None
        
        status_message = self._extract_message_from_status(status)
        if not status_message:
            return None
        
        parts = self._get_attr(status_message, 'parts', [])
        if not parts:
            return None
        
        # Look for tool calls in parts
        # Part wraps DataPart - access via part.root or part directly
        for part in parts:
            # Get the actual part data (Part.root points to DataPart/TextPart)
            part_data = self._get_attr(part, 'root', part)
            part_kind = self._get_attr(part_data, 'kind')
            
            if part_kind == 'data':
                # DataPart has a 'data' attribute that is a dict
                part_data_dict = self._get_attr(part_data, 'data')
                if part_data_dict:
                    # Handle both dict and object formats
                    if isinstance(part_data_dict, dict):
                        tool_calls = part_data_dict.get(TOOL_CALLS_KEY)
                    else:
                        tool_calls = self._get_attr(part_data_dict, TOOL_CALLS_KEY)
                    if tool_calls:
                        return tool_calls
        
        return None
    
    def _extract_tool_calls(self, task: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from task status message DataPart."""
        status = self._get_attr(task, 'status')
        return self._extract_tool_calls_from_status(status)
    
    def _extract_tool_output(self, result: Dict[str, Any]) -> str:
        """Extract text output from tool result dict.
        
        Handles:
        - {"result": "text"} -> "text"
        - {"error": "error message"} -> "error message"
        - Other dicts -> JSON stringified
        - Strings -> as-is
        """
        if isinstance(result, dict):
            if "result" in result:
                return str(result["result"])
            elif "error" in result:
                return str(result["error"])
            else:
                return json.dumps(result)
        return str(result)
    
    def _build_tool_result_message(
        self,
        tool_results: List[Dict[str, Any]],
        task_id: Optional[str],
        context_id: Optional[str],
    ) -> Any:
        """Build a user message carrying tool results via DataPart."""
        tool_result_msg = create_text_message_object(role=Role.user, content="")
        if task_id:
            tool_result_msg.task_id = task_id
        if context_id:
            tool_result_msg.context_id = context_id
        # DataPart tool_results maps to OpenAI tool messages in the A2A server.
        tool_result_msg.parts.append(Part(DataPart(data={TOOL_RESULTS_KEY: tool_results})))
        return tool_result_msg
    
    def _extract_final_message(self, task: Any) -> str:
        """Extract final message text from a completed task.
        
        Only extracts the final completed message, not incremental updates.
        Trusts status.message as the source of truth.
        """
        status = self._get_attr(task, 'status')
        if not status:
            raise ValueError("Task has no status")
        
        status_message = self._extract_message_from_status(status)
        if not status_message:
            raise ValueError("Status has no message")
        
        parts = self._get_attr(status_message, 'parts', [])
        message_text = self._extract_text_from_parts(parts)
        
        if not message_text.strip():
            raise ValueError("Status message has no text content")
        
        return message_text.strip()
    
    async def _handle_agent_handoff(
        self,
        agent_uri: str,
        message: str,
        *,
        httpx_client: Any = None,
    ) -> str:
        """Handle agent handoff by calling the A2A agent at agent_uri.

        Fetches system prompt and tools from the target agent's Environment (MCP)
        and includes them in the A2A message metadata before calling the agent.

        When httpx_client is provided (e.g. from the sampling callback), the A2A
        client uses it so outgoing HTTP requests carry the current trace context
        (Traceparent header), ensuring distributed tracing across the handoff.
        """
        # Fetch system prompt and tools from target agent's Environment
        target_env_uri = f"{agent_uri}/mcp"
        
        # Get agent name from agent card to use for system prompt
        if not httpx_client:
            raise ValueError("httpx_client required for handoff to fetch agent card")
        
        agent_card_url = f"{agent_uri}/.well-known/agent-card.json"
        card_resp = await httpx_client.get(agent_card_url, timeout=10.0)
        card_resp.raise_for_status()
        card_data = card_resp.json()
        agent_name = card_data.get("name")
        if not agent_name:
            raise ValueError(f"Agent card at {agent_card_url} has no 'name' field")
        
        # Fetch system prompt and tools from target agent's Environment
        async with streamable_http_client(target_env_uri) as (read, write, _):
            async with ClientSession(read, write) as mcp_session:
                await mcp_session.initialize()
                
                # Get system prompt
                prompt_args = {"agent_name": agent_name}
                system_prompt_result = await mcp_session.get_prompt("system_prompt", prompt_args)
                if not system_prompt_result.messages or not system_prompt_result.messages[0].content.text:
                    raise ValueError(f"System prompt 'system_prompt' returned empty result from environment at {target_env_uri}")
                system_prompt = system_prompt_result.messages[0].content.text
                
                # Get tools
                tools_result = await mcp_session.list_tools()
                tools = [
                    convert_mcp_tool_to_openai(tool)
                    for tool in tools_result.tools
                ]
        
        # Use ClientFactory with optional instrumented httpx client for trace propagation
        client_config = ClientConfig(httpx_client=httpx_client) if httpx_client else None
        a2a_client = await ClientFactory.connect(agent_uri, client_config=client_config)
        
        try:
            message_obj = create_text_message_object(role="user", content=message)
            
            # Include system prompt and tools in metadata (AgentExecutor will extract these)
            message_obj.metadata = {
                "system_prompt": system_prompt,
                "tools": tools,
            }
            
            # Process message using ClientFactory (non-streaming, polling mode)
            final_message = ""
            task_id = None
            context_id = None
            
            # Process message stream until completion
            while True:
                # Send message - Client.send_message returns an async generator
                # Client.send_message expects just the message object, not SendMessageRequest
                
                # Process message stream
                state_value = None
                tool_calls = None
                async for event in a2a_client.send_message(message_obj):
                    # Extract task info from event
                    extracted_task_id, extracted_context_id, task = self._extract_task_info(event)
                    if extracted_task_id:
                        task_id = extracted_task_id
                    if extracted_context_id:
                        context_id = extracted_context_id
                    
                    # Get event_data to check status from the event itself
                    event_data = extract_event_data(event)
                    event_status = self._get_attr(event_data, 'status')
                    if not event_status:
                        raise ValueError("Event has no status")
                    
                    # Extract state from event status
                    state_value = self._extract_state_value(event_status)
                    
                    if state_value == "completed":
                        final_message = self._extract_final_message(task)
                        break
                    
                    if state_value == "input-required":
                        # Tool calls MUST be in the event status when input-required is emitted
                        tool_calls = self._extract_tool_calls_from_status(event_status)
                        if not tool_calls:
                            raise ValueError("Handoff: input-required but no tool calls found in event status")
                        
                        # Execute tools via MCP
                        tool_results = []
                        for tc in tool_calls:
                            tool_name = tc.get("name", "")
                            tool_args = tc.get("arguments", {})
                            call_id = tc.get("call_id", "")
                            
                            # Construct MCP endpoint URI from agent_uri
                            target_env_uri = f"{agent_uri}/mcp"
                            
                            result = await self._execute_tool_via_mcp(tool_name, tool_args, target_env_uri)
                            # Extract text from result dict before passing to build_tool_result_message
                            output_text = self._extract_tool_output(result)
                            
                            tool_results.append({
                                "call_id": call_id,
                                "name": tool_name,
                                "output": output_text,
                            })
                        
                        # Build tool result message and send it
                        tool_result_msg = self._build_tool_result_message(tool_results, task_id, context_id)
                        # Include system prompt and tools in metadata for tool result message too
                        self._add_system_prompt_and_tools_to_metadata(tool_result_msg, system_prompt, tools)
                        message_obj = tool_result_msg
                        # Break from inner loop to continue outer loop with new message
                        break
                
                # If we completed, exit outer loop
                if state_value == "completed":
                    break
                
                # If we broke from the loop due to tool calls, continue outer loop
                if state_value == "input-required" and tool_calls:
                    continue
                
                # Otherwise, exit
                break
            
            if not final_message.strip():
                raise ValueError("Handoff completed but no final message content")
            return final_message.strip()
        except Exception as e:
            logger.error(f"Error during handoff: {str(e)}", exc_info=True)
            return f"Error during handoff: {str(e)}"
    
    def _create_sampling_callback(self):
        """Create MCP sampling callback for handoff."""
        async def sampling_callback(
            mcp_context: RequestContext["ClientSession", Any],
            params: mcp_types.CreateMessageRequestParams,
        ) -> mcp_types.CreateMessageResult | mcp_types.CreateMessageResultWithTools | mcp_types.ErrorData:
            """MCP sampling callback that handles sampling requests from the MCP server."""
            try:
                # Extract agent_uri from metadata (passed by handoff tool)
                metadata = params.metadata or {}
                agent_uri = metadata.get("agent_uri")
                if not agent_uri:
                    return mcp_types.ErrorData(
                        code=mcp_types.INVALID_PARAMS,
                        message="agent_uri is required for sampling",
                    )
                
                # Extract and attach trace context from MCP metadata if available
                # This ensures trace continuity across the async handoff boundary
                trace_context_token = None
                if propagate is not None and context is not None:
                    # Extract trace context from metadata (injected by handoff tool)
                    trace_carrier = {}
                    if "traceparent" in metadata:
                        trace_carrier["traceparent"] = metadata["traceparent"]
                    if "tracestate" in metadata:
                        trace_carrier["tracestate"] = metadata["tracestate"]
                    
                    if trace_carrier:
                        # Extract trace context from carrier and attach to current context
                        extracted_context = propagate.extract(trace_carrier)
                        trace_context_token = context.attach(extracted_context)
                
                try:
                    # Extract message text from params.messages (SamplingMessage objects with TextContent)
                    message_text = params.messages[0].content.text if params.messages else "Please help with this task."
                    
                    # Use an httpx client created in the current context so A2A requests carry
                    # Traceparent header (distributed tracing across the handoff boundary)
                    if httpx is not None:
                        async with httpx.AsyncClient() as http_client:
                            result_text = await self._handle_agent_handoff(
                                agent_uri=agent_uri,
                                message=message_text,
                                httpx_client=http_client,
                            )
                    else:
                        result_text = await self._handle_agent_handoff(
                            agent_uri=agent_uri,
                            message=message_text,
                        )
                    
                    return mcp_types.CreateMessageResult(
                        role="assistant",
                        content=mcp_types.TextContent(type="text", text=result_text),
                        model="a2a-agent"
                    )
                finally:
                    # Detach trace context token if we attached one
                    if trace_context_token is not None and context is not None:
                        context.detach(trace_context_token)
            except Exception as e:
                return mcp_types.ErrorData(
                    code=mcp_types.INTERNAL_ERROR,
                    message=f"Sampling error: {str(e)}",
                )
        return sampling_callback
    
    async def _execute_tool_via_mcp(self, tool_name: str, arguments: Dict[str, Any], environment_uri: str) -> Dict[str, Any]:
        """Execute a single tool via MCP."""
        try:
            async with streamable_http_client(environment_uri) as (read, write, _):
                async with ClientSession(read, write, sampling_callback=self.sampling_callback) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    
                    if result.content:
                        text_parts = [item.text for item in result.content if isinstance(item, TextContent)]
                        if text_parts:
                            result_text = " ".join(text_parts)
                            
                            # Special handling for handoff tool: extract response from nested structure
                            if tool_name == "handoff":
                                parsed = json.loads(result_text)
                                if not isinstance(parsed, dict) or "response" not in parsed:
                                    raise ValueError(f"Handoff tool returned invalid format: {result_text}")
                                result_text = parsed["response"]
                            
                            return {"result": result_text}
                        return {"result": None}
                    return {"result": None}
        except Exception as e:
            return {"error": str(e)}
    
    def _convert_responses_input_to_a2a(self, input_data: Any) -> Message:
        """Convert Responses API input to A2A message format.
        
        Input can be:
        - A string
        - A list of messages (for compatibility)
        - A list of items (from previous response output)
        """
        # Handle string input
        if isinstance(input_data, str):
            user_message = create_text_message_object(role=Role.user, content=input_data)
            return user_message
        
        # Handle list input (messages or items)
        if isinstance(input_data, list):
            # Find the last user message or text item
            user_content = ""
            tool_results = []
            
            for item in input_data:
                # Handle message format (for compatibility)
                if isinstance(item, dict):
                    role = item.get("role", "")
                    if role == "user":
                        content = item.get("content", "")
                        if isinstance(content, str):
                            user_content = content
                        elif isinstance(content, list):
                            text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                            user_content = " ".join(text_parts)
                    elif role == "tool":
                        tool_call_id = item.get("tool_call_id") or item.get("call_id")
                        tool_content = item.get("content", "")
                        if tool_call_id:
                            tool_results.append({
                                "call_id": tool_call_id,
                                "output": tool_content,
                            })
                    # Handle item format (from Responses API output)
                    elif item.get("type") == "message" and item.get("role") == "user":
                        content = item.get("content", [])
                        if isinstance(content, list):
                            text_parts = [part.get("text", "") for part in content if part.get("type") == "output_text"]
                            user_content = " ".join(text_parts)
                    elif item.get("type") == "function_call_output":
                        call_id = item.get("call_id")
                        output = item.get("output", "")
                        if call_id:
                            tool_results.append({
                                "call_id": call_id,
                                "output": output,
                            })
            
            if not user_content:
                raise ValueError("No user input found in input data")
            
            a2a_message = create_text_message_object(role=Role.user, content=user_content)
            if tool_results:
                a2a_message.parts.append(Part(DataPart(data={TOOL_RESULTS_KEY: tool_results})))
            return a2a_message
        
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _convert_a2a_response_to_responses(self, task: Any, content: str = "", status: Any = None) -> Dict[str, Any]:
        """Convert A2A task response to Responses API format."""
        # Use provided status or get from task
        if status is None:
            status = self._get_attr(task, 'status')
        if not status:
            raise ValueError("Task has no status")
        
        # Extract content from task status message if not provided
        if not content:
            status_message = self._extract_message_from_status(status)
            if status_message:
                parts = self._get_attr(status_message, 'parts', [])
                content = self._extract_text_from_parts(parts)
        
        # Build output items array
        output_items = []
        
        # Add tool calls if present
        tool_calls = self._extract_tool_calls(task)
        if tool_calls:
            for tc in tool_calls:
                output_items.append({
                    "id": f"fc_{uuid4().hex}",
                    "type": "function_call",
                    "call_id": tc.get("call_id", ""),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", {})
                })
        
        # Add message item
        if content or not tool_calls:
            output_items.append({
                "id": f"msg_{uuid4().hex}",
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": content
                }] if content else [],
                "role": "assistant"
            })
        
        response = {
            "id": f"resp_{uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "model": self.model_name,
            "output": output_items
        }
        
        return response
    
    def _add_system_prompt_and_tools_to_metadata(
        self,
        message: Message,
        system_prompt: str,
        tools: List[Dict[str, Any]],
        instructions: Optional[str] = None,
    ) -> None:
        """Add system prompt and tools to message metadata."""
        message.metadata = {
            "system_prompt": system_prompt,
            "tools": tools,
        }
        if instructions:
            message.metadata["instructions"] = instructions
    
    async def _handle_responses_non_streaming(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Handle non-streaming /v1/responses request."""
        # Responses API uses 'input' instead of 'messages'
        input_data = body.get("input")
        if input_data is None:
            raise HTTPException(status_code=400, detail="input is required")
        
        # Fetch system prompt and tools from Environment (MCP) once at the start
        instructions = body.get("instructions")
        system_prompt, tools = await self._fetch_system_prompt_and_tools(instructions=instructions)
        
        # Use ClientFactory (supports JSON-RPC by default)
        a2a_client = await ClientFactory.connect(self.agent_base_url)
        
        # Convert Responses API input to A2A
        # A2A will automatically generate context_id if not provided
        a2a_message = self._convert_responses_input_to_a2a(input_data)
        
        # Include system prompt and tools in metadata (AgentExecutor will extract these)
        self._add_system_prompt_and_tools_to_metadata(a2a_message, system_prompt, tools, instructions)
        
        task_id = None
        context_id = None  # Will be extracted from A2A response
        final_content = ""
        
        # Process conversation loop until completion
        while True:
            # Send message - Client.send_message returns an async generator
            # Client.send_message expects just the message object, not SendMessageRequest
            
            # Iterate over events from send_message
            state_value = None
            tool_calls = None
            async for event in a2a_client.send_message(a2a_message):
                # Extract event data
                event_data = extract_event_data(event)
                event_kind = self._get_attr(event_data, 'kind')
                
                # Only process status-update events
                if event_kind != 'status-update':
                    continue
                
                # Extract task info from event
                extracted_task_id, extracted_context_id, task = self._extract_task_info(event)
                if extracted_task_id:
                    task_id = extracted_task_id
                if extracted_context_id:
                    context_id = extracted_context_id
                
                # Get status from event
                event_status = self._get_attr(event_data, 'status')
                if not event_status:
                    raise ValueError("Status-update event has no status")
                
                # Extract state from event status
                state_value = self._extract_state_value(event_status)
                
                # Check if task is completed
                if state_value == "completed":
                    # Extract final content from status.message (trust it as source of truth)
                    status = self._get_attr(task, 'status')
                    if status:
                        status_message = self._extract_message_from_status(status)
                        if status_message:
                            parts = self._get_attr(status_message, 'parts', [])
                            final_content = self._extract_text_from_parts(parts)
                    break
                
                # Check if tool calls are needed
                if state_value == "input-required":
                    # Tool calls MUST be in the event status when input-required is emitted
                    tool_calls = self._extract_tool_calls_from_status(event_status)
                    if not tool_calls:
                        raise ValueError("input-required state but no tool calls found in event status")
                    
                    # Execute tools using the single environment URI
                    tool_results = []
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("arguments", {})
                        call_id = tc.get("call_id", "")
                        
                        result = await self._execute_tool_via_mcp(tool_name, tool_args, self.environment_uri)
                        # Extract text from result dict before passing to build_tool_result_message
                        output_text = self._extract_tool_output(result)
                        tool_results.append({
                            "call_id": call_id,
                            "name": tool_name,
                            "output": output_text,
                        })
                    
                    # Build tool result message and send it
                    a2a_message = self._build_tool_result_message(tool_results, task_id, context_id)
                    # Include system prompt and tools in metadata for tool result message too
                    self._add_system_prompt_and_tools_to_metadata(a2a_message, system_prompt, tools, instructions)
                    # Break from inner loop to continue outer loop with new message
                    break
                
                # In non-streaming mode, skip "working" states - we only want the final "completed" state
                # Working states are incremental updates that we don't need in non-streaming
                if state_value == "working":
                    continue
            
            # If we broke from the loop due to tool calls, continue outer loop
            if state_value == "input-required" and tool_calls:
                continue
            
            # If we completed or broke for other reasons, exit outer loop
            if state_value == "completed" or state_value != "input-required":
                break
        
        # Convert to Responses API format - get final status from task
        if not task:
            raise ValueError("No task available for response conversion")
        final_status = self._get_attr(task, 'status')
        return self._convert_a2a_response_to_responses(task, final_content, final_status)
    
    async def _handle_responses_streaming(self, body: Dict[str, Any]):
        """Handle streaming /v1/responses request.
        
        Note: The root span is created in the endpoint handler wrapper function.
        This function runs within that span's context, so all child operations
        (OpenAI, MCP) will automatically be nested under the root span.
        """
        try:
            # Responses API uses 'input' instead of 'messages'
            input_data = body.get("input")
            if input_data is None:
                yield f"data: {json.dumps({'error': {'message': 'input is required'}})}\n\n"
                return
        
            # Fetch system prompt and tools from Environment (MCP) once at the start
            instructions = body.get("instructions")
            system_prompt, tools = await self._fetch_system_prompt_and_tools(instructions=instructions)
            
            # Use ClientFactory (supports JSON-RPC by default)
            a2a_client = await ClientFactory.connect(self.agent_base_url)
            
            # Convert Responses API input to A2A
            # A2A will automatically generate context_id if not provided
            a2a_message = self._convert_responses_input_to_a2a(input_data)
            
            # Include system prompt and tools in metadata (AgentExecutor will extract these)
            self._add_system_prompt_and_tools_to_metadata(a2a_message, system_prompt, tools, instructions)
            
            task_id = None
            context_id = None  # Will be extracted from A2A response
            accumulated_content = ""
            last_sent_content = ""  # Track what we've already sent

            # Process conversation loop
            while True:
                    # Send message - Client.send_message() already returns an async generator (streaming)
                    # There is no separate send_message_streaming method in the new Client API
                    
                    tool_calls_found = False
                    
                    async for event in a2a_client.send_message(a2a_message):
                        # Extract event data
                        event_data = extract_event_data(event)
                        
                        # Extract state from event
                        event_kind = self._get_attr(event_data, 'kind')
                        
                        # Check if this is a status-update (TaskStatusUpdateEvent)
                        is_status_update = event_kind == 'status-update'
                        
                        if is_status_update:
                            event_status = self._get_attr(event_data, 'status')
                            if not event_status:
                                raise ValueError("Status-update event has no status")
                            
                            state_value = self._extract_state_value(event_status)
                            # Extract task info from event
                            extracted_task_id, extracted_context_id, task = self._extract_task_info(event)
                            if extracted_task_id:
                                task_id = extracted_task_id
                            if extracted_context_id:
                                context_id = extracted_context_id
                            
                            # Extract is_final from TaskStatusUpdateEvent
                            is_final = self._get_attr(event_data, 'final', False)
                            
                            # Stream content updates
                            if state_value == "working":
                                # Get message from status
                                status_message = self._extract_message_from_status(event_status)
                                if status_message:
                                    parts = self._get_attr(status_message, 'parts', [])
                                    current_text = self._extract_text_from_parts(parts)
                                    
                                    # Only send the new content (delta)
                                    if current_text and current_text != last_sent_content:
                                        # Find the new part
                                        if current_text.startswith(last_sent_content):
                                            delta_content = current_text[len(last_sent_content):]
                                        else:
                                            # If text doesn't start with what we sent, send the whole thing
                                            delta_content = current_text
                                        
                                        if delta_content:
                                            accumulated_content = current_text
                                            last_sent_content = current_text
                                            chunk_data = {
                                                'id': f'resp_{uuid4().hex}',
                                                'object': 'response.delta',
                                                'created_at': int(time.time()),
                                                'model': self.model_name,
                                                'output': [{
                                                    'type': 'message',
                                                    'delta': {
                                                        'content': [{
                                                            'type': 'output_text',
                                                            'text': delta_content
                                                        }]
                                                    }
                                                }]
                                            }
                                            yield f"data: {json.dumps(chunk_data)}\n\n"
                            
                            # Check if completed
                            if state_value == "completed":
                                # Extract final content from completed message (trust status.message)
                                status_message = self._extract_message_from_status(event_status)
                                if not status_message:
                                    raise ValueError("Completed state but no message in event status")
                                parts = self._get_attr(status_message, 'parts', [])
                                final_content = self._extract_text_from_parts(parts)
                                
                                # Send any remaining content that hasn't been sent yet
                                if final_content:
                                    if last_sent_content:
                                        if final_content.startswith(last_sent_content):
                                            remaining_content = final_content[len(last_sent_content):]
                                        else:
                                            remaining_content = final_content
                                    else:
                                        # No content sent yet, send the full thing
                                        remaining_content = final_content
                                    
                                    if remaining_content:
                                        chunk_data = {
                                            'id': f'resp_{uuid4().hex}',
                                            'object': 'response.delta',
                                            'created_at': int(time.time()),
                                            'model': self.model_name,
                                            'output': [{
                                                'type': 'message',
                                                'delta': {
                                                    'content': [{
                                                        'type': 'output_text',
                                                        'text': remaining_content
                                                    }]
                                                }
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                        last_sent_content = final_content
                                
                                # Only return if this is the final message (is_final flag is True)
                                # Otherwise continue streaming to get more content
                                if is_final:
                                    # Send completion marker
                                    chunk_data = {
                                        'id': f'resp_{uuid4().hex}',
                                        'object': 'response.delta',
                                        'created_at': int(time.time()),
                                        'model': self.model_name,
                                        'output': [{
                                            'type': 'message',
                                            'delta': {}
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return
                                # If not final, continue streaming (might be intermediate completed state)
                            
                            # Check if tool calls needed
                            if state_value == "input-required":
                                # Tool calls MUST be in the event status when input-required is emitted
                                tool_calls = self._extract_tool_calls_from_status(event_status)
                                if not tool_calls:
                                    raise ValueError("input-required state but no tool calls found in event status")
                                
                                tool_calls_found = True
                                
                                tool_results = []
                                for tc in tool_calls:
                                    tool_name = tc.get("name", "")
                                    tool_args = tc.get("arguments", {})
                                    call_id = tc.get("call_id", "")
                                    
                                    result = await self._execute_tool_via_mcp(tool_name, tool_args, self.environment_uri)
                                    # Extract text from result dict before passing to build_tool_result_message
                                    output_text = self._extract_tool_output(result)
                                    tool_results.append({
                                        "call_id": call_id,
                                        "name": tool_name,
                                        "output": output_text,
                                    })
                                
                                # Build tool result message and continue outer loop
                                a2a_message = self._build_tool_result_message(tool_results, task_id, context_id)
                                # Include system prompt and tools in metadata for tool result message too
                                self._add_system_prompt_and_tools_to_metadata(a2a_message, system_prompt, tools, instructions)
                                # Reset tracking for new streaming request
                                accumulated_content = ""
                                last_sent_content = ""
                                break  # Break from inner loop, continue outer loop
                    
                    # If we broke from inner loop due to tool calls, continue outer loop
                    if tool_calls_found:
                        continue
                    else:
                        # No more chunks and no tool calls - break
                        break
        finally:
            # Span is managed by the wrapper function in the endpoint handler
            pass
