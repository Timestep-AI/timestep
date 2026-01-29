"""Loop class - Orchestrates agent-environment interaction loop until completion."""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

from a2a.client import ClientFactory
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
    convert_agui_event_to_responses_api,
)
from timestep.utils.event_helpers import (
    extract_event_data,
    extract_task_from_event,
    extract_task_from_tuple,
    add_canonical_type_to_message,
)


class Loop:
    """Loop orchestrates agent-environment interaction until completion (full rollout/episode).
    
    This component provides a /v1/responses endpoint (OpenAI-compatible) that:
    1. Handles tool execution via MCP
    2. Continues agent loop until final response (complete rollout/episode)
    3. Supports both streaming and non-streaming modes
    """
    
    def __init__(
        self,
        agent: Any,  # Agent instance
        agent_base_url: str,
        context_id_to_environment_uri: Dict[str, str],
        sampling_callback: Optional[Callable] = None,  # Deprecated: will be auto-created
    ):
        """Initialize Loop.
        
        Args:
            agent: The Agent instance
            agent_base_url: Base URL for the agent (e.g., "http://localhost:9999")
            context_id_to_environment_uri: Mapping from context_id to MCP environment URI
            sampling_callback: Deprecated - will be auto-created (kept for backward compatibility)
        """
        self.agent = agent
        self.agent_base_url = agent_base_url
        self.context_id_to_environment_uri = context_id_to_environment_uri
        # Always create sampling callback (handoff tool is always in Environment)
        self.sampling_callback = sampling_callback if sampling_callback else self._create_sampling_callback()
        
        # Create FastAPI app for this component
        self.app = FastAPI()
        
        # Register the /v1/responses endpoint
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
    
    def _extract_tool_calls(self, task: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from task status message DataPart."""
        # Handle both dict and object access
        if isinstance(task, dict):
            status = task.get('status', {})
        else:
            status = getattr(task, 'status', None)
        
        if not status:
            return None
        
        # Get message from status
        if isinstance(status, dict):
            status_message = status.get('message')
        else:
            status_message = getattr(status, 'message', None)
        
        if not status_message:
            return None
        
        # Get parts from message
        if isinstance(status_message, dict):
            parts = status_message.get('parts', [])
        else:
            parts = getattr(status_message, 'parts', [])
        
        if not parts:
            return None
        
        # Look for tool calls in parts
        for part in parts:
            part_data = part.root if hasattr(part, 'root') else part
            if isinstance(part_data, dict):
                part_kind = part_data.get('kind')
                part_data_dict = part_data.get('data') if part_kind == 'data' else None
            else:
                part_kind = getattr(part_data, 'kind', None)
                part_data_dict = getattr(part_data, 'data', None) if part_kind == 'data' else None
            
            if part_kind == 'data' and part_data_dict:
                if isinstance(part_data_dict, dict):
                    tool_calls = part_data_dict.get(TOOL_CALLS_KEY)
                else:
                    tool_calls = getattr(part_data_dict, TOOL_CALLS_KEY, None) if hasattr(part_data_dict, TOOL_CALLS_KEY) else None
                
                if tool_calls:
                    return tool_calls
        
        return None
    
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
        # Add canonical_type for tool call results
        tool_result_msg = add_canonical_type_to_message(tool_result_msg, "ToolCallResultEvent")
        return tool_result_msg
    
    def _extract_final_message(self, task: Any) -> str:
        """Extract final message text from a completed task.
        
        Only extracts the final completed message, not incremental updates.
        """
        message_text = ""
        
        # First, try to extract from task.status.message (this is the final completed message)
        status = getattr(task, 'status', None) if hasattr(task, 'status') else (task.get('status') if isinstance(task, dict) else None)
        if status:
            # Get message from status - handle both dict and object access
            if isinstance(status, dict):
                status_message = status.get('message')
            else:
                status_message = getattr(status, 'message', None)
            
            if status_message:
                # Get parts from message - handle both dict and object access
                if isinstance(status_message, dict):
                    parts = status_message.get('parts', [])
                else:
                    parts = getattr(status_message, 'parts', [])
                
                for part in parts:
                    part_data = part.root if hasattr(part, 'root') else part
                    if isinstance(part_data, dict):
                        if part_data.get('kind') == 'text' and part_data.get('text'):
                            message_text += part_data.get('text', '')
                    elif hasattr(part_data, 'kind') and part_data.kind == 'text':
                        if hasattr(part_data, 'text'):
                            message_text += part_data.text
        
        # If we got text from status.message, return it (this is the final message)
        if message_text.strip():
            return message_text.strip()
        
        # Otherwise, check task history for the LAST agent message (not all of them)
        # This handles cases where status.message might not be set
        if isinstance(task, dict):
            task_history = task.get('history', [])
        else:
            task_history = getattr(task, 'history', []) if hasattr(task, 'history') else []
        
        if task_history:
            # Iterate in reverse to find the last agent message
            for msg in reversed(task_history):
                # Handle both dict and object access for message
                if isinstance(msg, dict):
                    msg_role = msg.get('role')
                    msg_parts = msg.get('parts', [])
                    msg_content = msg.get('content', '')
                else:
                    msg_role = getattr(msg, 'role', None)
                    msg_parts = getattr(msg, 'parts', []) if hasattr(msg, 'parts') else []
                    msg_content = getattr(msg, 'content', '') if hasattr(msg, 'content') else ''
                
                if msg_role == Role.agent or str(msg_role) == "agent":
                    # Extract text from parts - only from this last agent message
                    for part in msg_parts:
                        part_data = part.root if hasattr(part, 'root') else part
                        if isinstance(part_data, dict):
                            if part_data.get('kind') == 'text' and part_data.get('text'):
                                message_text += part_data.get('text', '')
                        elif hasattr(part_data, 'kind') and part_data.kind == 'text':
                            if hasattr(part_data, 'text'):
                                message_text += part_data.text
                    # Also check for direct content if no parts
                    if not msg_parts and msg_content:
                        message_text += str(msg_content)
                    
                    # Found the last agent message, return it (don't accumulate from earlier messages)
                    break
        
        return message_text.strip()
    
    async def _handle_agent_handoff(self, agent_uri: str, message: str) -> str:
        """Handle agent handoff by calling the A2A agent at agent_uri."""
        # Use ClientFactory (supports JSON-RPC by default)
        a2a_client = await ClientFactory.connect(agent_uri)
        
        try:
            message_obj = create_text_message_object(role="user", content=message)
            
            # Process message using ClientFactory (non-streaming, polling mode)
            final_message = ""
            task_id = None
            context_id = None
            
            # Process message stream until completion
            while True:
                # Send message - Client.send_message returns an async generator
                # Client.send_message expects just the message object, not SendMessageRequest
                
                # Process message stream
                task = None
                state_value = None
                tool_calls = None
                async for event in a2a_client.send_message(message_obj):
                    # Extract Task from tuple (first element contains full Task object)
                    task = extract_task_from_tuple(event)
                    
                    # Extract TaskStatusUpdateEvent for state information (second element)
                    event_data = extract_event_data(event)
                    
                    if task:
                        task_id = getattr(task, 'id', None) or task_id
                        context_id = getattr(task, 'context_id', None) or context_id
                    
                    if not task:
                        logger.warning("Handoff: No task extracted from event, continuing...")
                        continue
                    
                    # Get status from Task object
                    status = getattr(task, 'status', None)
                    if not status:
                        logger.warning("Handoff: Task has no status, continuing...")
                        continue
                    
                    # Extract state from TaskStatusUpdateEvent if available, otherwise from Task
                    if event_data and hasattr(event_data, 'status'):
                        event_status = getattr(event_data, 'status', None)
                        if event_status:
                            if hasattr(event_status, 'state'):
                                state_obj = getattr(event_status, 'state', None)
                                if state_obj:
                                    state_value = getattr(state_obj, 'value', None)
                    else:
                        # Fallback to Task status
                        if isinstance(status, dict):
                            state_value = status.get('state', {}).get('value') if isinstance(status.get('state'), dict) else status.get('state')
                        else:
                            state_obj = getattr(status, 'state', None)
                            state_value = getattr(state_obj, 'value', None) if state_obj else None
                    
                    if state_value == "completed":
                        final_message = self._extract_final_message(task)
                        break
                    
                    if state_value == "input-required":
                        tool_calls = self._extract_tool_calls(task)
                        if tool_calls:
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
                            message_obj = tool_result_msg
                            # Break from inner loop to continue outer loop with new message
                            break
                        else:
                            logger.warning("Handoff: input-required but no tool calls found")
                            break
                
                # If we completed, exit outer loop
                if state_value == "completed":
                    break
                
                # If we broke from the loop due to tool calls, continue outer loop
                if state_value == "input-required" and tool_calls:
                    continue
                
                # Otherwise, exit
                break
            
            return final_message.strip() or "Task completed."
        except Exception as e:
            logger.error(f"Error during handoff: {str(e)}", exc_info=True)
            return f"Error during handoff: {str(e)}"
    
    def _create_sampling_callback(self):
        """Create MCP sampling callback for handoff."""
        async def sampling_callback(
            context: RequestContext["ClientSession", Any],
            params: mcp_types.CreateMessageRequestParams,
        ) -> mcp_types.CreateMessageResult | mcp_types.CreateMessageResultWithTools | mcp_types.ErrorData:
            """MCP sampling callback that handles sampling requests from the MCP server."""
            try:
                # Extract agent_uri from metadata (passed by handoff tool)
                agent_uri = (params.metadata or {}).get("agent_uri")
                if not agent_uri:
                    return mcp_types.ErrorData(
                        code=mcp_types.INVALID_PARAMS,
                        message="agent_uri is required for sampling",
                    )
                
                # Extract message text from params.messages (SamplingMessage objects with TextContent)
                message_text = params.messages[0].content.text if params.messages else "Please help with this task."
                
                result_text = await self._handle_agent_handoff(agent_uri=agent_uri, message=message_text)
                
                return mcp_types.CreateMessageResult(
                    role="assistant",
                    content=mcp_types.TextContent(type="text", text=result_text),
                    model="a2a-agent"
                )
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
                                try:
                                    # Try to parse as JSON to extract "response" field
                                    parsed = json.loads(result_text)
                                    if isinstance(parsed, dict) and "response" in parsed:
                                        result_text = parsed["response"]
                                except (json.JSONDecodeError, KeyError, TypeError):
                                    # If parsing fails, use the text as-is
                                    pass
                            
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
            # Add canonical_type for user text message
            user_message = add_canonical_type_to_message(user_message, "TextMessageContentEvent")
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
                # Add canonical_type for tool call results
                a2a_message = add_canonical_type_to_message(a2a_message, "ToolCallResultEvent")
            else:
                # Add canonical_type for user text message
                a2a_message = add_canonical_type_to_message(a2a_message, "TextMessageContentEvent")
            return a2a_message
        
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _convert_a2a_response_to_responses(self, task: Any, content: str = "", status: Any = None) -> Dict[str, Any]:
        """Convert A2A task response to Responses API format."""
        # Use provided status or get from task
        if status is None:
            status = getattr(task, 'status', None)
        
        # Extract content from task status message
        if not content and status:
            status_message = getattr(status, 'message', None) if hasattr(status, 'message') else status.get('message') if isinstance(status, dict) else None
            if status_message:
                if isinstance(status_message, dict):
                    parts = status_message.get('parts', [])
                else:
                    parts = getattr(status_message, 'parts', [])
                for part in parts:
                    part_data = part.root if hasattr(part, 'root') else part
                    if hasattr(part_data, 'kind') and part_data.kind == 'text':
                        if hasattr(part_data, 'text'):
                            content += part_data.text
        
        # Build output items array
        output_items = []
        
        # Add tool calls if present - create a task-like object for extract_tool_calls
        task_for_extraction = task
        if status and not hasattr(task, 'status'):
            # Create a wrapper object with status attribute
            class TaskWrapper:
                def __init__(self, task_obj, status_obj):
                    self._task = task_obj
                    self.status = status_obj
            task_for_extraction = TaskWrapper(task, status)
        
        tool_calls = self._extract_tool_calls(task_for_extraction)
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
            "model": self.agent.model,
            "output": output_items
        }
        
        return response
    
    async def _handle_responses_non_streaming(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Handle non-streaming /v1/responses request."""
        # Responses API uses 'input' instead of 'messages'
        input_data = body.get("input")
        if input_data is None:
            raise HTTPException(status_code=400, detail="input is required")
        
        # Get context_id (use first available or generate one)
        context_id = None
        for cid in self.context_id_to_environment_uri.keys():
            context_id = cid
            break
        if not context_id:
            context_id = str(uuid4())
        
        environment_uri = self.context_id_to_environment_uri.get(context_id)
        if not environment_uri:
            raise HTTPException(status_code=400, detail=f"No environment found for context_id: {context_id}")
        
        # Use ClientFactory (supports JSON-RPC by default)
        a2a_client = await ClientFactory.connect(self.agent_base_url)
        
        # Convert Responses API input to A2A
        a2a_message = self._convert_responses_input_to_a2a(input_data)
        if context_id:
            a2a_message.context_id = context_id
        
        task_id = None
        final_content = ""
        
        # Process conversation loop until completion
        while True:
            # Send message - Client.send_message returns an async generator
            # Client.send_message expects just the message object, not SendMessageRequest
            
            # Iterate over events from send_message
            task = None
            state_value = None
            tool_calls = None
            async for event in a2a_client.send_message(a2a_message):
                # Extract event data (handles both tuple and direct event objects)
                event_data = extract_event_data(event)
                
                # Extract the Task object from the tuple (first element) if available
                # This gives us the full task with history
                task_obj = extract_task_from_tuple(event)
                
                # Extract task_id from TaskStatusUpdateEvent (event_data)
                # TaskStatusUpdateEvent uses 'task_id' (snake_case)
                if hasattr(event_data, 'task_id'):
                    extracted_task_id = event_data.task_id
                elif isinstance(event_data, dict):
                    extracted_task_id = event_data.get('task_id')
                else:
                    extracted_task_id = None
                
                if extracted_task_id:
                    task_id = extracted_task_id
                
                # Extract context_id from TaskStatusUpdateEvent
                if hasattr(event_data, 'context_id'):
                    extracted_context_id = event_data.context_id
                elif isinstance(event_data, dict):
                    extracted_context_id = event_data.get('context_id')
                else:
                    extracted_context_id = None
                
                if extracted_context_id:
                    context_id = extracted_context_id
                
                # Use the Task object from tuple if available (has full history), otherwise use event_data
                if task_obj:
                    task = task_obj
                else:
                    # Fallback: try to extract task from event_data
                    task = extract_task_from_event(event_data)
                
                if not task:
                    continue
                
                # Get status - handle both object and dict access
                status = getattr(task, 'status', None)
                if status is None:
                    status = {}
                if isinstance(status, dict):
                    state_value = status.get('state', {}).get('value') if isinstance(status.get('state'), dict) else status.get('state')
                else:
                    state_value = getattr(status, 'state', None)
                    if state_value:
                        state_value = getattr(state_value, 'value', None) or str(state_value)
                
                # Check if task is completed
                if state_value == "completed":
                    # Extract final content from status.message
                    status_message = getattr(status, 'message', None) if hasattr(status, 'message') else status.get('message') if isinstance(status, dict) else None
                    if status_message:
                        # Handle both object and dict access for message
                        if isinstance(status_message, dict):
                            parts = status_message.get('parts', [])
                        else:
                            parts = getattr(status_message, 'parts', [])
                        for part in parts:
                            part_data = part.root if hasattr(part, 'root') else part
                            if hasattr(part_data, 'kind') and part_data.kind == 'text':
                                if hasattr(part_data, 'text'):
                                    final_content += part_data.text
                    
                    # Also check task history for agent messages (important after tool execution)
                    # Only get the LAST agent message to avoid incremental updates
                    task_history = getattr(task, 'history', None) if hasattr(task, 'history') else (task.get('history') if isinstance(task, dict) else None)
                    if task_history:
                        # Go through history in reverse to find the last agent message
                        for msg in reversed(task_history):
                            # Handle both dict and object access
                            if isinstance(msg, dict):
                                msg_role = msg.get('role')
                                msg_parts = msg.get('parts', [])
                            else:
                                msg_role = getattr(msg, 'role', None)
                                msg_parts = getattr(msg, 'parts', [])
                            
                            if msg_role == 'agent' or str(msg_role) == 'agent':
                                # Only get text from the last agent message (first in reverse)
                                agent_text = ""
                                for part in msg_parts:
                                    part_data = part.root if hasattr(part, 'root') else part
                                    if isinstance(part_data, dict):
                                        if part_data.get('kind') == 'text' and part_data.get('text'):
                                            agent_text += part_data.get('text', '')
                                    elif hasattr(part_data, 'kind') and part_data.kind == 'text':
                                        if hasattr(part_data, 'text'):
                                            agent_text += part_data.text
                                
                                # Use the agent text if we don't have content from status.message
                                if agent_text and not final_content:
                                    final_content = agent_text
                                elif agent_text and agent_text != final_content:
                                    # If agent text is different, use the longer one (final version)
                                    final_content = agent_text if len(agent_text) > len(final_content) else final_content
                                
                                # Only use the first (last in reverse) agent message
                                break
                    
                    break
                
                # Check if tool calls are needed
                if state_value == "input-required":
                    tool_calls = self._extract_tool_calls(task)
                    if tool_calls:
                        # Check task history
                        task_history = getattr(task, 'history', None) if hasattr(task, 'history') else (task.get('history') if isinstance(task, dict) else None)
                        if not task_history:
                            logger.warning(f"task.history is None or empty when tool calls received")
                        
                        # Execute tools
                        tool_results = []
                        for tc in tool_calls:
                            tool_name = tc.get("name", "")
                            tool_args = tc.get("arguments", {})
                            call_id = tc.get("call_id", "")
                            
                            result = await self._execute_tool_via_mcp(tool_name, tool_args, environment_uri)
                            # Extract text from result dict before passing to build_tool_result_message
                            output_text = self._extract_tool_output(result)
                            tool_results.append({
                                "call_id": call_id,
                                "name": tool_name,
                                "output": output_text,
                            })
                        
                        # Build tool result message and send it
                        a2a_message = self._build_tool_result_message(tool_results, task_id, context_id)
                        # Break from inner loop to continue outer loop with new message
                        break
                    else:
                        # No tool calls but input required - break
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
        final_status = getattr(task, 'status', None) if task else None
        return self._convert_a2a_response_to_responses(task, final_content, final_status)
    
    async def _handle_responses_streaming(self, body: Dict[str, Any]):
        """Handle streaming /v1/responses request."""
        # Responses API uses 'input' instead of 'messages'
        input_data = body.get("input")
        if input_data is None:
            yield f"data: {json.dumps({'error': {'message': 'input is required'}})}\n\n"
            return
        
        # Get context_id
        context_id = None
        for cid in self.context_id_to_environment_uri.keys():
            context_id = cid
            break
        if not context_id:
            context_id = str(uuid4())
        
        environment_uri = self.context_id_to_environment_uri.get(context_id)
        if not environment_uri:
            yield f"data: {json.dumps({'error': {'message': f'No environment found for context_id: {context_id}'}})}\n\n"
            return
        
        # Use ClientFactory (supports JSON-RPC by default)
        a2a_client = await ClientFactory.connect(self.agent_base_url)
        
        # Convert Responses API input to A2A
        a2a_message = self._convert_responses_input_to_a2a(input_data)
        if context_id:
            a2a_message.context_id = context_id
        
        task_id = None
        accumulated_content = ""
        last_sent_content = ""  # Track what we've already sent
        
        # Process conversation loop
        while True:
                # Send message - Client.send_message() already returns an async generator (streaming)
                # There is no separate send_message_streaming method in the new Client API
                
                tool_calls_found = False
                async for event in a2a_client.send_message(a2a_message):
                    # Extract event data (handles both tuple and direct event objects)
                    event_data = extract_event_data(event)
                    
                    # Handle both dict and object access for TaskStatusUpdateEvent
                    if isinstance(event_data, dict):
                        event_kind = event_data.get('kind')
                        event_status = event_data.get('status', {})
                        if isinstance(event_status, dict):
                            state_value = event_status.get('state', {}).get('value') if isinstance(event_status.get('state'), dict) else event_status.get('state')
                        else:
                            state_value = getattr(event_status, 'state', {}).get('value') if hasattr(event_status, 'state') else None
                    else:
                        event_kind = getattr(event_data, 'kind', None)
                        event_status = getattr(event_data, 'status', None)
                        if event_status:
                            state_obj = getattr(event_status, 'state', None)
                            state_value = getattr(state_obj, 'value', None) if state_obj else None
                        else:
                            state_value = None
                    
                    # Check if this is a status-update (TaskStatusUpdateEvent)
                    is_status_update = event_kind == 'status-update' or (isinstance(event_data, dict) and event_data.get('kind') == 'status-update')
                    
                    if is_status_update:
                        # Extract event data (handles both tuple and direct event objects)
                        event_data_for_extraction = extract_event_data(event)
                        
                        # Extract the Task object from the tuple (first element) if available
                        task_obj = extract_task_from_tuple(event)
                        
                        # Extract task_id from TaskStatusUpdateEvent (event_data)
                        # TaskStatusUpdateEvent uses 'task_id' (snake_case)
                        if hasattr(event_data_for_extraction, 'task_id'):
                            extracted_task_id = event_data_for_extraction.task_id
                        elif isinstance(event_data_for_extraction, dict):
                            extracted_task_id = event_data_for_extraction.get('task_id')
                        else:
                            extracted_task_id = None
                        
                        if extracted_task_id:
                            task_id = extracted_task_id
                        
                        # Extract context_id from TaskStatusUpdateEvent
                        if hasattr(event_data_for_extraction, 'context_id'):
                            extracted_context_id = event_data_for_extraction.context_id
                        elif isinstance(event_data_for_extraction, dict):
                            extracted_context_id = event_data_for_extraction.get('context_id')
                        else:
                            extracted_context_id = None
                        
                        if extracted_context_id:
                            context_id = extracted_context_id
                        
                        # Extract is_final from TaskStatusUpdateEvent
                        if hasattr(event_data_for_extraction, 'final'):
                            is_final = event_data_for_extraction.final
                        elif isinstance(event_data_for_extraction, dict):
                            is_final = event_data_for_extraction.get('final', False)
                        else:
                            is_final = False
                        
                        # Use the Task object from tuple if available (has full history), otherwise use event
                        if task_obj:
                            task = task_obj
                        else:
                            task = event
                        
                        # Stream content updates
                        if state_value == "working":
                            # Get message from status
                            if isinstance(event_status, dict):
                                status_message = event_status.get('message')
                            else:
                                status_message = getattr(event_status, 'message', None)
                            
                            if status_message:
                                # Handle both dict and object access
                                if isinstance(status_message, dict):
                                    parts = status_message.get('parts', [])
                                else:
                                    parts = getattr(status_message, 'parts', [])
                                
                                current_text = ""
                                for part in parts:
                                    part_data = part.root if hasattr(part, 'root') else part
                                    if isinstance(part_data, dict):
                                        if part_data.get('kind') == 'text' and part_data.get('text'):
                                            current_text += part_data.get('text', '')
                                    elif hasattr(part_data, 'kind') and part_data.kind == 'text':
                                        if hasattr(part_data, 'text'):
                                            current_text += part_data.text
                                
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
                                            'model': self.agent.model,
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
                            # Extract final content from completed message
                            if isinstance(event_status, dict):
                                status_message = event_status.get('message')
                            else:
                                status_message = getattr(event_status, 'message', None)
                            
                            final_content = accumulated_content
                            if status_message:
                                # Extract text from completed message
                                if isinstance(status_message, dict):
                                    parts = status_message.get('parts', [])
                                else:
                                    parts = getattr(status_message, 'parts', [])
                                
                                for part in parts:
                                    part_data = part.root if hasattr(part, 'root') else part
                                    if hasattr(part_data, 'kind') and part_data.kind == 'text':
                                        if hasattr(part_data, 'text'):
                                            final_content = part_data.text  # Use final content
                            
                            # Get task history from the Task object (from tuple)
                            task_history = []
                            if task:
                                if isinstance(task, dict):
                                    task_history = task.get('history', [])
                                elif hasattr(task, 'history'):
                                    task_history = getattr(task, 'history', [])
                            
                            if task_history:
                                # Get the last agent message from history
                                for msg in reversed(task_history):
                                    if isinstance(msg, dict):
                                        msg_role = msg.get('role')
                                        msg_parts = msg.get('parts', [])
                                    else:
                                        msg_role = getattr(msg, 'role', None)
                                        msg_parts = getattr(msg, 'parts', [])
                                    
                                    if msg_role == 'agent' or str(msg_role) == 'agent':
                                        agent_text = ""
                                        for part in msg_parts:
                                            part_data = part.root if hasattr(part, 'root') else part
                                            if isinstance(part_data, dict):
                                                if part_data.get('kind') == 'text' and part_data.get('text'):
                                                    agent_text += part_data.get('text', '')
                                            elif hasattr(part_data, 'kind') and part_data.kind == 'text':
                                                if hasattr(part_data, 'text'):
                                                    agent_text += part_data.text
                                        
                                        if agent_text:
                                            # Use the longer text (final version)
                                            if len(agent_text) > len(final_content):
                                                final_content = agent_text
                                        break  # Only use the last agent message
                            
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
                                        'model': self.agent.model,
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
                                    'model': self.agent.model,
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
                            tool_calls = self._extract_tool_calls(task)
                            if tool_calls:
                                # Check task history
                                task_history = []
                                if isinstance(task, dict):
                                    task_history = task.get('history', [])
                                elif hasattr(task, 'history'):
                                    task_history = getattr(task, 'history', [])
                                if not task_history:
                                    logger.warning(f"task.history is None or empty when tool calls received (streaming)")
                                
                                tool_calls_found = True
                                # Execute tools
                                tool_results = []
                                for tc in tool_calls:
                                    tool_name = tc.get("name", "")
                                    tool_args = tc.get("arguments", {})
                                    call_id = tc.get("call_id", "")
                                    
                                    result = await self._execute_tool_via_mcp(tool_name, tool_args, environment_uri)
                                    # Extract text from result dict before passing to build_tool_result_message
                                    output_text = self._extract_tool_output(result)
                                    tool_results.append({
                                        "call_id": call_id,
                                        "name": tool_name,
                                        "output": output_text,
                                    })
                                
                                # Build tool result message and continue outer loop
                                a2a_message = self._build_tool_result_message(tool_results, task_id, context_id)
                                # Reset tracking for new streaming request
                                accumulated_content = ""
                                last_sent_content = ""
                                break  # Break from inner loop, continue outer loop
                            else:
                                # No tool calls - finish
                                chunk_data = {
                                    'id': f'resp_{uuid4().hex}',
                                    'object': 'response.delta',
                                    'created_at': int(time.time()),
                                    'model': self.agent.model,
                                    'output': [{
                                        'type': 'message',
                                        'delta': {}
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                yield "data: [DONE]\n\n"
                                return
                
                # If we broke from inner loop due to tool calls, continue outer loop
                if tool_calls_found:
                    continue
                else:
                    # No more chunks and no tool calls - break
                    break
