"""ResponsesAPI class - Provides /v1/responses endpoint for agent applications."""

import json
import time
from typing import Dict, Any, List, Optional, Callable
from uuid import uuid4
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

from a2a.client import A2ACardResolver, A2AClient
import httpx
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Message,
    Part,
    DataPart,
    Role,
)
from a2a.client.helpers import create_text_message_object
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
from mcp.types import TextContent

from timestep.utils.message_helpers import (
    TOOL_CALLS_KEY,
    TOOL_RESULTS_KEY,
)


class ResponsesAPI:
    """ResponsesAPI provides a /v1/responses endpoint for agent applications.
    
    This component encapsulates all logic for handling the Responses API endpoint,
    including streaming and non-streaming modes, tool execution, and A2A protocol
    integration.
    """
    
    def __init__(
        self,
        agent: Any,  # Agent instance
        agent_base_url: str,
        context_id_to_environment_uri: Dict[str, str],
        sampling_callback: Optional[Callable] = None,
    ):
        """Initialize ResponsesAPI.
        
        Args:
            agent: The Agent instance
            agent_base_url: Base URL for the agent (e.g., "http://localhost:9999")
            context_id_to_environment_uri: Mapping from context_id to MCP environment URI
            sampling_callback: Optional MCP sampling callback for handoffs
        """
        self.agent = agent
        self.agent_base_url = agent_base_url
        self.context_id_to_environment_uri = context_id_to_environment_uri
        self.sampling_callback = sampling_callback
        
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
        return tool_result_msg
    
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
            return create_text_message_object(role=Role.user, content=input_data)
        
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
        
        # Use A2AClient for JSON-RPC agents (ClientFactory doesn't support JSON-RPC yet)
        async with httpx.AsyncClient(timeout=300.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.agent_base_url)
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Convert Responses API input to A2A
            a2a_message = self._convert_responses_input_to_a2a(input_data)
            if context_id:
                a2a_message.context_id = context_id
            
            task_id = None
            final_content = ""
            
            # Process conversation loop until completion
            while True:
                # Send message
                send_request = SendMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(message=a2a_message)
                )
                response = await a2a_client.send_message(send_request)
                
                # Extract task from response - response is a Pydantic model
                # Try to get result from dict representation first (most reliable)
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else {}
                task_data = response_dict.get('result')
                
                if not task_data:
                    # Try direct attribute access as fallback
                    task_data = getattr(response, 'result', None)
                
                if not task_data:
                    raise HTTPException(status_code=500, detail="Failed to get result from response")
                
                # Convert to object-like access if it's a dict
                if isinstance(task_data, dict):
                    class TaskObj:
                        def __init__(self, d):
                            for k, v in d.items():
                                if isinstance(v, dict):
                                    setattr(self, k, TaskObj(v))
                                else:
                                    setattr(self, k, v)
                    task = TaskObj(task_data)
                else:
                    task = task_data
                
                if not task:
                    raise HTTPException(status_code=500, detail="Failed to get task from response")
                
                task_id = getattr(task, 'id', None) or task_id
                context_id = getattr(task, 'context_id', None) or context_id
                
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
                        
                        # Build tool result message and continue loop to get next response
                        a2a_message = self._build_tool_result_message(tool_results, task_id, context_id)
                        continue  # Continue outer loop to send tool results and get next response
                    else:
                        # No tool calls but input required - break
                        break
                
                # If working state, accumulate content (shouldn't happen in non-streaming, but handle it)
                if state_value == "working":
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
                    # In non-streaming, working state means we should continue to get final response
                    continue
            
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
        
        # Use A2AClient for JSON-RPC agents (ClientFactory doesn't support JSON-RPC yet)
        async with httpx.AsyncClient(timeout=300.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.agent_base_url)
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Convert Responses API input to A2A
            a2a_message = self._convert_responses_input_to_a2a(input_data)
            if context_id:
                a2a_message.context_id = context_id
            
            task_id = None
            accumulated_content = ""
            last_sent_content = ""  # Track what we've already sent
            
            # Process conversation loop
            while True:
                # Send streaming message
                streaming_request = SendStreamingMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(message=a2a_message)
                )
                
                stream_response = a2a_client.send_message_streaming(streaming_request)
                
                tool_calls_found = False
                async for chunk in stream_response:
                    # Handle chunk - can be dict or object
                    chunk_dict = chunk.model_dump() if hasattr(chunk, 'model_dump') else (dict(chunk) if hasattr(chunk, '__dict__') else chunk)
                    event = chunk_dict.get('result') if isinstance(chunk_dict, dict) else (getattr(chunk, 'result', None) or chunk)
                    
                    # Handle both dict and object access
                    if isinstance(event, dict):
                        event_kind = event.get('kind')
                        event_status = event.get('status', {})
                        if isinstance(event_status, dict):
                            state_value = event_status.get('state', {}).get('value') if isinstance(event_status.get('state'), dict) else event_status.get('state')
                        else:
                            state_value = getattr(event_status, 'state', {}).get('value') if hasattr(event_status, 'state') else None
                    else:
                        event_kind = getattr(event, 'kind', None)
                        event_status = getattr(event, 'status', None)
                        if event_status:
                            state_obj = getattr(event_status, 'state', None)
                            state_value = getattr(state_obj, 'value', None) if state_obj else None
                        else:
                            state_value = None
                    
                    # Check if this is a task or status-update
                    is_task = event_kind == 'task' or (isinstance(event, dict) and event.get('kind') == 'task')
                    is_status_update = event_kind == 'status-update' or (isinstance(event, dict) and event.get('kind') == 'status-update')
                    
                    if is_task or is_status_update:
                        task = event
                        
                        # Extract task_id and context_id
                        if isinstance(task, dict):
                            task_id = task.get('taskId') or task.get('task_id') or task_id
                            context_id = task.get('contextId') or task.get('context_id') or context_id
                            is_final = task.get('final', False)
                        else:
                            task_id = getattr(task, 'taskId', None) or getattr(task, 'task_id', None) or task_id
                            context_id = getattr(task, 'contextId', None) or getattr(task, 'context_id', None) or context_id
                            is_final = getattr(task, 'final', False)
                        
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
                            
                            # Also check the event/task itself for history
                            # The event might be a task object with history
                            task_history = []
                            if isinstance(event, dict):
                                # Check if event itself has history
                                if 'history' in event:
                                    task_history = event.get('history', [])
                                elif isinstance(task, dict) and 'history' in task:
                                    task_history = task.get('history', [])
                            else:
                                # Event is an object, try to get history
                                if hasattr(event, 'history'):
                                    task_history = getattr(event, 'history', [])
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
