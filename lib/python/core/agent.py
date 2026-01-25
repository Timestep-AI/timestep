"""Agent class - A2A Server that contains Loop internally."""

import json
import os
import time
import uuid
from typing import Dict, Optional, List, Any, AsyncGenerator
import uvicorn
from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    TransportProtocol,
    Message,
    Part,
    DataPart,
    Role,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    TaskStatusUpdateEvent,
)
from a2a.client.helpers import create_text_message_object
from a2a.client import A2AClient
import httpx

from timestep.core.loop import Loop
from timestep.utils.message_helpers import (
    extract_user_text_and_tool_results,
    TOOL_CALLS_KEY,
    TOOL_RESULTS_KEY,
)


class Agent:
    """Agent is an A2A Server that contains Loop (AgentExecutor) internally.
    
    The Agent:
    1. Contains Loop (AgentExecutor) internally
    2. Exposes agent URI via A2A protocol
    3. Responds to A2A requests
    4. Maps context_id to environment URI
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        model: str = "gpt-4o-mini",
        context_id_to_environment_uri: Optional[Dict[str, str]] = None,
        human_in_loop: bool = False,
    ):
        self.agent_id = agent_id
        self.name = name
        self.model = model
        self.context_id_to_environment_uri = context_id_to_environment_uri or {}
        self.human_in_loop = human_in_loop
        
        # Loop (AgentExecutor) is inside Agent
        self.loop = Loop(
            agent_id=agent_id,
            model=model,
            context_id_to_environment_uri=self.context_id_to_environment_uri,
            human_in_loop=human_in_loop,
        )
        
        # Create A2A server with Loop as AgentExecutor
        self.handler = DefaultRequestHandler(
            agent_executor=self.loop,
            task_store=InMemoryTaskStore(),
        )
        
        # Create agent card
        self.agent_card = self._create_agent_card()
        
        # Create A2A JSON-RPC app (primary)
        # A2AFastAPIApplication handles JSON-RPC requests at / (POST)
        # This is what A2AClient uses (JSON-RPC transport)
        self.app = A2AFastAPIApplication(
            agent_card=self.agent_card,
            http_handler=self.handler,
        )
        
        # Build A2A app - this returns a FastAPI app with:
        # - Agent card endpoint at /.well-known/agent-card.json
        # - JSON-RPC endpoint at / (POST)
        self.fastapi_app = self.app.build()
        
        # Add OpenAI-compatible /v1/chat/completions endpoint
        self._add_openai_endpoint()
    
    def _create_agent_card(self) -> AgentCard:
        """Create an agent card for this agent."""
        base_url = os.getenv("A2A_BASE_URL", "http://localhost:9999")
        return AgentCard(
            name=self.name,
            version="1.0.0",
            description=f"{self.name} agent",
            url=f"{base_url}",
            preferred_transport=TransportProtocol.http_json,
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[AgentSkill(
                id=self.agent_id,
                name=self.name,
                description=f"{self.name} agent",
                tags=[],
            )],
        )
    
    async def start(self, port: int = 9999, host: str = "0.0.0.0") -> str:
        """Start A2A server and return agent URI.
        
        Args:
            port: Port to run the A2A server on
            host: Host to bind to
            
        Returns:
            Agent URI (e.g., "http://localhost:9999")
        """
        # Convert bind address to HTTP hostname
        http_host = "localhost" if host == "0.0.0.0" else host
        
        # Set base URL environment variable if not set
        if not os.getenv("A2A_BASE_URL"):
            os.environ["A2A_BASE_URL"] = f"http://{http_host}:{port}"
        
        # Start server
        config = uvicorn.Config(
            app=self.fastapi_app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        
        # Start in background
        import asyncio
        asyncio.create_task(server.serve())
        
        # Return agent URI
        return f"http://{http_host}:{port}"
    
    def run(self, port: int = 9999, host: str = "0.0.0.0"):
        """Run the A2A server (blocking)."""
        # Convert bind address to HTTP hostname
        http_host = "localhost" if host == "0.0.0.0" else host
        
        # Set base URL environment variable if not set
        if not os.getenv("A2A_BASE_URL"):
            os.environ["A2A_BASE_URL"] = f"http://{http_host}:{port}"
        
        uvicorn.run(self.fastapi_app, host=host, port=port)
    
    def _convert_openai_messages_to_a2a(self, openai_messages: List[Dict[str, Any]]) -> Message:
        """Convert OpenAI messages array to A2A Message format.
        
        Args:
            openai_messages: List of OpenAI message dicts (role, content, tool_calls, etc.)
            
        Returns:
            A2A Message object with user text and tool results
        """
        user_text = ""
        tool_results: List[Dict[str, Any]] = []
        
        for msg in openai_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                # Extract text content
                if isinstance(content, str):
                    user_text += content
                elif isinstance(content, list):
                    # Handle content array format
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_text += item.get("text", "")
            
            elif role == "tool":
                # Extract tool results
                tool_call_id = msg.get("tool_call_id", "")
                tool_content = msg.get("content", "")
                
                if tool_call_id:
                    tool_results.append({
                        "call_id": tool_call_id,
                        "output": tool_content,
                    })
            
            # Skip system messages - handled by environment
            # Skip assistant messages - these are responses, not inputs
        
        # Create A2A message
        a2a_message = create_text_message_object(role=Role.user, content=user_text)
        
        # Add tool results if present
        if tool_results:
            a2a_message.parts.append(Part(DataPart(data={TOOL_RESULTS_KEY: tool_results})))
        
        return a2a_message
    
    def _convert_a2a_task_to_openai(
        self,
        task: Any,
        model: str,
        stream: bool = False,
        delta_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert A2A task to OpenAI ChatCompletion format.
        
        Args:
            task: A2A Task object or status update
            model: Model name to include in response
            stream: Whether this is for streaming response
            delta_content: Optional incremental content for streaming
            
        Returns:
            OpenAI ChatCompletion response dict
        """
        # Extract message content and tool calls from task status
        message_content = delta_content or ""
        tool_calls = []
        
        # Handle both Task objects and status update events (dict or object format)
        status = None
        if isinstance(task, dict):
            # Dict format from A2A client
            status = task.get("status")
        elif hasattr(task, "status"):
            status = task.status
        elif hasattr(task, "result") and hasattr(task.result, "status"):
            status = task.result.status
        else:
            status = task
        
        # Extract message from status (handle both dict and object)
        message = None
        if isinstance(status, dict):
            message = status.get("message")
        elif status and hasattr(status, "message"):
            message = status.message
        
        if message:
            # Handle both dict and object format for message
            parts = []
            if isinstance(message, dict):
                parts = message.get("parts", [])
            elif hasattr(message, "parts"):
                parts = message.parts
            
            for part in parts:
                # Handle both dict and object format for part
                part_data = None
                if isinstance(part, dict):
                    part_data = part
                elif hasattr(part, "root"):
                    part_data = part.root
                else:
                    part_data = part
                
                if isinstance(part_data, dict):
                    part_kind = part_data.get("kind")
                elif hasattr(part_data, "kind"):
                    part_kind = part_data.kind
                else:
                    continue
                
                if part_kind == "text":
                    text = None
                    if isinstance(part_data, dict):
                        text = part_data.get("text")
                    elif hasattr(part_data, "text"):
                        text = part_data.text
                    
                    if text:
                        if delta_content:
                            # For streaming, only use delta_content
                            pass
                        else:
                            message_content += text
                elif part_kind == "data":
                    data = None
                    if isinstance(part_data, dict):
                        data = part_data.get("data")
                    elif hasattr(part_data, "data"):
                        data = part_data.data
                    
                    if isinstance(data, dict):
                        calls = data.get(TOOL_CALLS_KEY, [])
                        if isinstance(calls, list):
                            tool_calls = calls
        
        # Determine finish reason
        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        elif status:
            state_value = None
            if isinstance(status, dict):
                state_value = status.get("state")
            elif hasattr(status, "state"):
                state_value = status.state.value if hasattr(status.state, "value") else str(status.state)
            
            if state_value == "input-required" and tool_calls:
                finish_reason = "tool_calls"
        
        # Build OpenAI response - extract task_id
        task_id = None
        if isinstance(task, dict):
            task_id = task.get("id")
        elif hasattr(task, "id"):
            task_id = task.id
        elif hasattr(task, "result") and hasattr(task.result, "id"):
            task_id = task.result.id
        
        if not task_id:
            task_id = str(uuid.uuid4())
        response_id = f"chatcmpl-{task_id[:29] if len(task_id) > 29 else task_id}"
        created = int(time.time())
        
        if stream:
            # Streaming chunk format
            delta_dict = {}
            if message_content:
                delta_dict["content"] = message_content
            if tool_calls:
                delta_dict["tool_calls"] = self._convert_tool_calls_to_openai(tool_calls)
            
            # For streaming, don't set finish_reason in chunks with content or tool_calls
            # Finish reason goes in the final empty chunk
            chunk_finish_reason = None
            if not delta_dict:
                # Empty delta means this is a role chunk or final chunk
                chunk_finish_reason = finish_reason if finish_reason != "stop" else None
            
            return {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta_dict if delta_dict else {"role": "assistant"},
                    "finish_reason": chunk_finish_reason,
                }]
            }
        else:
            # Non-streaming format
            message_dict = {
                "role": "assistant",
                "content": message_content,
            }
            if tool_calls:
                message_dict["tool_calls"] = self._convert_tool_calls_to_openai(tool_calls)
            
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason,
                }]
            }
    
    def _convert_tool_calls_to_openai(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert A2A/MCP tool calls to OpenAI format.
        
        Args:
            tool_calls: List of tool call dicts with call_id, name, arguments
            
        Returns:
            List of OpenAI tool call dicts
        """
        openai_tool_calls = []
        for idx, tc in enumerate(tool_calls):
            call_id = tc.get("call_id", "")
            name = tc.get("name", "")
            arguments = tc.get("arguments", {})
            
            if call_id and name:
                openai_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                    },
                    "index": idx,
                })
        
        return openai_tool_calls
    
    def _add_openai_endpoint(self):
        """Add OpenAI-compatible /v1/chat/completions endpoint to FastAPI app."""
        
        @self.fastapi_app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            """OpenAI-compatible chat completions endpoint that uses A2A protocol internally."""
            try:
                body = await request.json()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
            
            # Extract OpenAI request parameters
            messages = body.get("messages", [])
            model = body.get("model", self.model)
            stream = body.get("stream", False)
            
            if not messages:
                raise HTTPException(status_code=400, detail="messages is required")
            
            # Convert OpenAI messages to A2A format
            a2a_message = self._convert_openai_messages_to_a2a(messages)
            
            if stream:
                # Streaming response
                return StreamingResponse(
                    self._handle_streaming_request(a2a_message, model),
                    media_type="text/event-stream",
                )
            else:
                # Non-streaming response
                return await self._handle_non_streaming_request(a2a_message, model)
    
    async def _handle_non_streaming_request(
        self,
        a2a_message: Message,
        model: str,
    ) -> Dict[str, Any]:
        """Handle non-streaming OpenAI request via A2A protocol."""
        # Use A2A client library to send message
        async with httpx.AsyncClient() as httpx_client:
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=self.agent_card,
            )
            
            # Create A2A request
            request_params = MessageSendParams(message=a2a_message)
            a2a_request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=request_params,
            )
            
            # Send message via A2A client
            try:
                response = await client.send_message(a2a_request)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error sending message to A2A server: {str(e)}")
            
            # Extract task from response (same as test client)
            # Use model_dump() to get dict representation, then access 'result'
            response_dict = response.model_dump(mode='json', exclude_none=True)
            task = response_dict.get("result")
            
            if not task:
                raise HTTPException(status_code=500, detail="Task not found after processing: response.result is None or empty")
            
            # Convert to OpenAI format
            try:
                return self._convert_a2a_task_to_openai(task, model, stream=False)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error converting task to OpenAI format: {str(e)}")
    
    async def _handle_streaming_request(
        self,
        a2a_message: Message,
        model: str,
    ) -> AsyncGenerator[str, None]:
        """Handle streaming OpenAI request via A2A protocol."""
        # Use A2A client library to send streaming message
        async with httpx.AsyncClient() as httpx_client:
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=self.agent_card,
            )
            
            # Create A2A streaming request
            request_params = MessageSendParams(message=a2a_message)
            a2a_request = SendStreamingMessageRequest(
                id=str(uuid.uuid4()),
                params=request_params,
            )
            
            # Send streaming message via A2A client
            accumulated_content = ""
            events_received = False
            
            try:
                async for event in client.send_message_streaming(a2a_request):
                    events_received = True
                    # Extract task/status from event (same as test client)
                    # Use model_dump() to get dict representation
                    event_dict = event.model_dump(mode='json', exclude_none=True)
                    result = event_dict.get("result")
                    
                    if not result:
                        continue
                    
                    # Check if this is a status update
                    if isinstance(result, dict) and result.get("kind") == "status-update":
                        status_dict = result.get("status", {})
                        
                        # Extract incremental content and tool calls from status message
                        delta_content = None
                        tool_calls = []
                        message_dict = status_dict.get("message", {})
                        if message_dict:
                            parts = message_dict.get("parts", [])
                            for part in parts:
                                part_kind = part.get("kind")
                                if part_kind == "text":
                                    text = part.get("text", "")
                                    # Calculate delta (new content since last update)
                                    if len(text) > len(accumulated_content):
                                        delta_content = text[len(accumulated_content):]
                                        accumulated_content = text
                                elif part_kind == "data":
                                    data = part.get("data", {})
                                    if isinstance(data, dict):
                                        calls = data.get(TOOL_CALLS_KEY, [])
                                        if isinstance(calls, list):
                                            tool_calls = calls
                        
                        # Get task_id from result for response ID
                        task_id = result.get("taskId") or result.get("id") or str(uuid.uuid4())
                        
                        # Create a task-like dict for conversion
                        task_dict = {
                            "id": task_id,
                            "status": status_dict
                        }
                        
                        # Convert to OpenAI streaming chunk
                        chunk = self._convert_a2a_task_to_openai(
                            task_dict,
                            model,
                            stream=True,
                            delta_content=delta_content,
                        )
                        
                        # Format as SSE
                        yield f"data: {json.dumps(chunk)}\n\n"
                        
                        # Check if task is completed or input-required (with tool calls)
                        state_value = status_dict.get("state")
                        if state_value == "completed":
                            # Send final chunk with finish_reason
                            final_chunk = {
                                "id": chunk["id"],
                                "object": "chat.completion.chunk",
                                "created": chunk["created"],
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }]
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                        elif state_value == "input-required" and tool_calls:
                            # Tool calls are already in the chunk - return them to client
                            # Client will execute tools and send results back
                            # Send final chunk with tool_calls finish_reason
                            final_chunk = {
                                "id": chunk["id"],
                                "object": "chat.completion.chunk",
                                "created": chunk["created"],
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "tool_calls",
                                }]
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break
            except Exception as e:
                # If there's an error, yield an error chunk
                error_chunk = {
                    "id": f"chatcmpl-error",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"Error: {str(e)}"},
                        "finish_reason": "stop",
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                raise
            
            # If we didn't receive any events, yield an empty response
            if not events_received:
                empty_chunk = {
                    "id": f"chatcmpl-empty",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }]
                }
                yield f"data: {json.dumps(empty_chunk)}\n\n"
                yield "data: [DONE]\n\n"