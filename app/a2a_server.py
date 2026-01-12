# /// script
# dependencies = [
#   "a2a-sdk[http-server]",
#   "openai",
#   "uvicorn",
# ]
# ///

"""
A2A Server using a2a-sdk with FastAPI.
Handles task creation and continuation via A2A protocol.
"""

import os
import uuid
import json
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, Request, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from openai import OpenAI
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.apps.rest.fastapi_app import A2ARESTFastAPIApplication
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, Message, TaskStatusUpdateEvent, TaskStatus, TaskState, Role, TransportProtocol
from a2a.client.helpers import create_text_message_object

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Agent IDs
PERSONAL_ASSISTANT_ID = "00000000-0000-0000-0000-000000000000"
WEATHER_ASSISTANT_ID = "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"

# Tool definitions (OpenAI function calling format)
HANDOFF_TOOL = {
    "type": "function",
    "function": {
        "name": "handoff",
        "description": "Hand off to another agent to handle a specific task",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_uri": {
                    "type": "string",
                    "description": "The URI to call for the handoff (client sampling endpoint URL)"
                },
                "context_id": {
                    "type": "string",
                    "description": "Optional context ID for the handoff"
                },
                "message": {
                    "type": "string",
                    "description": "The message to send to the other agent"
                }
            },
            "required": ["agent_uri", "message"]
        }
    }
}

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for (e.g., city name)"
                }
            },
            "required": ["location"]
        }
    }
}

# Agent registry mapping agent IDs to tool configurations
AGENT_TOOLS: Dict[str, List[Dict[str, Any]]] = {
    PERSONAL_ASSISTANT_ID: [HANDOFF_TOOL],
    WEATHER_ASSISTANT_ID: [GET_WEATHER_TOOL],
}

# Simple in-memory task storage (messages per task, keyed by agent_id:task_id)
task_messages: Dict[str, List[Dict[str, Any]]] = {}


class MultiAgentExecutor(AgentExecutor):
    """Agent executor that uses OpenAI directly and configures tools based on agent_id."""
    
    def __init__(self, agent_id: str, model: str = "gpt-4o-mini"):
        self.agent_id = agent_id
        self.model = model
        self.tools = AGENT_TOOLS.get(agent_id, [])
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute agent task using OpenAI."""
        task_id = context.task_id
        context_id = context.context_id
        
        # Get messages for this task (or initialize empty)
        # Use agent_id:task_id as key for task isolation
        task_key = f"{self.agent_id}:{task_id}"
        if task_key not in task_messages:
            task_messages[task_key] = []
        
        messages = task_messages[task_key]
        
        # Get new message from request context
        # RequestContext has a 'message' property, not 'request.message'
        if context.message:
            msg = context.message
            # Extract text from message parts
            text_content = ""
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'kind') and part.kind == 'text':
                        text_content += part.text
            elif hasattr(msg, 'content'):
                text_content = msg.content
            
            # Check if this is a tool result (JSON content that might be a tool result)
            # For now, treat all incoming messages as user messages
            # The content might be a JSON string with tool result
            messages.append({
                "role": "user",
                "content": text_content,
            })
        
        # Convert messages to OpenAI format
        openai_messages = []
        pending_tool_results = {}  # Track tool results by tool_call_id
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle A2A message format (parts)
            if isinstance(content, list):
                text_content = ""
                for part in content:
                    if isinstance(part, dict) and part.get("kind") == "text":
                        text_content += part.get("text", "")
                content = text_content
            
            # Check if this is a tool result (comes after assistant message with tool_calls)
            if i > 0 and messages[i-1].get("role") == "assistant":
                prev_tool_calls = messages[i-1].get("tool_calls", [])
                if prev_tool_calls and len(prev_tool_calls) > 0:
                    # This might be a tool result - try to match it to a tool call
                    # For simplicity, we'll send tool results as user messages
                    # OpenAI will process them in context
                    pass
            
            openai_msg = {"role": role, "content": content}
            
            # Add tool calls if present (for assistant messages)
            if role == "assistant" and "tool_calls" in msg:
                openai_msg["tool_calls"] = msg["tool_calls"]
            
            openai_messages.append(openai_msg)
        
        # Call OpenAI with agent-specific tools
        try:
            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": 0.0,
            }
            # Add tools if configured for this agent
            if self.tools:
                request_params["tools"] = self.tools
                request_params["tool_choice"] = "auto"
            
            response = openai_client.chat.completions.create(**request_params)
            
            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls or []
            
            # Convert OpenAI response to A2A format
            assistant_content = assistant_message.content or ""
            
            # Build A2A message using helper function
            # Role.agent is the correct role for assistant messages in A2A
            a2a_message = create_text_message_object(
                role=Role.agent,
                content=assistant_content,
            )
            a2a_message.context_id = context_id
            a2a_message.task_id = task_id
            
            # Publish assistant message
            await event_queue.enqueue_event(a2a_message)
            
            # If there are tool calls, publish them and update status
            if tool_calls:
                # Update status to input-required
                status_update = TaskStatusUpdateEvent(
                    task_id=task_id or "",
                    context_id=context_id or "",
                    status=TaskStatus(state=TaskState.input_required),
                    final=False,
                )
                await event_queue.enqueue_event(status_update)
            else:
                # No tool calls, task is complete
                status_update = TaskStatusUpdateEvent(
                    task_id=task_id or "",
                    context_id=context_id or "",
                    status=TaskStatus(state=TaskState.completed),
                    final=True,
                )
                await event_queue.enqueue_event(status_update)
            
            # Update task messages
            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ] if tool_calls else [],
            })
            
        except Exception as e:
            status_update = TaskStatusUpdateEvent(
                task_id=task_id or "",
                context_id=context_id or "",
                status=TaskStatus(state=TaskState.failed, message=create_text_message_object(content=str(e))),
                final=True,
            )
            await event_queue.enqueue_event(status_update)
            raise
    
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel an ongoing task."""
        task_id = context.task_id
        context_id = context.context_id
        
        # Publish canceled status
        status_update = TaskStatusUpdateEvent(
            task_id=task_id or "",
            context_id=context_id or "",
            status=TaskStatus(state=TaskState.canceled),
            final=True,
        )
        await event_queue.enqueue_event(status_update)


def create_agent_card(agent_id: str, agent_name: str, description: str) -> AgentCard:
    """Create an agent card for a specific agent."""
    base_url = os.getenv("A2A_BASE_URL", "http://localhost:8000")
    return AgentCard(
        name=agent_name,
        version="1.0.0",
        description=description,
        url=f"{base_url}/agents/{agent_id}",
        preferred_transport=TransportProtocol.http_json,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[AgentSkill(id=agent_id, name=agent_name, description=description, tags=[])],
    )

# Create base FastAPI app
app = FastAPI()

# Agent routing: map agent_id to executor and handler
agent_handlers: Dict[str, DefaultRequestHandler] = {}

def get_or_create_handler(agent_id: str) -> DefaultRequestHandler:
    """Get or create a request handler for an agent."""
    if agent_id not in agent_handlers:
        if agent_id not in AGENT_TOOLS:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        executor = MultiAgentExecutor(agent_id=agent_id)
        task_store = InMemoryTaskStore()
        handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store,
        )
        agent_handlers[agent_id] = handler
    return agent_handlers[agent_id]

# Create A2A apps for routing (we'll use these to get the routes)
personal_assistant_card = create_agent_card(
    PERSONAL_ASSISTANT_ID,
    "personal-assistant",
    "Personal Assistant agent with handoff capability"
)
weather_assistant_card = create_agent_card(
    WEATHER_ASSISTANT_ID,
    "weather-assistant",
    "Weather Assistant agent with weather lookup capability"
)

# Create routers for each agent's A2A endpoints
personal_handler = get_or_create_handler(PERSONAL_ASSISTANT_ID)
weather_handler = get_or_create_handler(WEATHER_ASSISTANT_ID)

personal_a2a_app = A2ARESTFastAPIApplication(
    agent_card=personal_assistant_card,
    http_handler=personal_handler,
)
weather_a2a_app = A2ARESTFastAPIApplication(
    agent_card=weather_assistant_card,
    http_handler=weather_handler,
)

# Build the A2A apps to get their routes
personal_app = personal_a2a_app.build()
weather_app = weather_a2a_app.build()

# Add agent card endpoints
@app.get("/agents/{agent_id}/.well-known/agent-card.json")
async def get_agent_card(agent_id: str):
    """Get agent card for a specific agent."""
    if agent_id == PERSONAL_ASSISTANT_ID:
        card = personal_assistant_card
    elif agent_id == WEATHER_ASSISTANT_ID:
        card = weather_assistant_card
    else:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return JSONResponse(card.model_dump(mode="json"))

# Create agent-specific A2A apps and mount their routes
agent_a2a_apps: Dict[str, A2ARESTFastAPIApplication] = {
    PERSONAL_ASSISTANT_ID: personal_a2a_app,
    WEATHER_ASSISTANT_ID: weather_a2a_app,
}

# Mount agent-specific routes
for agent_id, a2a_app_instance in agent_a2a_apps.items():
    # Get routes from the A2A app
    a2a_app = a2a_app_instance.build()
    
    # Mount the A2A app under /agents/{agent_id}/
    # We'll use a sub-application approach
    from fastapi import APIRouter
    agent_router = APIRouter()
    
    # Get routes from the built A2A app and add them to our router with agent prefix
    # Since we can't easily inspect FastAPI routes, we'll manually add the routes we need
    pass  # We'll handle routing manually below

# Add routes for agent-specific A2A endpoints manually
@app.post("/agents/{agent_id}/v1/message:send")
async def agent_message_send(request: Request, agent_id: str):
    """Handle message:send for a specific agent (standard A2A endpoint)."""
    if agent_id not in agent_a2a_apps:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    a2a_app_instance = agent_a2a_apps[agent_id]
    adapter = a2a_app_instance._adapter
    
    # Modify request path to remove agent prefix
    scope = dict(request.scope)
    scope["path"] = "/v1/message:send"
    scope["raw_path"] = b"/v1/message:send"
    from starlette.requests import Request as StarletteRequest
    modified_request = StarletteRequest(scope, request.receive)
    
    # Use adapter's _handle_request method (it takes method, request)
    return await adapter._handle_request(adapter.handler.on_message_send, modified_request)

@app.get("/agents/{agent_id}/v1/tasks/{task_id}")
async def agent_get_task(request: Request, agent_id: str, task_id: str):
    """Handle get task for a specific agent (standard A2A endpoint)."""
    if agent_id not in agent_a2a_apps:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    a2a_app_instance = agent_a2a_apps[agent_id]
    adapter = a2a_app_instance._adapter
    
    # Modify request path
    scope = dict(request.scope)
    scope["path"] = f"/v1/tasks/{task_id}"
    scope["raw_path"] = f"/v1/tasks/{task_id}".encode()
    from starlette.requests import Request as StarletteRequest
    modified_request = StarletteRequest(scope, request.receive)
    
    return await adapter._handle_request(adapter.handler.on_get_task, modified_request)

@app.post("/agents/{agent_id}/v1/tasks/{task_id}:cancel")
async def agent_cancel_task(request: Request, agent_id: str, task_id: str):
    """Handle cancel task for a specific agent (standard A2A endpoint)."""
    if agent_id not in agent_a2a_apps:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    a2a_app_instance = agent_a2a_apps[agent_id]
    adapter = a2a_app_instance._adapter
    
    # Modify request path
    scope = dict(request.scope)
    scope["path"] = f"/v1/tasks/{task_id}:cancel"
    scope["raw_path"] = f"/v1/tasks/{task_id}:cancel".encode()
    from starlette.requests import Request as StarletteRequest
    modified_request = StarletteRequest(scope, request.receive)
    
    return await adapter._handle_request(adapter.handler.on_cancel_task, modified_request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("A2A_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
