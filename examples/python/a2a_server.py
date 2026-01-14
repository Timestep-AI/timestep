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
import datetime
from pathlib import Path
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
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, Message, TaskStatusUpdateEvent, TaskStatus, TaskState, Role, TransportProtocol, Task, Part, DataPart
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
                    "description": "The A2A URI of the agent to hand off to"
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

# Agent descriptions
AGENT_DESCRIPTIONS: Dict[str, str] = {
    PERSONAL_ASSISTANT_ID: "Personal Assistant",
    WEATHER_ASSISTANT_ID: "Weather Assistant",
}


def build_system_message(agent_id: str, tools: List[Dict[str, Any]]) -> str:
    """Build system message explaining who the agent is and what tools are available."""
    agent_name = AGENT_DESCRIPTIONS.get(agent_id, "Assistant")
    
    # Get base URL for agent endpoints
    base_url = os.getenv("A2A_BASE_URL", "http://localhost:8000")
    
    system_parts = [f"You are a {agent_name}."]
    
    if tools:
        system_parts.append("\nYou have access to the following tools:")
        for tool in tools:
            func = tool.get("function", {})
            tool_name = func.get("name", "unknown")
            tool_desc = func.get("description", "No description available")
            params = func.get("parameters", {})
            
            # Build tool description with format
            tool_info = f"- {tool_name}: {tool_desc}"
            
            # Add parameter details
            if params and "properties" in params:
                tool_info += "\n  Parameters:"
                properties = params.get("properties", {})
                required = params.get("required", [])
                for param_name, param_spec in properties.items():
                    param_type = param_spec.get("type", "string")
                    param_desc = param_spec.get("description", "")
                    required_marker = " (required)" if param_name in required else " (optional)"
                    tool_info += f"\n    - {param_name} ({param_type}){required_marker}: {param_desc}"
            
            # Special handling for handoff tool - list available agent URIs from agent cards
            if tool_name == "handoff":
                tool_info += "\n  Available agents for handoff:"
                # List all other agents (excluding current agent) with their A2A endpoint URLs
                for other_agent_id, other_agent_name in AGENT_DESCRIPTIONS.items():
                    if other_agent_id != agent_id:
                        # Use the standard A2A agent endpoint URL (from agent card)
                        agent_uri = f"{base_url}/agents/{other_agent_id}"
                        tool_info += f"\n    - {other_agent_name} (ID: {other_agent_id[:8]}...): Use agent_uri=\"{agent_uri}\""
            
            system_parts.append(tool_info)
    else:
        system_parts.append("\nYou do not have access to any tools.")
    
    return "\n".join(system_parts)


# Simple in-memory task storage (messages per task, keyed by agent_id:task_id)
task_messages: Dict[str, List[Dict[str, Any]]] = {}

# Track all task IDs per agent for listing
agent_task_ids: Dict[str, List[str]] = {}

def write_trace(task_id: str, agent_id: str, input_messages: List[Dict], input_tools: List[Dict], output_message: Dict) -> None:
    """Write trace to traces/ folder."""
    traces_dir = Path("/workspace/traces")
    traces_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().isoformat().replace(":", "-")
    # Use short task_id for filename (first 8 chars)
    task_id_short = task_id[:8] if task_id else "unknown"
    agent_id_short = agent_id[:8] if agent_id else "unknown"
    trace_file = traces_dir / f"{timestamp}_{task_id_short}_{agent_id_short}.json"
    
    trace = {
        "task_id": task_id,
        "agent_id": agent_id,
        "timestamp": timestamp,
        "input": {
            "messages": input_messages,
            "tools": input_tools,
        },
        "output": {
            "content": output_message.get("content", ""),
            "tool_calls": output_message.get("tool_calls", []),
        }
    }
    
    with open(trace_file, "w") as f:
        json.dump(trace, f, indent=2)


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
            # Track task ID for this agent
            if self.agent_id not in agent_task_ids:
                agent_task_ids[self.agent_id] = []
            if task_id and task_id not in agent_task_ids[self.agent_id]:
                agent_task_ids[self.agent_id].append(task_id)
        
        messages = task_messages[task_key]
        
        # Extract text from incoming message for OpenAI processing
        if context.message:
            msg = context.message
            text_content = ""
            
            # Extract from parts (preferred)
            if msg.parts:
                for part in msg.parts:
                    # Handle Part wrapper with root attribute
                    if hasattr(part, 'root'):
                        part_data = part.root
                        if hasattr(part_data, 'kind') and part_data.kind == 'text' and hasattr(part_data, 'text'):
                            text_content += part_data.text
                    # Handle direct TextPart access
                    elif hasattr(part, 'kind') and part.kind == 'text' and hasattr(part, 'text'):
                        text_content += part.text
            
            # Fallback to content attribute
            if not text_content and hasattr(msg, 'content'):
                if isinstance(msg.content, str):
                    text_content = msg.content
            
            if text_content:
                messages.append({"role": "user", "content": text_content})
        
        # Convert messages to OpenAI format
        openai_messages = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Check if this is a tool result (comes after assistant message with tool_calls)
            if i > 0 and messages[i-1].get("role") == "assistant":
                prev_tool_calls = messages[i-1].get("tool_calls", [])
                if prev_tool_calls:
                    # Format as tool message for OpenAI
                    tool_call_id = prev_tool_calls[0].get("id")
                    if tool_call_id:
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content,
                        })
                        continue
            
            openai_msg = {"role": role, "content": content}
            if role == "assistant" and "tool_calls" in msg:
                openai_msg["tool_calls"] = msg["tool_calls"]
            openai_messages.append(openai_msg)
        
        # Ensure we have at least one message before calling OpenAI
        if not openai_messages:
            raise ValueError(f"No messages to send to OpenAI. Task: {task_id}, Agent: {self.agent_id}, Messages count: {len(messages)}, Context message: {context.message}")
        
        # Build system message explaining agent identity and available tools
        system_message_content = build_system_message(self.agent_id, self.tools or [])
        
        # Add system message at the beginning
        openai_messages_with_system = [
            {"role": "system", "content": system_message_content}
        ] + openai_messages
        
        # Call OpenAI with agent-specific tools
        try:
            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages_with_system,
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
            
            # Capture trace: input messages + output message
            output_message_dict = {
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
            }
            write_trace(
                task_id=task_id or "",
                agent_id=self.agent_id,
                input_messages=openai_messages_with_system,
                input_tools=self.tools or [],
                output_message=output_message_dict,
            )
            
            # Build A2A message using helper function
            # Role.agent is the correct role for assistant messages in A2A
            a2a_message = create_text_message_object(
                role=Role.agent,
                content=assistant_content,
            )
            a2a_message.context_id = context_id
            a2a_message.task_id = task_id
            
            # Add tool calls as a DataPart in the message parts (per A2A spec)
            if tool_calls:
                tool_calls_data = {
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
                    ]
                }
                # Add DataPart with tool calls to message parts
                a2a_message.parts.append(Part(DataPart(data=tool_calls_data)))
            
            # If there are tool calls, publish them and update status
            if tool_calls:
                # Update status to input-required with assistant message (per A2A spec 4.1.2)
                status_update = TaskStatusUpdateEvent(
                    task_id=task_id or "",
                    context_id=context_id or "",
                    status=TaskStatus(
                        state=TaskState.input_required,
                        message=a2a_message,  # Include assistant message with tool calls
                    ),
                    final=False,
                )
                await event_queue.enqueue_event(status_update)
            else:
                # No tool calls, task is complete
                status_update = TaskStatusUpdateEvent(
                    task_id=task_id or "",
                    context_id=context_id or "",
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=a2a_message,  # Include final assistant message
                    ),
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

# Mount the A2A apps directly under /agents/{agent_id}/
# This allows the A2A SDK to handle all routing including streaming
app.mount("/agents/00000000-0000-0000-0000-000000000000", personal_app)
app.mount("/agents/FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF", weather_app)

@app.get("/agents/{agent_id}/v1/tasks")
async def agent_list_tasks(request: Request, agent_id: str):
    """List all tasks for a specific agent."""
    if agent_id not in agent_a2a_apps:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Get task IDs for this agent
    task_ids = agent_task_ids.get(agent_id, [])
    
    # Return task IDs - client can query individual tasks if needed
    return JSONResponse({"task_ids": task_ids, "count": len(task_ids)})

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
