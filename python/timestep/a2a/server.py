"""A2A server setup for Timestep agent."""

from typing import Any

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from .agent_executor import TimestepAgentExecutor
from .message_converter import openai_to_a2a
from .postgres_task_store import PostgresTaskStore
from .postgres_agent_store import PostgresAgentStore, Agent


def _create_skills_from_tools(tool_names: list[str]) -> list[AgentSkill]:
    """Create AgentSkill objects from available tools."""
    skills = []
    
    all_skills = {
        "get_weather": AgentSkill(
            id="get_weather",
            name="Get Weather",
            description="Returns weather info for the specified city.",
            tags=["weather", "city"],
            examples=[
                "What's the weather in Oakland?",
                "Get weather for San Francisco",
                "Weather in New York",
            ],
        ),
        "web_search": AgentSkill(
            id="web_search",
            name="Web Search",
            description="A tool that lets the LLM search the web using Firecrawl.",
            tags=["search", "web", "information"],
            examples=[
                "Search for Python tutorials",
                "Find information about machine learning",
                "Look up the latest news about AI",
            ],
        ),
        "handoff": AgentSkill(
            id="handoff",
            name="Handoff",
            description="Hand off a message to another agent via A2A protocol.",
            tags=["agent", "handoff"],
            examples=[
                "Hand off to weather assistant",
                "Transfer to another agent",
            ],
        ),
    }
    
    for tool_name in tool_names:
        if tool_name in all_skills:
            skills.append(all_skills[tool_name])
    
    return skills


def create_agent_card(agent: Agent, url: str = "http://localhost:8080/") -> AgentCard:
    """Create the AgentCard for a specific agent.

    Args:
        agent: Agent configuration.
        url: Base URL for the agent server.

    Returns:
        AgentCard instance.
    """
    skills = _create_skills_from_tools(agent.tools)

    card = AgentCard(
        name=agent.name,
        description=agent.description or "Timestep agent with tool support.",
        url=url,
        version="2026.0.5",
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        capabilities=AgentCapabilities(
            streaming=True,
        ),
        skills=skills,
    )
    # Set examples if the field exists (may be optional in some SDK versions)
    if hasattr(card, "examples"):
        card.examples = ["What's the weather in Oakland?", "Search for Python tutorials", "What's 2+2?"]
    return card


def create_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    tools: list[type] | None = None,
    model: str = "gpt-4.1",
) -> Any:
    """Create and configure the A2A server with agent-scoped routes.

    Args:
        host: Host address to bind to.
        port: Port number to listen on.
        tools: List of tool classes to use (deprecated, agents loaded from DB).
        model: OpenAI model name to use (deprecated, models loaded from DB).

    Returns:
        Configured Starlette application.
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    
    agent_store = PostgresAgentStore()
    agent_handlers: dict[str, Starlette] = {}
    
    async def get_agent_handler(agent_id: str) -> Starlette | None:
        """Get or create agent-specific handler."""
        if agent_id in agent_handlers:
            return agent_handlers[agent_id]
        
        agent = await agent_store.get_agent(agent_id)
        if not agent:
            return None
        
        # Build tools from agent configuration
        # Import tools here to avoid circular imports
        from ..core import GetWeather, WebSearch, Handoff
        
        tool_map = {
            "get_weather": GetWeather,
            "web_search": WebSearch,
            "handoff": Handoff,
        }
        
        agent_tools = [
            tool_map[tool_name]
            for tool_name in agent.tools
            if tool_name in tool_map
        ] if agent.tools else None
        
        url = f"http://{host}:{port}/agents/{agent.id}/"
        agent_card = create_agent_card(agent, url)
        
        # Create agent-specific task store with agent_id
        agent_task_store = PostgresTaskStore(agent_id=agent.id)
        
        agent_executor = TimestepAgentExecutor(
            tools=agent_tools, model=agent.model, task_store=agent_task_store
        )
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=agent_task_store,
        )
        
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
        
        agent_app = server.build()
        agent_handlers[agent_id] = agent_app
        return agent_app
    
    app = Starlette()
    
    # Agent-agnostic context endpoints (must be before /agents/:agentId routes)
    # GET /contexts - List all contexts, optionally filtered by parent_id
    async def list_contexts_global(request: Any) -> JSONResponse:
        """List all contexts, optionally filtered by parent_id."""
        try:
            parent_id = request.query_params.get("parent_id")
            task_store = PostgresTaskStore()
            contexts = await task_store.list_contexts(parent_id)
            return JSONResponse([{
                "id": c["id"],
                "created_at": c["created_at"],
                "updated_at": c["updated_at"],
                "metadata": c.get("metadata"),
                "parent_context_id": c.get("parent_context_id"),
            } for c in contexts])
        except Exception as e:
            print(f"Error listing contexts: {e}")
            return JSONResponse({"error": "Failed to list contexts"}, status_code=500)
    
    # GET /contexts/:contextId - Get a single context
    async def get_context_global(request: Any) -> JSONResponse:
        """Get a single context by ID."""
        try:
            context_id = request.path_params["contextId"]
            task_store = PostgresTaskStore()
            context = await task_store.get_context(context_id)
            if not context:
                return JSONResponse({"error": "Context not found"}, status_code=404)
            return JSONResponse({
                "id": context["id"],
                "created_at": context["created_at"],
                "updated_at": context["updated_at"],
                "metadata": context.get("metadata"),
                "parent_context_id": context.get("parent_context_id"),
            })
        except Exception as e:
            print(f"Error getting context: {e}")
            return JSONResponse({"error": "Failed to get context"}, status_code=500)
    
    # PATCH /contexts/:contextId - Update context (e.g., set parent_context_id)
    async def update_context_global(request: Any) -> JSONResponse:
        """Update context fields."""
        try:
            context_id = request.path_params["contextId"]
            body = await request.json()
            updates = body if isinstance(body, dict) else {}
            task_store = PostgresTaskStore()
            context = await task_store.update_context(context_id, updates)
            return JSONResponse({
                "id": context["id"],
                "created_at": context["created_at"],
                "updated_at": context["updated_at"],
                "metadata": context.get("metadata"),
                "parent_context_id": context.get("parent_context_id"),
            })
        except ValueError as e:
            if "not found" in str(e).lower():
                return JSONResponse({"error": str(e)}, status_code=404)
            return JSONResponse({"error": str(e)}, status_code=500)
        except Exception as e:
            print(f"Error updating context: {e}")
            return JSONResponse({"error": "Failed to update context"}, status_code=500)
    
    app.add_route("/contexts", list_contexts_global, methods=["GET"])
    app.add_route("/contexts/{contextId}", get_context_global, methods=["GET"])
    app.add_route("/contexts/{contextId}", update_context_global, methods=["PATCH"])
    
    # GET /contexts/:contextId/messages - Get messages for a context (agent-agnostic)
    async def get_context_messages_global(request: Any) -> JSONResponse:
        """Get all messages for a context (agent-agnostic)."""
        try:
            context_id = request.path_params["contextId"]
            task_store = PostgresTaskStore()  # No agent_id - agent-agnostic
            
            # Get OpenAI messages for the context
            try:
                openai_messages = await task_store.get_openai_messages_by_context_id(context_id)
            except Exception as e:
                print(f"Error loading OpenAI messages, falling back to task history: {e}")
                openai_messages = []
            
            if len(openai_messages) == 0:
                # Fallback to task history if no OpenAI messages stored yet
                tasks = await task_store.load_by_context_id(context_id)
                messages = []
                for task in tasks:
                    if task.history:
                        for msg in task.history:
                            text_parts = "".join(
                                p.text
                                for p in msg.parts
                                if hasattr(p, "text") and p.text
                            )
                            if text_parts:
                                messages.append(
                                    {
                                        "role": msg.role,
                                        "content": text_parts,
                                        "timestamp": msg.timestamp.isoformat()
                                        if hasattr(msg, "timestamp") and msg.timestamp
                                        else None,
                                        "taskId": str(task.id),
                                    }
                                )
                
                messages.sort(key=lambda m: m["timestamp"] or "")
                return JSONResponse({"contextId": context_id, "messages": messages})
            
            # Convert OpenAI messages to A2A format
            messages = []
            for msg in openai_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text from content array
                    text_parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    content = "".join(text_parts)
                elif not isinstance(content, str):
                    content = str(content)
                
                messages.append({
                    "role": role,
                    "content": content,
                })
            
            return JSONResponse({"contextId": context_id, "messages": messages})
        except Exception as e:
            print(f"Error loading context messages: {e}")
            return JSONResponse({"error": "Failed to load context messages"}, status_code=500)
    
    app.add_route("/contexts/{contextId}/messages", get_context_messages_global, methods=["GET"])
    
    # GET /agents - List all agents
    async def list_agents(request: Any) -> JSONResponse:
        """List all agents."""
        try:
            agents = await agent_store.list_agents()
            return JSONResponse([{
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "tools": a.tools,
                "model": a.model,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "updated_at": a.updated_at.isoformat() if a.updated_at else None,
            } for a in agents])
        except Exception as e:
            print(f"Error listing agents: {e}")
            return JSONResponse({"error": "Failed to list agents"}, status_code=500)
    
    app.add_route("/agents", list_agents, methods=["GET"])
    
    # GET /agents/:agentId/.well-known/agent-card.json
    async def get_agent_card(request: Any) -> JSONResponse:
        """Get agent card."""
        try:
            agent_id = request.path_params["agentId"]
            agent = await agent_store.get_agent(agent_id)
            if not agent:
                return JSONResponse({"error": f"Agent not found: {agent_id}"}, status_code=404)
            
            url = f"http://{host}:{port}/agents/{agent.id}/"
            agent_card = create_agent_card(agent, url)
            return JSONResponse(agent_card.model_dump(mode="json") if hasattr(agent_card, "model_dump") else agent_card.dict())
        except Exception as e:
            print(f"Error creating agent card: {e}")
            return JSONResponse({"error": "Failed to create agent card"}, status_code=500)
    
    app.add_route("/agents/{agentId}/.well-known/agent-card.json", get_agent_card, methods=["GET"])
    
    # Agent-scoped A2A routes middleware
    class AgentMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            # Skip middleware for context endpoints and agent card
            if path.startswith("/contexts") or path.endswith("/.well-known/agent-card.json"):
                return await call_next(request)
            if path.startswith("/agents/") and "/.well-known/agent-card.json" not in path:
                # Extract agentId from path
                parts = path.split("/")
                if len(parts) >= 3 and parts[1] == "agents":
                    agent_id = parts[2]
                    agent_app = await get_agent_handler(agent_id)
                    if not agent_app:
                        return JSONResponse({"error": f"Agent not found: {agent_id}"}, status_code=404)
                    
                    # Modify path to remove /agents/:agentId prefix
                    new_path = "/" + "/".join(parts[3:]) if len(parts) > 3 else "/"
                    request.scope["path"] = new_path
                    request.scope["raw_path"] = new_path.encode()
                    
                    # Handle with agent app
                    return await agent_app(request.scope, request.receive, request._send)
            
            return await call_next(request)
    
    app.add_middleware(AgentMiddleware)
    
    # Agent-scoped context management endpoints
    async def list_contexts(request: Any) -> JSONResponse:
        """List all contexts for an agent."""
        try:
            agent_id = request.path_params["agentId"]
            agent_task_store = PostgresTaskStore(agent_id=agent_id)
            contexts = await agent_task_store.list_contexts()
            return JSONResponse([{
                "id": c["id"],
                "created_at": c["created_at"].isoformat() if isinstance(c["created_at"], type(c["created_at"])) else str(c["created_at"]),
                "updated_at": c["updated_at"].isoformat() if isinstance(c["updated_at"], type(c["updated_at"])) else str(c["updated_at"]),
                "metadata": c.get("metadata"),
                "parent_context_id": c.get("parent_context_id"),
            } for c in contexts])
        except Exception as e:
            print(f"Error listing contexts: {e}")
            return JSONResponse({"error": "Failed to list contexts"}, status_code=500)
    
    async def create_context(request: Any) -> JSONResponse:
        """Create a new context for an agent."""
        try:
            agent_id = request.path_params["agentId"]
            body = await request.json()
            metadata = body.get("metadata") if body else None
            agent_task_store = PostgresTaskStore(agent_id=agent_id)
            context = await agent_task_store.create_context(metadata)
            return JSONResponse({
                "id": context["id"],
                "created_at": context["created_at"].isoformat() if hasattr(context["created_at"], "isoformat") else str(context["created_at"]),
                "updated_at": context["updated_at"].isoformat() if hasattr(context["updated_at"], "isoformat") else str(context["updated_at"]),
                "metadata": context.get("metadata"),
            })
        except Exception as e:
            print(f"Error creating context: {e}")
            return JSONResponse({"error": "Failed to create context"}, status_code=500)
    
    async def delete_context(request: Any) -> JSONResponse:
        """Delete a context and all its messages."""
        try:
            context_id = request.path_params["contextId"]
            agent_id = request.path_params["agentId"]
            agent_task_store = PostgresTaskStore(agent_id=agent_id)
            await agent_task_store.delete_context(context_id)
            return JSONResponse({"success": True, "contextId": context_id})
        except Exception as e:
            print(f"Error deleting context: {e}")
            return JSONResponse({"error": "Failed to delete context"}, status_code=500)
    
    async def get_context_messages(request: Any) -> JSONResponse:
        """Get all messages for a context."""
        try:
            context_id = request.path_params["contextId"]
            agent_id = request.path_params["agentId"]
            agent_task_store = PostgresTaskStore(agent_id=agent_id)
            
            # Get OpenAI messages for the context
            try:
                openai_messages = await agent_task_store.get_openai_messages_by_context_id(
                    context_id
                )
            except Exception as e:
                print(f"Error loading OpenAI messages, falling back to task history: {e}")
                openai_messages = []
            
            if len(openai_messages) == 0:
                # Fallback to task history if no OpenAI messages stored yet
                tasks = await agent_task_store.load_by_context_id(context_id)
                messages = []
                for task in tasks:
                    if task.history:
                        for msg in task.history:
                            text_parts = "".join(
                                p.text
                                for p in msg.parts
                                if hasattr(p, "text") and p.text
                            )
                            if text_parts:
                                messages.append(
                                    {
                                        "role": msg.role,
                                        "content": text_parts,
                                        "timestamp": msg.timestamp.isoformat()
                                        if hasattr(msg, "timestamp") and msg.timestamp
                                        else None,
                                        "taskId": str(task.id),
                                    }
                                )
                
                messages.sort(key=lambda m: m["timestamp"] or "")
                return JSONResponse({"contextId": context_id, "messages": messages})
            
            # Convert OpenAI messages to A2A format
            messages = []
            for msg in openai_messages:
                role = msg.get("role", "user")
                if role == "system":
                    continue
                
                content = msg.get("content")
                if isinstance(content, str):
                    text_content = content
                elif isinstance(content, list):
                    text_content = "".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                else:
                    text_content = ""
                
                if text_content:
                    messages.append({
                        "role": role,
                        "content": text_content,
                        "taskId": msg.get("taskId"),
                    })
            
            return JSONResponse({"contextId": context_id, "messages": messages})
        except Exception as e:
            print(f"Error loading context messages: {e}")
            return JSONResponse({"error": "Failed to load context messages"}, status_code=500)
    
    # Add routes
    app.routes.extend([
        Route("/agents", list_agents, methods=["GET"]),
        Route("/agents/{agentId:str}/.well-known/agent-card.json", get_agent_card, methods=["GET"]),
        Route("/agents/{agentId:str}/contexts", list_contexts, methods=["GET"]),
        Route("/agents/{agentId:str}/contexts", create_context, methods=["POST"]),
        Route("/agents/{agentId:str}/contexts/{contextId:str}", delete_context, methods=["DELETE"]),
        Route("/agents/{agentId:str}/contexts/{contextId:str}/messages", get_context_messages, methods=["GET"]),
    ])
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    tools: list[type] | None = None,
    model: str = "gpt-4.1",
) -> None:
    """Run the A2A server.

    Args:
        host: Host address to bind to.
        port: Port number to listen on.
        tools: List of tool classes to use. If None, uses default tools.
        model: OpenAI model name to use.
    """
    app = create_server(host=host, port=port, tools=tools, model=model)
    uvicorn.run(app, host=host, port=port)

