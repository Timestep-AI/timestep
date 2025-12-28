"""A2A server for Timestep agents."""

import json
import os
from typing import Any, Dict
from uuid import UUID

import uvicorn
from a2a.server.tasks import DatabaseTaskStore
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from timestep.services.agent_service import AgentService
from timestep.stores.agent_store import AgentStore
from timestep.utils.exceptions import AgentConfigError


# REST API endpoints

async def create_agent(request: Request) -> Response:
    """Create a new agent. POST /agents"""
    agent_service: AgentService = request.app.state.agent_service
    
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    try:
        agent = await agent_service.create_agent(body)
        return JSONResponse(agent, status_code=201)
    except AgentConfigError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Failed to create agent: {str(e)}"}, status_code=500)


async def list_agents(request: Request) -> Response:
    """List all agents. GET /agents"""
    agent_service: AgentService = request.app.state.agent_service
    
    try:
        agents = await agent_service.list_agents()
        return JSONResponse(agents)
    except Exception as e:
        return JSONResponse({"error": f"Failed to list agents: {str(e)}"}, status_code=500)


async def get_agent(request: Request) -> Response:
    """Get agent by ID. GET /agents/{agent_id}"""
    agent_service: AgentService = request.app.state.agent_service
    
    try:
        agent_id = UUID(request.path_params["agent_id"])
    except ValueError:
        return JSONResponse({"error": "Invalid agent ID format"}, status_code=400)
    
    try:
        agent = await agent_service.get_agent(agent_id)
        if agent is None:
            return JSONResponse({"error": "Agent not found"}, status_code=404)
        return JSONResponse(agent)
    except Exception as e:
        return JSONResponse({"error": f"Failed to get agent: {str(e)}"}, status_code=500)


async def update_agent(request: Request) -> Response:
    """Update an agent. PUT /agents/{agent_id}"""
    agent_service: AgentService = request.app.state.agent_service
    
    try:
        agent_id = UUID(request.path_params["agent_id"])
    except ValueError:
        return JSONResponse({"error": "Invalid agent ID format"}, status_code=400)
    
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    try:
        agent = await agent_service.update_agent(agent_id, body)
        if agent is None:
            return JSONResponse({"error": "Agent not found"}, status_code=404)
        return JSONResponse(agent)
    except AgentConfigError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Failed to update agent: {str(e)}"}, status_code=500)


async def delete_agent(request: Request) -> Response:
    """Delete an agent. DELETE /agents/{agent_id}"""
    agent_service: AgentService = request.app.state.agent_service
    
    try:
        agent_id = UUID(request.path_params["agent_id"])
    except ValueError:
        return JSONResponse({"error": "Invalid agent ID format"}, status_code=400)
    
    try:
        deleted = await agent_service.delete_agent(agent_id)
        if not deleted:
            return JSONResponse({"error": "Agent not found"}, status_code=404)
        return Response(status_code=204)
    except Exception as e:
        return JSONResponse({"error": f"Failed to delete agent: {str(e)}"}, status_code=500)


# A2A endpoints

async def get_agent_card(request: Request) -> Response:
    """Get agent's public card. GET /agents/{agent_id}/.well-known/agent-card.json"""
    agent_service: AgentService = request.app.state.agent_service
    
    try:
        agent_id = UUID(request.path_params["agent_id"])
    except ValueError:
        return JSONResponse({"error": "Invalid agent ID format"}, status_code=400)
    
    try:
        base_url = str(request.base_url).rstrip("/")
        card = await agent_service.create_agent_card(agent_id, base_url, extended=False)
        return JSONResponse(card.model_dump(mode="json", exclude_none=True))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Failed to get agent card: {str(e)}"}, status_code=500)


async def get_extended_agent_card(request: Request) -> Response:
    """Get agent's extended card. GET /agents/{agent_id}/agent/authenticatedExtendedCard"""
    agent_service: AgentService = request.app.state.agent_service
    
    try:
        agent_id = UUID(request.path_params["agent_id"])
    except ValueError:
        return JSONResponse({"error": "Invalid agent ID format"}, status_code=400)
    
    try:
        base_url = str(request.base_url).rstrip("/")
        card = await agent_service.create_agent_card(agent_id, base_url, extended=True)
        return JSONResponse(card.model_dump(mode="json", exclude_none=True))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Failed to get extended agent card: {str(e)}"}, status_code=500)


async def a2a_message_handler(request: Request) -> Response:
    """Handle A2A JSON-RPC messages. POST /agents/{agent_id}/"""
    agent_service: AgentService = request.app.state.agent_service
    task_store: DatabaseTaskStore = request.app.state.task_store
    
    try:
        agent_id = UUID(request.path_params["agent_id"])
    except ValueError:
        return JSONResponse({"error": "Invalid agent ID format"}, status_code=400)
    
    try:
        base_url = str(request.base_url).rstrip("/")
        a2a_app = await agent_service.get_a2a_app(agent_id, task_store, base_url)
        
        # Adjust path to remove /agents/{agent_id} prefix for A2A app
        agent_id_str = str(agent_id)
        scope = dict(request.scope)
        original_path = scope["path"]
        new_path = original_path.replace(f"/agents/{agent_id_str}", "", 1) or "/"
        scope["path"] = new_path
        scope["raw_path"] = new_path.encode()
        
        return await a2a_app(scope, request.receive, request._send)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Failed to handle A2A message: {str(e)}"}, status_code=500)


if __name__ == '__main__':
    # Require DATABASE_URL for PostgreSQL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")
    
    # Convert postgresql:// to postgresql+asyncpg:// for async SQLAlchemy
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    # Create database engine
    engine = create_async_engine(database_url)
    
    # Create agent store
    agent_store = AgentStore(engine)
    
    # Create agent service
    agent_service = AgentService(agent_store)
    
    # Create task store
    task_store = DatabaseTaskStore(engine=engine)
    
    # Create main app with all routes
    app = Starlette(routes=[
        # REST API routes
        Route("/agents", create_agent, methods=["POST"]),
        Route("/agents", list_agents, methods=["GET"]),
        Route("/agents/{agent_id}", get_agent, methods=["GET"]),
        Route("/agents/{agent_id}", update_agent, methods=["PUT"]),
        Route("/agents/{agent_id}", delete_agent, methods=["DELETE"]),
        # A2A routes
        Route("/agents/{agent_id}/.well-known/agent-card.json", get_agent_card, methods=["GET"]),
        Route("/agents/{agent_id}/agent/authenticatedExtendedCard", get_extended_agent_card, methods=["GET"]),
        Route("/agents/{agent_id}/", a2a_message_handler, methods=["POST"]),
    ])
    
    # Store state
    app.state.agent_service = agent_service
    app.state.task_store = task_store
    
    # Add CORS middleware
    app.user_middleware.insert(0, (
        CORSMiddleware,
        (),
        {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    ))
    app.middleware_stack = app.build_middleware_stack()
    
    # Initialize database tables
    import asyncio
    asyncio.run(agent_store.create_tables())
    
    # Get server config from environment
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", 9999))
    
    uvicorn.run(app, host=host, port=port)
