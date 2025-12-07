"""Agent store for saving and loading Agent configurations from database."""

import json
import uuid
from typing import Optional, Any, Callable
from .db_connection import DatabaseConnection
from ._vendored_imports import (
    Agent, ModelSettings, Tool, InputGuardrail, OutputGuardrail, Handoff
)
from .tool_registry import register_tool, get_tool
from .guardrail_registry import (
    register_input_guardrail, register_output_guardrail,
    get_input_guardrail, get_output_guardrail
)


async def save_agent(agent: Agent, db: DatabaseConnection) -> str:
    """
    Save an agent configuration to the database.
    
    Args:
        agent: The Agent object to save
        db: DatabaseConnection instance
        
    Returns:
        The agent_id (UUID as string)
    """
    # Extract serializable fields from agent
    instructions = None
    instructions_fn = None
    if isinstance(agent.instructions, str):
        instructions = agent.instructions
    elif callable(agent.instructions):
        instructions_fn = getattr(agent.instructions, '__name__', str(id(agent.instructions)))
    
    prompt = None
    prompt_fn = None
    if agent.prompt is not None:
        if callable(agent.prompt):
            prompt_fn = getattr(agent.prompt, '__name__', str(id(agent.prompt)))
        else:
            # Serialize prompt object
            if hasattr(agent.prompt, 'model_dump'):
                prompt = agent.prompt.model_dump(mode='json')
            elif hasattr(agent.prompt, '__dict__'):
                prompt = agent.prompt.__dict__
            else:
                prompt = json.loads(json.dumps(agent.prompt, default=str))
    
    # Serialize model_settings
    model_settings_json = agent.model_settings.to_json_dict() if hasattr(agent.model_settings, 'to_json_dict') else {}
    
    # Serialize output_type
    output_type_json = None
    if agent.output_type is not None:
        if hasattr(agent.output_type, '__name__'):
            output_type_json = {'type': 'class', 'name': agent.output_type.__name__}
        elif hasattr(agent.output_type, 'model_dump'):
            output_type_json = {'type': 'object', 'data': agent.output_type.model_dump(mode='json')}
        else:
            output_type_json = {'type': 'other', 'data': str(agent.output_type)}
    
    # Serialize tool_use_behavior
    tool_use_behavior_json = None
    if isinstance(agent.tool_use_behavior, str):
        tool_use_behavior_json = {'type': 'string', 'value': agent.tool_use_behavior}
    elif isinstance(agent.tool_use_behavior, dict):
        tool_use_behavior_json = {'type': 'dict', 'value': agent.tool_use_behavior}
    elif callable(agent.tool_use_behavior):
        tool_use_behavior_json = {'type': 'function', 'name': getattr(agent.tool_use_behavior, '__name__', str(id(agent.tool_use_behavior)))}
    
    # Serialize mcp_config (Python only)
    mcp_config_json = agent.mcp_config if hasattr(agent, 'mcp_config') else {}
    
    # Insert agent record
    agent_id = str(uuid.uuid4())
    await db.execute("""
        INSERT INTO agents (
            id, name, instructions, instructions_fn, prompt, prompt_fn,
            handoff_description, model, model_settings, output_type,
            tool_use_behavior, reset_tool_choice, mcp_config
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
    """, agent_id, agent.name, instructions, instructions_fn, 
        json.dumps(prompt) if prompt else None, prompt_fn,
        agent.handoff_description, 
        agent.model if isinstance(agent.model, str) else None,
        json.dumps(model_settings_json), json.dumps(output_type_json) if output_type_json else None,
        json.dumps(tool_use_behavior_json) if tool_use_behavior_json else None,
        agent.reset_tool_choice, json.dumps(mcp_config_json))
    
    # Save tools
    for order_index, tool in enumerate(agent.tools):
        tool_id = await _save_tool(tool, db)
        await db.execute("""
            INSERT INTO agent_tools (agent_id, tool_id, order_index)
            VALUES ($1, $2, $3)
        """, agent_id, tool_id, order_index)
    
    # Save input guardrails
    for order_index, guardrail in enumerate(agent.input_guardrails):
        guardrail_id = await _save_guardrail(guardrail, 'input', db)
        await db.execute("""
            INSERT INTO agent_guardrails (agent_id, guardrail_id, type, order_index)
            VALUES ($1, $2, $3, $4)
        """, agent_id, guardrail_id, 'input', order_index)
    
    # Save output guardrails
    for order_index, guardrail in enumerate(agent.output_guardrails):
        guardrail_id = await _save_guardrail(guardrail, 'output', db)
        await db.execute("""
            INSERT INTO agent_guardrails (agent_id, guardrail_id, type, order_index)
            VALUES ($1, $2, $3, $4)
        """, agent_id, guardrail_id, 'output', order_index)
    
    # Save handoffs (will need to save handoff agents first recursively)
    for order_index, handoff in enumerate(agent.handoffs):
        await _save_handoff(handoff, agent_id, order_index, db)
    
    # Save mcp_servers
    if hasattr(agent, 'mcp_servers'):
        for order_index, mcp_server in enumerate(agent.mcp_servers):
            mcp_server_id = await _save_mcp_server(mcp_server, db)
            await db.execute("""
                INSERT INTO agent_mcp_servers (agent_id, mcp_server_id, order_index)
                VALUES ($1, $2, $3)
            """, agent_id, mcp_server_id, order_index)
    
    return agent_id


async def _save_tool(tool: Tool, db: DatabaseConnection) -> str:
    """Save a tool and return its ID."""
    # Check if tool already exists by name
    existing = await db.fetchrow("""
        SELECT id FROM tools WHERE name = $1 AND type = $2
    """, tool.name if hasattr(tool, 'name') else None, 
        tool.type if hasattr(tool, 'type') else 'function')
    
    if existing:
        tool_id = existing['id']
    else:
        tool_id = str(uuid.uuid4())
        # Extract tool metadata
        tool_metadata = {}
        if hasattr(tool, 'invoke') and callable(tool.invoke):
            tool_metadata['invoke_fn'] = getattr(tool.invoke, '__name__', str(id(tool.invoke)))
        if hasattr(tool, 'needs_approval'):
            needs_approval = tool.needs_approval
            if callable(needs_approval):
                tool_metadata['needs_approval_fn'] = getattr(needs_approval, '__name__', str(id(needs_approval)))
            else:
                tool_metadata['needs_approval'] = needs_approval
        if hasattr(tool, 'is_enabled'):
            is_enabled = tool.is_enabled
            if callable(is_enabled):
                tool_metadata['is_enabled_fn'] = getattr(is_enabled, '__name__', str(id(is_enabled)))
            else:
                tool_metadata['is_enabled'] = is_enabled
        
        parameters = None
        if hasattr(tool, 'parameters'):
            if isinstance(tool.parameters, dict):
                parameters = tool.parameters
            elif hasattr(tool.parameters, 'model_dump'):
                parameters = tool.parameters.model_dump(mode='json')
        
        await db.execute("""
            INSERT INTO tools (id, type, name, description, parameters, strict, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, tool_id, 
            tool.type if hasattr(tool, 'type') else 'function',
            tool.name if hasattr(tool, 'name') else None,
            tool.description if hasattr(tool, 'description') else None,
            json.dumps(parameters) if parameters else None,
            tool.strict if hasattr(tool, 'strict') else None,
            json.dumps(tool_metadata))
    
    # Register tool in registry
    register_tool(tool_id, tool)
    return tool_id


async def _save_guardrail(guardrail: InputGuardrail | OutputGuardrail, guardrail_type: str, db: DatabaseConnection) -> str:
    """Save a guardrail and return its ID."""
    guardrail_name = None
    if hasattr(guardrail, 'get_name'):
        guardrail_name = guardrail.get_name()
    elif hasattr(guardrail, 'name'):
        guardrail_name = guardrail.name
    
    # Check if guardrail already exists
    existing = await db.fetchrow("""
        SELECT id FROM guardrails WHERE name = $1 AND type = $2
    """, guardrail_name, guardrail_type)
    
    if existing:
        guardrail_id = existing['id']
    else:
        guardrail_id = str(uuid.uuid4())
        # Extract guardrail metadata
        metadata = {}
        if hasattr(guardrail, 'guardrail_function') and callable(guardrail.guardrail_function):
            metadata['execute_fn'] = getattr(guardrail.guardrail_function, '__name__', str(id(guardrail.guardrail_function)))
        elif hasattr(guardrail, 'execute') and callable(guardrail.execute):
            metadata['execute_fn'] = getattr(guardrail.execute, '__name__', str(id(guardrail.execute)))
        
        if guardrail_type == 'input' and hasattr(guardrail, 'run_in_parallel'):
            metadata['run_in_parallel'] = guardrail.run_in_parallel
        
        await db.execute("""
            INSERT INTO guardrails (id, name, type, metadata)
            VALUES ($1, $2, $3, $4)
        """, guardrail_id, guardrail_name, guardrail_type, json.dumps(metadata))
    
    # Register guardrail in registry
    if guardrail_type == 'input':
        register_input_guardrail(guardrail_id, guardrail)
    else:
        register_output_guardrail(guardrail_id, guardrail)
    
    return guardrail_id


async def _save_handoff(handoff: Agent | Handoff, agent_id: str, order_index: int, db: DatabaseConnection) -> None:
    """Save a handoff (Agent or Handoff object)."""
    if isinstance(handoff, Agent):
        # Save the handoff agent first
        handoff_agent_id = await save_agent(handoff, db)
        tool_name = f"transfer_to_{handoff.name.lower().replace(' ', '_')}"
        tool_description = f"Handoff to the {handoff.name} agent"
        input_schema = {}
        metadata = {}
    else:
        # Handoff object
        handoff_agent_id = None
        if hasattr(handoff, 'agent') and isinstance(handoff.agent, Agent):
            handoff_agent_id = await save_agent(handoff.agent, db)
        
        tool_name = handoff.tool_name if hasattr(handoff, 'tool_name') else None
        tool_description = handoff.tool_description if hasattr(handoff, 'tool_description') else None
        input_schema = handoff.input_json_schema if hasattr(handoff, 'input_json_schema') else {}
        
        metadata = {}
        if hasattr(handoff, 'input_filter') and callable(handoff.input_filter):
            metadata['input_filter_fn'] = getattr(handoff.input_filter, '__name__', str(id(handoff.input_filter)))
        if hasattr(handoff, 'is_enabled'):
            is_enabled = handoff.is_enabled
            if callable(is_enabled):
                metadata['is_enabled_fn'] = getattr(is_enabled, '__name__', str(id(is_enabled)))
            else:
                metadata['is_enabled'] = is_enabled
        if hasattr(handoff, 'on_invoke_handoff') and callable(handoff.on_invoke_handoff):
            metadata['on_invoke_handoff_fn'] = getattr(handoff.on_invoke_handoff, '__name__', str(id(handoff.on_invoke_handoff)))
    
    await db.execute("""
        INSERT INTO agent_handoffs (
            agent_id, handoff_agent_id, tool_name, tool_description,
            input_json_schema, strict_json_schema, order_index, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """, agent_id, handoff_agent_id, tool_name, tool_description,
        json.dumps(input_schema),
        handoff.strict_json_schema if hasattr(handoff, 'strict_json_schema') else True,
        order_index, json.dumps(metadata))


async def _save_mcp_server(mcp_server: Any, db: DatabaseConnection) -> str:
    """Save an MCP server and return its ID."""
    server_name = getattr(mcp_server, 'name', None) or getattr(mcp_server, '__class__', type(mcp_server)).__name__
    server_type = 'unknown'
    if hasattr(mcp_server, '__class__'):
        class_name = mcp_server.__class__.__name__
        if 'Stdio' in class_name:
            server_type = 'stdio'
        elif 'Sse' in class_name:
            server_type = 'sse'
        elif 'Http' in class_name or 'StreamableHttp' in class_name:
            server_type = 'http'
    
    # Check if server already exists
    existing = await db.fetchrow("""
        SELECT id FROM mcp_servers WHERE name = $1 AND server_type = $2
    """, server_name, server_type)
    
    if existing:
        return existing['id']
    
    server_id = str(uuid.uuid4())
    config_data = {}
    # Try to extract config from server
    if hasattr(mcp_server, '__dict__'):
        config_data = {k: str(v) for k, v in mcp_server.__dict__.items() if not callable(v)}
    
    await db.execute("""
        INSERT INTO mcp_servers (id, name, server_type, config_data)
        VALUES ($1, $2, $3, $4)
    """, server_id, server_name, server_type, json.dumps(config_data))
    
    return server_id


async def load_agent(agent_id: str, db: DatabaseConnection) -> Agent:
    """
    Load an agent configuration from the database.
    
    Args:
        agent_id: The agent ID (UUID as string)
        db: DatabaseConnection instance
        
    Returns:
        The reconstructed Agent object
    """
    # Load agent record
    agent_row = await db.fetchrow("""
        SELECT * FROM agents WHERE id = $1
    """, agent_id)
    
    if not agent_row:
        raise ValueError(f"Agent with id {agent_id} not found")
    
    # Reconstruct instructions
    instructions = agent_row['instructions']
    if not instructions and agent_row['instructions_fn']:
        # Dynamic instructions - would need to be registered
        # For now, use empty string as fallback
        instructions = ""
    
    # Reconstruct prompt
    prompt = None
    if agent_row['prompt']:
        prompt = json.loads(agent_row['prompt'])
    elif agent_row['prompt_fn']:
        # Dynamic prompt - would need to be registered
        pass
    
    # Reconstruct model_settings
    model_settings_dict = json.loads(agent_row['model_settings']) if agent_row['model_settings'] else {}
    model_settings = ModelSettings(**model_settings_dict)
    
    # Reconstruct output_type
    output_type = None
    if agent_row['output_type']:
        output_type_data = json.loads(agent_row['output_type'])
        # Would need to reconstruct from type name or data
        # For now, leave as None
        output_type = None
    
    # Reconstruct tool_use_behavior
    tool_use_behavior = "run_llm_again"
    if agent_row['tool_use_behavior']:
        behavior_data = json.loads(agent_row['tool_use_behavior'])
        if behavior_data.get('type') == 'string':
            tool_use_behavior = behavior_data['value']
        elif behavior_data.get('type') == 'dict':
            tool_use_behavior = behavior_data['value']
        # Function type would need to be registered
    
    # Load tools
    tool_rows = await db.fetch("""
        SELECT t.* FROM tools t
        JOIN agent_tools at ON t.id = at.tool_id
        WHERE at.agent_id = $1
        ORDER BY at.order_index
    """, agent_id)
    
    tools = []
    for tool_row in tool_rows:
        tool = get_tool(tool_row['id'])
        if tool:
            tools.append(tool)
        else:
            # Tool not in registry - would need to reconstruct from metadata
            # For now, skip
            pass
    
    # Load input guardrails
    input_guardrail_rows = await db.fetch("""
        SELECT g.* FROM guardrails g
        JOIN agent_guardrails ag ON g.id = ag.guardrail_id
        WHERE ag.agent_id = $1 AND ag.type = 'input'
        ORDER BY ag.order_index
    """, agent_id)
    
    input_guardrails = []
    for guardrail_row in input_guardrail_rows:
        guardrail = get_input_guardrail(guardrail_row['id'])
        if guardrail:
            input_guardrails.append(guardrail)
    
    # Load output guardrails
    output_guardrail_rows = await db.fetch("""
        SELECT g.* FROM guardrails g
        JOIN agent_guardrails ag ON g.id = ag.guardrail_id
        WHERE ag.agent_id = $1 AND ag.type = 'output'
        ORDER BY ag.order_index
    """, agent_id)
    
    output_guardrails = []
    for guardrail_row in output_guardrail_rows:
        guardrail = get_output_guardrail(guardrail_row['id'])
        if guardrail:
            output_guardrails.append(guardrail)
    
    # Load handoffs (recursively load handoff agents)
    handoff_rows = await db.fetch("""
        SELECT * FROM agent_handoffs
        WHERE agent_id = $1
        ORDER BY order_index
    """, agent_id)
    
    handoffs = []
    for handoff_row in handoff_rows:
        if handoff_row['handoff_agent_id']:
            # Load the handoff agent
            handoff_agent = await load_agent(handoff_row['handoff_agent_id'], db)
            handoffs.append(handoff_agent)
        # TODO: Handle Handoff objects (not just Agent)
    
    # Load mcp_servers
    mcp_server_rows = await db.fetch("""
        SELECT ms.* FROM mcp_servers ms
        JOIN agent_mcp_servers ams ON ms.id = ams.mcp_server_id
        WHERE ams.agent_id = $1
        ORDER BY ams.order_index
    """, agent_id)
    
    mcp_servers = []
    # MCP servers cannot be fully reconstructed - would need to be reconnected
    # For now, leave empty
    
    # Reconstruct Agent
    agent_kwargs = {
        'name': agent_row['name'],
        'instructions': instructions,
        'prompt': prompt,
        'handoff_description': agent_row['handoff_description'],
        'model': agent_row['model'],
        'model_settings': model_settings,
        'tools': tools,
        'input_guardrails': input_guardrails,
        'output_guardrails': output_guardrails,
        'output_type': output_type,
        'tool_use_behavior': tool_use_behavior,
        'reset_tool_choice': agent_row['reset_tool_choice'],
        'handoffs': handoffs,
    }
    
    # Add mcp_config if it exists (Python only)
    if agent_row['mcp_config']:
        agent_kwargs['mcp_config'] = json.loads(agent_row['mcp_config'])
    
    # Add mcp_servers if any
    if mcp_servers:
        agent_kwargs['mcp_servers'] = mcp_servers
    
    return Agent(**agent_kwargs)

