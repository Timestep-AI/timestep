/** Agent store for saving and loading Agent configurations from database. */

import { randomUUID } from 'crypto';
import type { Agent, Tool, InputGuardrail, OutputGuardrail, Handoff, ModelSettings } from '@openai/agents';
import { DatabaseConnection } from '../shared/db_connection.ts';
import { registerTool, getTool } from '../tool_registry.ts';
import {
  registerInputGuardrail,
  registerOutputGuardrail,
  getInputGuardrail,
  getOutputGuardrail,
} from '../guardrail_registry.ts';

export async function saveAgent(agent: Agent): Promise<string> {
  /**
   * Save an agent configuration to the database.
   * Manages database connection internally.
   *
   * @param agent - The Agent object to save
   * @returns The agent_id (UUID as string)
   */
  // Get connection string from environment
  const connectionString = process.env.PG_CONNECTION_URI;
  if (!connectionString) {
    throw new Error('PG_CONNECTION_URI environment variable not set');
  }

  // Create and manage database connection
  const db = new DatabaseConnection({ connectionString });
  await db.connect();
  try {
    return await saveAgentInternal(agent, db);
  } finally {
    await db.disconnect();
  }
}

async function saveAgentInternal(agent: Agent, db: DatabaseConnection): Promise<string> {
  /**
   * Internal function that saves an agent using an existing database connection.
   *
   * @param agent - The Agent object to save
   * @param db - DatabaseConnection instance (already connected)
   * @returns The agent_id (UUID as string)
   */
  // Extract serializable fields from agent
  let instructions: string | null = null;
  let instructionsFn: string | null = null;
  if (typeof agent.instructions === 'string') {
    instructions = agent.instructions;
  } else if (typeof agent.instructions === 'function') {
    instructionsFn = agent.instructions.name || String(agent.instructions);
  }

  let prompt: any = null;
  let promptFn: string | null = null;
  if (agent.prompt !== undefined && agent.prompt !== null) {
    if (typeof agent.prompt === 'function') {
      promptFn = agent.prompt.name || String(agent.prompt);
    } else {
      // Serialize prompt object
      if (typeof agent.prompt === 'object' && 'model_dump' in agent.prompt) {
        prompt = (agent.prompt as any).model_dump({ mode: 'json' });
      } else {
        prompt = JSON.parse(JSON.stringify(agent.prompt));
      }
    }
  }

  // Serialize model_settings
  const modelSettingsJson = agent.modelSettings?.toJSON?.() || {};

  // Serialize output_type
  let outputTypeJson: any = null;
  if (agent.outputType !== undefined && agent.outputType !== null) {
    if (typeof agent.outputType === 'string') {
      outputTypeJson = { type: 'string', value: agent.outputType };
    } else if (typeof agent.outputType === 'object') {
      outputTypeJson = { type: 'object', data: JSON.parse(JSON.stringify(agent.outputType)) };
    }
  }

  // Serialize tool_use_behavior
  let toolUseBehaviorJson: any = null;
  if (typeof agent.toolUseBehavior === 'string') {
    toolUseBehaviorJson = { type: 'string', value: agent.toolUseBehavior };
  } else if (typeof agent.toolUseBehavior === 'object') {
    toolUseBehaviorJson = { type: 'object', value: JSON.parse(JSON.stringify(agent.toolUseBehavior)) };
  } else if (typeof agent.toolUseBehavior === 'function') {
    toolUseBehaviorJson = { type: 'function', name: agent.toolUseBehavior.name || String(agent.toolUseBehavior) };
  }

  // Insert agent record
  const agentId = randomUUID();
  await db.query(
    `
    INSERT INTO agents (
      id, name, instructions, instructions_fn, prompt, prompt_fn,
      handoff_description, handoff_output_type_warning_enabled, model, model_settings, output_type,
      tool_use_behavior, reset_tool_choice, mcp_config
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
    `,
    [
      agentId,
      agent.name,
      instructions,
      instructionsFn,
      prompt ? JSON.stringify(prompt) : null,
      promptFn,
      agent.handoffDescription || null,
      (agent as any).handoffOutputTypeWarningEnabled || null,
      typeof agent.model === 'string' ? agent.model : null,
      JSON.stringify(modelSettingsJson),
      outputTypeJson ? JSON.stringify(outputTypeJson) : null,
      toolUseBehaviorJson ? JSON.stringify(toolUseBehaviorJson) : null,
      agent.resetToolChoice ?? true,
      JSON.stringify({}), // mcp_config (Python only, but we store empty object for TypeScript)
    ]
  );

  // Save tools
  for (let orderIndex = 0; orderIndex < agent.tools.length; orderIndex++) {
    const tool = agent.tools[orderIndex];
    const toolId = await saveTool(tool, db);
    await db.query(
      `
      INSERT INTO agent_tools (agent_id, tool_id, order_index)
      VALUES ($1, $2, $3)
      `,
      [agentId, toolId, orderIndex]
    );
  }

  // Save input guardrails
  for (let orderIndex = 0; orderIndex < agent.inputGuardrails.length; orderIndex++) {
    const guardrail = agent.inputGuardrails[orderIndex];
    const guardrailId = await saveGuardrail(guardrail, 'input', db);
    await db.query(
      `
      INSERT INTO agent_guardrails (agent_id, guardrail_id, type, order_index)
      VALUES ($1, $2, $3, $4)
      `,
      [agentId, guardrailId, 'input', orderIndex]
    );
  }

  // Save output guardrails
  for (let orderIndex = 0; orderIndex < agent.outputGuardrails.length; orderIndex++) {
    const guardrail = agent.outputGuardrails[orderIndex];
    const guardrailId = await saveGuardrail(guardrail, 'output', db);
    await db.query(
      `
      INSERT INTO agent_guardrails (agent_id, guardrail_id, type, order_index)
      VALUES ($1, $2, $3, $4)
      `,
      [agentId, guardrailId, 'output', orderIndex]
    );
  }

  // Save handoffs (will need to save handoff agents first recursively)
  for (let orderIndex = 0; orderIndex < agent.handoffs.length; orderIndex++) {
    const handoff = agent.handoffs[orderIndex];
    await saveHandoff(handoff, agentId, orderIndex, db);
  }

  // Save mcp_servers
  if (agent.mcpServers) {
    for (let orderIndex = 0; orderIndex < agent.mcpServers.length; orderIndex++) {
      const mcpServer = agent.mcpServers[orderIndex];
      const mcpServerId = await saveMcpServer(mcpServer, db);
      await db.query(
        `
        INSERT INTO agent_mcp_servers (agent_id, mcp_server_id, order_index)
        VALUES ($1, $2, $3)
        `,
        [agentId, mcpServerId, orderIndex]
      );
    }
  }

  return agentId;
}

async function saveTool(tool: Tool<any>, db: DatabaseConnection): Promise<string> {
  /** Save a tool and return its ID. */
  // Check if tool already exists by name
  const existing = await db.query(
    `
    SELECT id FROM tools WHERE name = $1 AND type = $2
    `,
    [tool.name || null, tool.type || 'function']
  );

  let toolId: string;
  if (existing.rows && existing.rows.length > 0) {
    toolId = existing.rows[0].id;
  } else {
    toolId = randomUUID();
    // Extract tool metadata
    const toolMetadata: any = {};
    if (tool.invoke && typeof tool.invoke === 'function') {
      toolMetadata.invokeFn = tool.invoke.name || String(tool.invoke);
    }
    if (tool.needsApproval !== undefined) {
      if (typeof tool.needsApproval === 'function') {
        toolMetadata.needsApprovalFn = tool.needsApproval.name || String(tool.needsApproval);
      } else {
        toolMetadata.needsApproval = tool.needsApproval;
      }
    }
    if (tool.isEnabled !== undefined) {
      if (typeof tool.isEnabled === 'function') {
        toolMetadata.isEnabledFn = tool.isEnabled.name || String(tool.isEnabled);
      } else {
        toolMetadata.isEnabled = tool.isEnabled;
      }
    }

    let parameters: any = null;
    if (tool.parameters) {
      if (typeof tool.parameters === 'object') {
        parameters = tool.parameters;
      } else if (typeof (tool.parameters as any).model_dump === 'function') {
        parameters = (tool.parameters as any).model_dump({ mode: 'json' });
      }
    }

    await db.query(
      `
      INSERT INTO tools (id, type, name, description, parameters, strict, metadata)
      VALUES ($1, $2, $3, $4, $5, $6, $7)
      `,
      [
        toolId,
        tool.type || 'function',
        tool.name || null,
        tool.description || null,
        parameters ? JSON.stringify(parameters) : null,
        tool.strict || null,
        JSON.stringify(toolMetadata),
      ]
    );
  }

  // Register tool in registry
  registerTool(toolId, tool);
  return toolId;
}

async function saveGuardrail(
  guardrail: InputGuardrail | OutputGuardrail,
  guardrailType: 'input' | 'output',
  db: DatabaseConnection
): Promise<string> {
  /** Save a guardrail and return its ID. */
  const guardrailName = guardrail.name || null;

  // Check if guardrail already exists
  const existing = await db.query(
    `
    SELECT id FROM guardrails WHERE name = $1 AND type = $2
    `,
    [guardrailName, guardrailType]
  );

  let guardrailId: string;
  if (existing.rows && existing.rows.length > 0) {
    guardrailId = existing.rows[0].id;
  } else {
    guardrailId = randomUUID();
    // Extract guardrail metadata
    const metadata: any = {};
    if (guardrail.execute && typeof guardrail.execute === 'function') {
      metadata.executeFn = guardrail.execute.name || String(guardrail.execute);
    }

    if (guardrailType === 'input' && 'runInParallel' in guardrail) {
      metadata.runInParallel = (guardrail as InputGuardrail).runInParallel;
    }

    await db.query(
      `
      INSERT INTO guardrails (id, name, type, metadata)
      VALUES ($1, $2, $3, $4)
      `,
      [guardrailId, guardrailName, guardrailType, JSON.stringify(metadata)]
    );
  }

  // Register guardrail in registry
  if (guardrailType === 'input') {
    registerInputGuardrail(guardrailId, guardrail as InputGuardrail);
  } else {
    registerOutputGuardrail(guardrailId, guardrail as OutputGuardrail);
  }

  return guardrailId;
}

async function saveHandoff(
  handoff: Agent | Handoff,
  agentId: string,
  orderIndex: number,
  db: DatabaseConnection
): Promise<void> {
  /** Save a handoff (Agent or Handoff object). */
  let handoffAgentId: string | null = null;
  let toolName: string | null = null;
  let toolDescription: string | null = null;
  let inputSchema: any = {};
  const metadata: any = {};

  if ('name' in handoff && 'tools' in handoff) {
    // It's an Agent (using internal function to reuse connection)
    handoffAgentId = await saveAgentInternal(handoff as Agent, db);
    toolName = `transfer_to_${(handoff as Agent).name.toLowerCase().replace(/\s+/g, '_')}`;
    toolDescription = `Handoff to the ${(handoff as Agent).name} agent`;
  } else {
    // Handoff object
    const handoffObj = handoff as Handoff;
    if (handoffObj.agent && 'name' in handoffObj.agent) {
      handoffAgentId = await saveAgentInternal(handoffObj.agent as Agent, db);
    }

    toolName = handoffObj.toolName || null;
    toolDescription = handoffObj.toolDescription || null;
    inputSchema = handoffObj.inputJsonSchema || {};

    if (handoffObj.inputFilter && typeof handoffObj.inputFilter === 'function') {
      metadata.inputFilterFn = handoffObj.inputFilter.name || String(handoffObj.inputFilter);
    }
    if (handoffObj.isEnabled !== undefined) {
      if (typeof handoffObj.isEnabled === 'function') {
        metadata.isEnabledFn = handoffObj.isEnabled.name || String(handoffObj.isEnabled);
      } else {
        metadata.isEnabled = handoffObj.isEnabled;
      }
    }
    if (handoffObj.onInvokeHandoff && typeof handoffObj.onInvokeHandoff === 'function') {
      metadata.onInvokeHandoffFn = handoffObj.onInvokeHandoff.name || String(handoffObj.onInvokeHandoff);
    }
  }

  await db.query(
    `
    INSERT INTO agent_handoffs (
      agent_id, handoff_agent_id, tool_name, tool_description,
      input_json_schema, strict_json_schema, order_index, metadata
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    `,
    [
      agentId,
      handoffAgentId,
      toolName,
      toolDescription,
      JSON.stringify(inputSchema),
      (handoff as any).strictJsonSchema ?? true,
      orderIndex,
      JSON.stringify(metadata),
    ]
  );
}

async function saveMcpServer(mcpServer: any, db: DatabaseConnection): Promise<string> {
  /** Save an MCP server and return its ID. */
  const serverName = mcpServer.name || mcpServer.constructor?.name || 'unknown';
  let serverType = 'unknown';
  if (mcpServer.constructor) {
    const className = mcpServer.constructor.name;
    if (className.includes('Stdio')) {
      serverType = 'stdio';
    } else if (className.includes('Sse')) {
      serverType = 'sse';
    } else if (className.includes('Http') || className.includes('StreamableHttp')) {
      serverType = 'http';
    }
  }

  // Check if server already exists
  const existing = await db.query(
    `
    SELECT id FROM mcp_servers WHERE name = $1 AND server_type = $2
    `,
    [serverName, serverType]
  );

  if (existing.rows && existing.rows.length > 0) {
    return existing.rows[0].id;
  }

  const serverId = randomUUID();
  const configData: any = {};
  // Try to extract config from server
  if (typeof mcpServer === 'object') {
    for (const [key, value] of Object.entries(mcpServer)) {
      if (typeof value !== 'function') {
        configData[key] = String(value);
      }
    }
  }

  await db.query(
    `
    INSERT INTO mcp_servers (id, name, server_type, config_data)
    VALUES ($1, $2, $3, $4)
    `,
    [serverId, serverName, serverType, JSON.stringify(configData)]
  );

  return serverId;
}

export async function loadAgent(agentId: string): Promise<Agent> {
  /**
   * Load an agent configuration from the database.
   * Manages database connection internally.
   *
   * @param agentId - The agent ID (UUID as string)
   * @returns The reconstructed Agent object
   */
  // Get connection string from environment
  const connectionString = process.env.PG_CONNECTION_URI;
  if (!connectionString) {
    throw new Error('PG_CONNECTION_URI environment variable not set');
  }

  // Create and manage database connection
  const db = new DatabaseConnection({ connectionString });
  await db.connect();
  try {
    return await loadAgentInternal(agentId, db);
  } finally {
    await db.disconnect();
  }
}

async function loadAgentInternal(agentId: string, db: DatabaseConnection): Promise<Agent> {
  /**
   * Internal function that loads an agent using an existing database connection.
   *
   * @param agentId - The agent ID (UUID as string)
   * @param db - DatabaseConnection instance (already connected)
   * @returns The reconstructed Agent object
   */
  // Load agent record
  const agentResult = await db.query(
    `
    SELECT * FROM agents WHERE id = $1
    `,
    [agentId]
  );

  if (!agentResult.rows || agentResult.rows.length === 0) {
    throw new Error(`Agent with id ${agentId} not found`);
  }

  const agentRow = agentResult.rows[0];

  // Reconstruct instructions
  let instructions: string | (() => string) = agentRow.instructions || '';
  if (!instructions && agentRow.instructions_fn) {
    // Dynamic instructions - would need to be registered
    // For now, use empty string as fallback
    instructions = '';
  }

  // Reconstruct prompt
  let prompt: any = null;
  if (agentRow.prompt) {
    // PostgreSQL JSON columns are returned as objects by pg driver, not strings
    if (typeof agentRow.prompt === 'string') {
      try {
        prompt = JSON.parse(agentRow.prompt);
      } catch {
        // If parsing fails, use as-is
        prompt = agentRow.prompt;
      }
    } else {
      // Already an object (pg driver returns JSON columns as objects)
      prompt = agentRow.prompt;
    }
  } else if (agentRow.prompt_fn) {
    // Dynamic prompt - would need to be registered
    // For now, leave as null
  }

  // Reconstruct model_settings
  let modelSettingsDict: any = {};
  if (agentRow.model_settings) {
    // PostgreSQL JSON columns are returned as objects by pg driver
    if (typeof agentRow.model_settings === 'string') {
      try {
        modelSettingsDict = JSON.parse(agentRow.model_settings);
      } catch {
        modelSettingsDict = {};
      }
    } else if (typeof agentRow.model_settings === 'object' && agentRow.model_settings !== null) {
      modelSettingsDict = agentRow.model_settings;
    }
  }
  // Would need to reconstruct ModelSettings object from dict
  const modelSettings = modelSettingsDict as ModelSettings;

  // Reconstruct output_type
  let outputType: any = null;
  if (agentRow.output_type) {
    // PostgreSQL JSON columns are returned as objects by pg driver
    let outputTypeData: any = null;
    if (typeof agentRow.output_type === 'string') {
      try {
        outputTypeData = JSON.parse(agentRow.output_type);
      } catch {
        outputTypeData = null;
      }
    } else if (typeof agentRow.output_type === 'object' && agentRow.output_type !== null) {
      outputTypeData = agentRow.output_type;
    }
    // Would need to reconstruct from type name or data
    // For now, leave as null
    outputType = null;
  }

  // Reconstruct tool_use_behavior
  let toolUseBehavior: any = 'run_llm_again';
  if (agentRow.tool_use_behavior) {
    let behaviorData: any = null;
    if (typeof agentRow.tool_use_behavior === 'string') {
      try {
        behaviorData = JSON.parse(agentRow.tool_use_behavior);
      } catch {
        behaviorData = null;
      }
    } else {
      behaviorData = agentRow.tool_use_behavior;
    }
    if (behaviorData) {
      if (behaviorData.type === 'string') {
        toolUseBehavior = behaviorData.value;
      } else if (behaviorData.type === 'object') {
        toolUseBehavior = behaviorData.value;
      }
      // Function type would need to be registered
    }
  }

  // Load tools
  const toolResult = await db.query(
    `
    SELECT t.* FROM tools t
    JOIN agent_tools at ON t.id = at.tool_id
    WHERE at.agent_id = $1
    ORDER BY at.order_index
    `,
    [agentId]
  );

  const tools: Tool<any>[] = [];
  for (const toolRow of toolResult.rows || []) {
    const tool = getTool(toolRow.id);
    if (tool) {
      tools.push(tool);
    }
    // Tool not in registry - would need to reconstruct from metadata
    // For now, skip
  }

  // Load input guardrails
  const inputGuardrailResult = await db.query(
    `
    SELECT g.* FROM guardrails g
    JOIN agent_guardrails ag ON g.id = ag.guardrail_id
    WHERE ag.agent_id = $1 AND ag.type = 'input'
    ORDER BY ag.order_index
    `,
    [agentId]
  );

  const inputGuardrails: InputGuardrail[] = [];
  for (const guardrailRow of inputGuardrailResult.rows || []) {
    const guardrail = getInputGuardrail(guardrailRow.id);
    if (guardrail) {
      inputGuardrails.push(guardrail);
    }
  }

  // Load output guardrails
  const outputGuardrailResult = await db.query(
    `
    SELECT g.* FROM guardrails g
    JOIN agent_guardrails ag ON g.id = ag.guardrail_id
    WHERE ag.agent_id = $1 AND ag.type = 'output'
    ORDER BY ag.order_index
    `,
    [agentId]
  );

  const outputGuardrails: OutputGuardrail[] = [];
  for (const guardrailRow of outputGuardrailResult.rows || []) {
    const guardrail = getOutputGuardrail(guardrailRow.id);
    if (guardrail) {
      outputGuardrails.push(guardrail);
    }
  }

  // Load handoffs (recursively load handoff agents)
  const handoffResult = await db.query(
    `
    SELECT * FROM agent_handoffs
    WHERE agent_id = $1
    ORDER BY order_index
    `,
    [agentId]
  );

  const handoffs: (Agent | Handoff)[] = [];
  for (const handoffRow of handoffResult.rows || []) {
    if (handoffRow.handoff_agent_id) {
      // Load the handoff agent (using internal function to reuse connection)
      const handoffAgent = await loadAgentInternal(handoffRow.handoff_agent_id, db);
      handoffs.push(handoffAgent);
    }
    // TODO: Handle Handoff objects (not just Agent)
  }

  // Load mcp_servers
  const mcpServerResult = await db.query(
    `
    SELECT ms.* FROM mcp_servers ms
    JOIN agent_mcp_servers ams ON ms.id = ams.mcp_server_id
    WHERE ams.agent_id = $1
    ORDER BY ams.order_index
    `,
    [agentId]
  );

  const mcpServers: any[] = [];
  // MCP servers cannot be fully reconstructed - would need to be reconnected
  // For now, leave empty

  // Reconstruct Agent
  const agentConfig: any = {
    name: agentRow.name,
    instructions,
    prompt,
    handoffDescription: agentRow.handoff_description,
    model: agentRow.model,
    modelSettings,
    tools,
    inputGuardrails,
    outputGuardrails,
    outputType,
    toolUseBehavior,
    resetToolChoice: agentRow.reset_tool_choice,
    handoffs,
  };

  // Add handoffOutputTypeWarningEnabled if it exists (TypeScript only)
  if (agentRow.handoff_output_type_warning_enabled !== null) {
    agentConfig.handoffOutputTypeWarningEnabled = agentRow.handoff_output_type_warning_enabled;
  }

  // Add mcp_servers if any
  if (mcpServers.length > 0) {
    agentConfig.mcpServers = mcpServers;
  }

  // Import Agent class dynamically to avoid circular dependencies
  const { Agent } = await import('@openai/agents');
  return new Agent(agentConfig);
}

