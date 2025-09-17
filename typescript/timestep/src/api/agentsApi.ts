/**
 * Agents API
 *
 * This module provides functions for managing agent configurations.
 * It uses the agent service pattern for data operations.
 */

import type { AgentCard, AgentSkill } from "@a2a-js/sdk";
import { TaskStore } from "@a2a-js/sdk/server";
import { AgentExecutor } from "@a2a-js/sdk/server";
import { ContextAwareRequestHandler } from "./contextAwareRequestHandler.js";
import { Request, Response, NextFunction } from "express";
import { AgentService } from "../services/agentService.js";
import { JsonlAgentRepository } from "../services/backing/jsonlAgentRepository.js";
import { Repository } from "../services/backing/repository.js";

/**
 * Represents an agent configuration
 */
export interface Agent {
  /** The agent identifier */
  id: string;
  /** The agent name */
  name: string;
  /** System instructions for the agent */
  instructions: string;
  /** Description for handoff purposes */
  handoffDescription?: string;
  /** IDs of agents this agent can hand off to */
  handoffIds?: string[];
  /** IDs of tools available to this agent */
  toolIds: string[];
  /** Model identifier to use */
  model: string;
  /** Model configuration settings */
  modelSettings: {
    temperature: number;
    [key: string]: any;
  };
}

/**
 * Response from the list agents endpoint
 */
export interface ListAgentsResponse {
  /** The object type, which is always "list" */
  object: 'list';
  /** Array of agent objects */
  data: Agent[];
}


/**
 * List all available agents using the agent service
 *
 * @param repository Optional repository for dependency injection. Defaults to JsonlAgentRepository
 * @returns Promise resolving to the list of agents
 */
export async function listAgents(repository?: Repository<Agent, string>): Promise<ListAgentsResponse> {
  const repo = repository || new JsonlAgentRepository();
  const agentService = new AgentService(repo);

  try {
    const agents = await agentService.listAgents();
    return {
      object: 'list',
      data: agents,
    };
  } catch (error) {
    throw new Error(`Failed to list agents: ${error}`);
  }
}

/**
 * Retrieve a specific agent by ID using the agent service
 *
 * @param agentId - The ID of the agent to retrieve
 * @param repository Optional repository for dependency injection. Defaults to JsonlAgentRepository
 * @returns Promise resolving to the agent details or null if not found
 */
export async function getAgent(agentId: string, repository?: Repository<Agent, string>): Promise<Agent | null> {
  const repo = repository || new JsonlAgentRepository();
  const agentService = new AgentService(repo);

  try {
    return await agentService.getAgent(agentId);
  } catch (error) {
    throw new Error(`Failed to get agent: ${error}`);
  }
}

/**
 * Check if an agent is available using the agent service
 *
 * @param agentId - The ID of the agent to check
 * @param repository Optional repository for dependency injection. Defaults to JsonlAgentRepository
 * @returns Promise resolving to true if agent exists, false otherwise
 */
export async function isAgentAvailable(agentId: string, repository?: Repository<Agent, string>): Promise<boolean> {
  const repo = repository || new JsonlAgentRepository();
  const agentService = new AgentService(repo);

  try {
    return await agentService.isAgentAvailable(agentId);
  } catch (error) {
    throw new Error(`Failed to check agent availability: ${error}`);
  }
}

/**
 * Create an agent card for a specific agent
 *
 * @param agentId - The ID of the agent
 * @param serverPort - The server port for the agent URL
 * @returns Promise resolving to the agent card
 */
export async function getAgentCard(agentId: string, serverPort: number): Promise<AgentCard> {
  const agent = await getAgent(agentId);
  if (!agent) {
    throw new Error(`Agent with ID ${agentId} not found`);
  }

  // Define agent skills
  const helloSkill: AgentSkill = {
    id: 'hello_world',
    name: 'Returns hello world',
    description: 'just returns hello world',
    tags: ['hello world'],
    examples: ['hi', 'hello world'],
  };

  const publicAgentCard: AgentCard = {
    name: agent.name,
    description: `A helpful AI agent powered by ${agent.name}`,
    url: `http://localhost:${serverPort}/agents/${agentId}/`,
    version: '1.0.0',
    protocolVersion: '0.3.0',
    preferredTransport: 'JSONRPC',
    defaultInputModes: ['text'],
    defaultOutputModes: ['text'],
    capabilities: { streaming: true },
    skills: [helloSkill],
    supportsAuthenticatedExtendedCard: true,
  };

  return publicAgentCard;
}

/**
 * Create a context-aware request handler for a specific agent
 *
 * @param agentId - The ID of the agent
 * @param taskStore - The shared task store
 * @param agentExecutor - The agent executor
 * @param serverPort - The server port for the agent URL
 * @returns Promise resolving to the configured request handler
 */
export async function createAgentRequestHandler(
  agentId: string,
  taskStore: TaskStore,
  agentExecutor: AgentExecutor,
  serverPort: number
): Promise<ContextAwareRequestHandler> {
  const agent = await getAgent(agentId);
  if (!agent) {
    throw new Error(`Agent with ID ${agentId} not found`);
  }

  const agentCard = await getAgentCard(agentId, serverPort);

  return new ContextAwareRequestHandler(
    agentId,
    agentCard,
    taskStore,
    agentExecutor
  );
}

/**
 * Express route handler for listing agents
 */
export async function handleListAgents(_req: Request, res: Response): Promise<void> {
  try {
    const response = await listAgents();
    res.json(response.data);
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to load agents'
    });
  }
}

/**
 * Express route handler for dynamic agent requests
 */
export async function handleAgentRequest(
  req: Request, 
  res: Response, 
  next: NextFunction,
  taskStore: TaskStore,
  agentExecutor: AgentExecutor,
  serverPort: number
): Promise<void> {
  console.log(`🔍 Dynamic route handler called for agent: ${req.params['agentId']}, method: ${req.method}, path: ${req.path}`);
  try {
    const agentId = req.params['agentId'];
    
    // Check if agent exists
    if (!(await isAgentAvailable(agentId))) {
      console.log(`❌ Agent ${agentId} not found`);
      res.status(404).json({ 
        error: 'Agent not found',
        agentId: agentId 
      });
      return;
    }
    
    // Create request handler dynamically
    const requestHandler = await createAgentRequestHandler(
      agentId, 
      taskStore, 
      agentExecutor, 
      serverPort
    );
    
    // Create A2A Express app and delegate
    const { A2AExpressApp } = await import("@a2a-js/sdk/server/express");
    const agentAppBuilder = new A2AExpressApp(requestHandler);
    const agentApp = (await import("express")).default();
    agentAppBuilder.setupRoutes(agentApp);
    
    // Delegate to the agent-specific app
    agentApp(req, res, next);
    
  } catch (error) {
    console.error(`Error handling request for agent ${req.params['agentId']}:`, error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}