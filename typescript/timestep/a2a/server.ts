/** A2A server setup for Timestep agent. */

import express, { Router } from 'express';
import type { z } from 'zod';
import type { AgentCard, AgentSkill, Message } from '@a2a-js/sdk';
import {
  TaskStore,
  AgentExecutor,
  DefaultRequestHandler,
} from '@a2a-js/sdk/server';
import { A2AExpressApp } from '@a2a-js/sdk/server/express';
import { TimestepAgentExecutor } from './agent_executor.js';
import { PostgresTaskStore } from './postgres_task_store.js';
import { PostgresAgentStore, type Agent } from './postgres_agent_store.js';
import { openaiToA2A } from './message_converter.js';
import { GetWeatherParameters, WebSearchParameters, HandoffParameters } from '../core/tools.js';

function createSkillsFromTools(toolNames: string[]): AgentSkill[] {
  /** Create AgentSkill objects from available tools. */
  const skills: AgentSkill[] = [];
  
  const allSkills: Record<string, AgentSkill> = {
    get_weather: {
      id: 'get_weather',
      name: 'Get Weather',
      description: 'Returns weather info for the specified city.',
      tags: ['weather', 'city'],
      examples: [
        "What's the weather in Oakland?",
        'Get weather for San Francisco',
        'Weather in New York',
      ],
      inputModes: ['text'],
      outputModes: ['text', 'task-status'],
    },
    web_search: {
      id: 'web_search',
      name: 'Web Search',
      description: 'A tool that lets the LLM search the web using Firecrawl.',
      tags: ['search', 'web', 'information'],
      examples: [
        'Search for Python tutorials',
        'Find information about machine learning',
        'Look up the latest news about AI',
      ],
      inputModes: ['text'],
      outputModes: ['text', 'task-status'],
    },
    handoff: {
      id: 'handoff',
      name: 'Handoff',
      description: 'Hand off a message to another agent via A2A protocol.',
      tags: ['agent', 'handoff'],
      examples: [
        'Hand off to weather assistant',
        'Transfer to another agent',
      ],
      inputModes: ['text'],
      outputModes: ['text', 'task-status'],
    },
  };
  
  for (const toolName of toolNames) {
    if (allSkills[toolName]) {
      skills.push(allSkills[toolName]);
    }
  }
  
  return skills;
}

export function createAgentCard(
  agent: Agent,
  url: string = 'http://localhost:8080/'
): AgentCard {
  /** Create the AgentCard for a specific agent.
   *
   * @param agent - Agent configuration.
   * @param url - Base URL for the agent server.
   * @returns AgentCard instance.
   */
  const skills = createSkillsFromTools(agent.tools);
  
  return {
    name: agent.name,
    description: agent.description || 'Timestep agent with tool support.',
    url: url,
    version: '2026.0.5',
    defaultInputModes: ['text'],
    defaultOutputModes: ['text', 'task-status'],
    capabilities: {
      streaming: true,
      pushNotifications: false,
      stateTransitionHistory: true,
    },
    skills: skills,
    supportsAuthenticatedExtendedCard: false,
  } as AgentCard & { examples?: string[] };
}

function buildToolsFromAgent(agent: Agent): Array<{ name: string; parameters: z.ZodTypeAny }> {
  /** Build tool array from agent configuration. */
  const toolMap: Record<string, z.ZodTypeAny> = {
    get_weather: GetWeatherParameters,
    web_search: WebSearchParameters,
    handoff: HandoffParameters,
  };
  
  return agent.tools
    .filter(toolName => toolMap[toolName])
    .map(toolName => ({
      name: toolName,
      parameters: toolMap[toolName],
    }));
}

export function createServer(
  host: string = '0.0.0.0',
  port: number = 8080,
  tools?: Array<{ name: string; parameters: z.ZodTypeAny }>,
  model: string = 'gpt-4.1'
): express.Application {
  /** Create and configure the A2A server with agent-scoped routes.
   *
   * @param host - Host address to bind to.
   * @param port - Port number to listen on.
   * @param tools - List of tool objects to use. If undefined, uses default tools (deprecated, agents loaded from DB).
   * @param model - OpenAI model name to use (deprecated, models loaded from DB).
   * @returns Configured Express application.
   */
  const app = express();
  app.use(express.json());
  
  // Get public-facing URL for agent cards (for inter-container communication)
  // Use environment variables if set, otherwise use Docker service name, fallback to host
  const publicHost = process.env.A2A_PUBLIC_HOST || (host === '0.0.0.0' ? 'a2a-server' : host);
  const publicPort = process.env.A2A_PUBLIC_PORT || port.toString();
  const publicUrl = `http://${publicHost}:${publicPort}`;
  
  console.log(`[Server] Using public URL for agent cards: ${publicUrl}`);
  
  const agentStore = new PostgresAgentStore();
  // Create a base task store (will be cloned per agent)
  const baseTaskStore: TaskStore = new PostgresTaskStore();
  
  // Agent handler cache to avoid recreating handlers
  const agentHandlers = new Map<string, express.Application>();
  
  const getAgentHandler = async (agentId: string): Promise<express.Application | null> => {
    if (agentHandlers.has(agentId)) {
      return agentHandlers.get(agentId)!;
    }
    
    const agent = await agentStore.getAgent(agentId);
    if (!agent) {
      return null;
    }
    
    const agentTools = buildToolsFromAgent(agent);
    // Use public URL for agent card so other containers can reach it
    const url = `${publicUrl}/agents/${agent.id}/`;
    const agentCard = createAgentCard(agent, url);
    
    // Create agent-specific task store with agent_id
    const agentTaskStore = new PostgresTaskStore(undefined, agent.id);
    
    const agentExecutor: AgentExecutor = new TimestepAgentExecutor(
      agentTools,
      agent.model,
      agentTaskStore
    );
    
    const requestHandler = new DefaultRequestHandler(
      agentCard,
      agentTaskStore,
      agentExecutor
    );
    
    const appBuilder = new A2AExpressApp(requestHandler);
    const agentApp = appBuilder.setupRoutes(express());
    
    agentHandlers.set(agentId, agentApp);
    return agentApp;
  };
  
  // Create router groups for better organization
  const contextsRouter = createContextsRouter();
  const agentsRouter = createAgentsRouter(agentStore, getAgentHandler, publicUrl);
  
  // Mount routers
  app.use('/contexts', contextsRouter);
  app.use('/agents', agentsRouter);
  
  return app;
}

/**
 * Create router for agent-agnostic context endpoints.
 */
function createContextsRouter(): Router {
  const router = Router();
  
  // GET /contexts - List all contexts, optionally filtered by parent_id
  router.get('/', async (req, res) => {
    try {
      const parentId = req.query.parent_id as string | undefined;
      const taskStore = new PostgresTaskStore();
      const contexts = await taskStore.listContexts(parentId);
      res.json(contexts);
    } catch (error) {
      console.error('Error listing contexts:', error);
      res.status(500).json({ error: 'Failed to list contexts' });
    }
  });

  // GET /contexts/:contextId - Get a single context
  router.get('/:contextId', async (req, res) => {
    try {
      const { contextId } = req.params;
      const taskStore = new PostgresTaskStore();
      const context = await taskStore.getContext(contextId);
      if (!context) {
        return res.status(404).json({ error: 'Context not found' });
      }
      res.json(context);
    } catch (error) {
      console.error('Error getting context:', error);
      res.status(500).json({ error: 'Failed to get context' });
    }
  });

  // PATCH /contexts/:contextId - Update context (e.g., set parent_context_id)
  router.patch('/:contextId', async (req, res) => {
    try {
      const { contextId } = req.params;
      const updates = req.body as { parent_context_id?: string };
      const taskStore = new PostgresTaskStore();
      const context = await taskStore.updateContext(contextId, updates);
      res.json(context);
    } catch (error) {
      console.error('Error updating context:', error);
      if (error instanceof Error && error.message.includes('not found')) {
        return res.status(404).json({ error: error.message });
      }
      res.status(500).json({ error: 'Failed to update context' });
    }
  });

  // GET /contexts/:contextId/messages/:messageId - Get OpenAI formatted message for a messageId
  router.get('/:contextId/messages/:messageId', async (req, res) => {
    try {
      const { contextId, messageId } = req.params;
      const taskStore = new PostgresTaskStore();
      
      // Get all OpenAI messages for the context
      const openaiMessages = await taskStore.getOpenAIMessagesByContextId(contextId);
      
      // Find the message that corresponds to this messageId
      // We need to convert OpenAI messages to A2A format and find the matching one
      const a2aMessages = openaiToA2A(openaiMessages, contextId);
      const matchingMessage = a2aMessages.find(msg => msg.messageId === messageId);
      
      if (!matchingMessage) {
        // Also check task history for A2A messages
        const tasks = await taskStore.loadByContextId(contextId);
        for (const task of tasks) {
          if (task.history) {
            const historyMsg = task.history.find((msg: Message) => msg.messageId === messageId);
            if (historyMsg) {
              // For A2A messages from history, we need to find the corresponding OpenAI message
              // Tool results with toolName won't have direct OpenAI equivalents
              return res.json({ messageId, message: historyMsg });
            }
          }
        }
        return res.status(404).json({ error: 'Message not found' });
      }
      
      // Find the corresponding OpenAI message
      // For assistant messages with tool calls, find the assistant message
      // For tool messages, find the tool message
      // For user/agent messages, find the corresponding message
      let openaiMessage: ChatCompletionMessageParam | null = null;
      
      if (matchingMessage.role === 'agent') {
        const msgWithToolCalls = matchingMessage as Message & { toolCalls?: unknown[]; tool_calls?: unknown[] };
        if (msgWithToolCalls.toolCalls || msgWithToolCalls.tool_calls) {
          // Find assistant message with matching tool calls
          for (const msg of openaiMessages) {
            if (msg.role === 'assistant') {
              const assistantMsg = msg as Extract<ChatCompletionMessageParam, { role: 'assistant' }>;
              if (assistantMsg.tool_calls) {
                // Check if tool calls match
                const a2aToolCalls = (msgWithToolCalls.toolCalls || msgWithToolCalls.tool_calls) as Array<{ id?: string }>;
                const openaiToolCalls = assistantMsg.tool_calls;
                if (a2aToolCalls.length === openaiToolCalls.length && 
                    a2aToolCalls[0]?.id === openaiToolCalls[0]?.id) {
                  openaiMessage = msg;
                  break;
                }
              }
            }
          }
        } else {
          // Regular assistant message - find by content
          const a2aText = matchingMessage.parts?.filter(p => p.kind === 'text').map(p => p.text).join('') || '';
          for (const msg of openaiMessages) {
            if (msg.role === 'assistant') {
              const assistantMsg = msg as Extract<ChatCompletionMessageParam, { role: 'assistant' }>;
              const msgText = typeof assistantMsg.content === 'string' 
                ? assistantMsg.content 
                : assistantMsg.content?.filter(p => p.type === 'text').map(p => p.text).join('') || '';
              if (msgText === a2aText && !assistantMsg.tool_calls) {
                openaiMessage = msg;
                break;
              }
            }
          }
        }
      } else if (matchingMessage.role === 'user') {
        const a2aText = matchingMessage.parts?.filter(p => p.kind === 'text').map(p => p.text).join('') || '';
        for (const msg of openaiMessages) {
          if (msg.role === 'user' && typeof msg.content === 'string' && msg.content === a2aText) {
            openaiMessage = msg;
            break;
          }
        }
      } else if (matchingMessage.role === 'tool') {
        // For tool messages, find the corresponding tool message in OpenAI format
        const msgWithToolName = matchingMessage as Message & { toolName?: string };
        const a2aText = matchingMessage.parts?.filter(p => p.kind === 'text').map(p => p.text).join('') || '';
        for (const msg of openaiMessages) {
          if (msg.role === 'tool') {
            const toolMsg = msg as Extract<ChatCompletionMessageParam, { role: 'tool' }>;
            const toolText = typeof toolMsg.content === 'string' ? toolMsg.content : JSON.stringify(toolMsg.content);
            // Try to match by content or find by tool_call_id
            if (toolText === a2aText || toolText.replace(/^"|"$/g, '') === a2aText) {
              openaiMessage = msg;
              break;
            }
          }
        }
      }
      
      res.json({ messageId, message: openaiMessage || matchingMessage });
    } catch (error) {
      console.error('Error getting message:', error);
      res.status(500).json({ error: 'Failed to get message' });
    }
  });

  // GET /contexts/:contextId/messages - Get messages for a context (agent-agnostic)
  router.get('/:contextId/messages', async (req, res) => {
    try {
      const { contextId } = req.params;
      console.log(`[Server] GET /contexts/${contextId}/messages - Fetching messages for context`);
      const taskStore = new PostgresTaskStore(); // No agent_id - agent-agnostic
      
      // Get OpenAI messages for the context
      const openaiMessagesWithTaskId = await taskStore.getOpenAIMessagesByContextId(contextId);
      console.log(`[Server] Found ${openaiMessagesWithTaskId.length} OpenAI messages for context ${contextId}`);
      
      // Extract taskId from messages (stored as _taskId during loading)
      // Build a map of message index -> taskId
      const messageTaskIds = new Map<number, string>();
      const openaiMessages: ChatCompletionMessageParam[] = [];
      for (let i = 0; i < openaiMessagesWithTaskId.length; i++) {
        const msgWithTaskId = openaiMessagesWithTaskId[i] as ChatCompletionMessageParam & { _taskId?: string };
        if (msgWithTaskId._taskId) {
          messageTaskIds.set(i, msgWithTaskId._taskId);
          console.log(`[Server] Message ${i} has taskId: ${msgWithTaskId._taskId}`);
        }
        // Remove _taskId before using the message
        const { _taskId, ...msg } = msgWithTaskId;
        openaiMessages.push(msg);
      }
      
      // Log tool calls in assistant messages for debugging
      for (let i = 0; i < Math.min(openaiMessages.length, 5); i++) {
        const msg = openaiMessages[i];
        if (msg.role === 'assistant') {
          const assistantMsg = msg as Extract<ChatCompletionMessageParam, { role: 'assistant' }>;
          if (assistantMsg.tool_calls && assistantMsg.tool_calls.length > 0) {
            const toolNames = assistantMsg.tool_calls.map(tc => tc.function.name);
            console.log(`[Server] Context ${contextId} message ${i}: assistant with tool_calls: ${toolNames.join(', ')}`);
          }
        } else if (msg.role === 'tool') {
          const toolMsg = msg as Extract<ChatCompletionMessageParam, { role: 'tool' }>;
          const contentPreview = typeof toolMsg.content === 'string' 
            ? toolMsg.content.substring(0, 50) 
            : JSON.stringify(toolMsg.content).substring(0, 50);
          console.log(`[Server] Context ${contextId} message ${i}: tool message, tool_call_id: ${toolMsg.tool_call_id}, content: ${contentPreview}...`);
        }
      }
      
      // Build a map of tool_call_id -> toolName from assistant messages
      const toolNameByToolCallId = new Map<string, string>();
      for (const msg of openaiMessages) {
        if (msg.role === 'assistant') {
          const assistantMsg = msg as Extract<ChatCompletionMessageParam, { role: 'assistant' }>;
          if (assistantMsg.tool_calls) {
            for (const toolCall of assistantMsg.tool_calls) {
              if (toolCall.function?.name) {
                toolNameByToolCallId.set(toolCall.id, toolCall.function.name);
              }
            }
          }
        }
      }
      
      // Convert OpenAI messages to A2A format - these are already in the correct conversation order
      // We need to pass taskId for each message, but openaiToA2A only accepts a single taskId
      // So we'll convert and then update taskIds afterward
      const a2aMessagesFromOpenAI = openaiToA2A(openaiMessages, contextId);
      console.log(`[Server] Converted ${openaiMessages.length} OpenAI messages to ${a2aMessagesFromOpenAI.length} A2A messages`);
      console.log(`[Server] A2A message roles: ${a2aMessagesFromOpenAI.map(m => m.role).join(', ')}`);
      
      // Update taskIds on A2A messages based on the message index
      // Note: A2A messages might have different length than OpenAI messages (system messages are skipped)
      // So we need to track which OpenAI message each A2A message corresponds to
      let openaiIndex = 0;
      for (let a2aIndex = 0; a2aIndex < a2aMessagesFromOpenAI.length; a2aIndex++) {
        // Find the corresponding OpenAI message (skip system messages)
        while (openaiIndex < openaiMessages.length && openaiMessages[openaiIndex].role === 'system') {
          openaiIndex++;
        }
        if (openaiIndex < openaiMessages.length) {
          const taskId = messageTaskIds.get(openaiIndex);
          if (taskId) {
            a2aMessagesFromOpenAI[a2aIndex].taskId = taskId;
          }
          openaiIndex++;
        }
      }
      
      // Add toolName to tool messages by matching tool_call_id
      for (const msg of a2aMessagesFromOpenAI) {
        if (msg.role === 'tool') {
          const msgWithToolCallId = msg as Message & { tool_call_id?: string };
          console.log(`[Server] Found tool message with tool_call_id: ${msgWithToolCallId.tool_call_id}`);
          if (msgWithToolCallId.tool_call_id) {
            const toolName = toolNameByToolCallId.get(msgWithToolCallId.tool_call_id);
            console.log(`[Server] Matched tool_call_id ${msgWithToolCallId.tool_call_id} to toolName: ${toolName}`);
            if (toolName) {
              (msg as Message & { toolName?: string }).toolName = toolName;
            } else {
              console.warn(`[Server] No toolName found for tool_call_id: ${msgWithToolCallId.tool_call_id}`);
            }
          } else {
            console.warn(`[Server] Tool message missing tool_call_id`);
          }
        }
      }
      
      const allMessages = a2aMessagesFromOpenAI;
      
      // Convert to the format expected by the frontend
      const messages = allMessages.map(msg => {
        const textParts = msg.parts
          ?.filter((p) => p.kind === 'text')
          .map((p) => {
            if (p.kind === 'text') {
              return p.text;
            }
            return '';
          })
          .filter((text) => text.length > 0)
          .join('') || '';
        
        const msgWithToolName = msg as Message & { toolName?: string };
        const msgWithToolCalls = msg as Message & { toolCalls?: unknown[]; tool_calls?: unknown[] };
        const result = {
          role: msg.role,
          content: textParts,
          timestamp: (msg as Message & { timestamp?: string }).timestamp || undefined,
          taskId: msg.taskId || undefined, // Use message taskId from the A2A message - this is the correct taskId for this message
          toolName: msgWithToolName.toolName || undefined,
          messageId: msg.messageId || undefined,
          tool_calls: msgWithToolCalls.tool_calls || msgWithToolCalls.toolCalls || undefined,
        };
        
        // Debug logging for tool messages
        if (msg.role === 'tool') {
          console.log(`[Server] Converting tool message: role=${result.role}, toolName=${result.toolName}, content=${textParts.substring(0, 50)}...`);
        }
        
        return result;
      });
      
      console.log(`[Server] Converted ${allMessages.length} A2A messages to ${messages.length} frontend messages`);
      console.log(`[Server] Message roles: ${messages.map(m => m.role).join(', ')}`);
      
      return res.json({ contextId, messages });
    } catch (error) {
      console.error('Error loading context messages:', error);
      return res.status(500).json({ error: 'Failed to load context messages' });
    }
  });
  
  return router;
}

/**
 * Create router for agent-related endpoints.
 */
function createAgentsRouter(
  agentStore: PostgresAgentStore,
  getAgentHandler: (agentId: string) => Promise<express.Application | null>,
  publicUrl: string
): Router {
  const router = Router();
  
  // GET /agents - List all agents (must be before /agents/:agentId routes)
  router.get('/', async (_req, res) => {
    try {
      console.log('[Server] GET /agents - Listing agents');
      const agents = await agentStore.listAgents();
      console.log('[Server] Found agents:', agents.length);
      res.json(agents);
    } catch (error) {
      console.error('[Server] Error listing agents:', error);
      res.status(500).json({ error: 'Failed to list agents', details: error instanceof Error ? error.message : String(error) });
    }
  });
  
  // GET /agents/:agentId/.well-known/agent-card.json (must be before general /agents/:agentId middleware)
  router.get('/:agentId/.well-known/agent-card.json', async (req, res) => {
    try {
      const { agentId } = req.params;
      const agent = await agentStore.getAgent(agentId);
      
      if (!agent) {
        return res.status(404).json({ error: `Agent not found: ${agentId}` });
      }
      
      // Use public URL for agent card so other containers can reach it
      const url = `${publicUrl}/agents/${agent.id}/`;
      const agentCard = createAgentCard(agent, url);
      res.json(agentCard);
    } catch (error) {
      console.error('Error creating agent card:', error);
      res.status(500).json({ error: 'Failed to create agent card' });
    }
  });
  
  // Agent-scoped context management endpoints (must come before the catch-all /:agentId middleware)
  // GET /agents/:agentId/contexts
  router.get('/:agentId/contexts', async (req, res) => {
    try {
      const { agentId } = req.params;
      const agentTaskStore = new PostgresTaskStore(undefined, agentId);
      const contexts = await agentTaskStore.listContexts();
      res.json(contexts);
    } catch (error) {
      console.error('Error listing contexts:', error);
      res.status(500).json({ error: 'Failed to list contexts' });
    }
  });

  // POST /agents/:agentId/contexts
  router.post('/:agentId/contexts', async (req, res) => {
    try {
      const { agentId } = req.params;
      const agentTaskStore = new PostgresTaskStore(undefined, agentId);
      const context = await agentTaskStore.createContext(req.body.metadata);
      res.json(context);
    } catch (error) {
      console.error('Error creating context:', error);
      res.status(500).json({ error: 'Failed to create context' });
    }
  });

  // DELETE /agents/:agentId/contexts/:contextId
  router.delete('/:agentId/contexts/:contextId', async (req, res) => {
    try {
      const { contextId } = req.params;
      const { agentId } = req.params;
      const agentTaskStore = new PostgresTaskStore(undefined, agentId);
      await agentTaskStore.deleteContext(contextId);
      return res.json({ success: true, contextId });
    } catch (error) {
      console.error('Error deleting context:', error);
      return res.status(500).json({ error: 'Failed to delete context' });
    }
  });

  // GET /agents/:agentId/contexts/:contextId/messages
  router.get('/:agentId/contexts/:contextId/messages', async (req, res) => {
    try {
      const { contextId, agentId } = req.params;
      const agentTaskStore = new PostgresTaskStore(undefined, agentId);
      
      // Get OpenAI messages for the context
      const openaiMessages = await agentTaskStore.getOpenAIMessagesByContextId(contextId);
      
      if (openaiMessages.length === 0) {
        // Fallback to task history if no OpenAI messages stored yet
        const tasks = await agentTaskStore.loadByContextId(contextId);
        const messages: Array<{
          role: string;
          content: string;
          timestamp?: string;
          taskId?: string;
        }> = [];
        
        for (const task of tasks) {
          if (task.history) {
            for (const msg of task.history) {
              const textParts = msg.parts
                ?.filter((p: { kind: string }) => p.kind === 'text')
                .map((p: { kind: string; text?: string }) => {
                  if (p.kind === 'text' && p.text) {
                    return p.text;
                  }
                  return '';
                })
                .filter((text: string) => text.length > 0)
                .join('') || '';
              
              if (textParts) {
                messages.push({
                  role: msg.role,
                  content: textParts,
                  timestamp: (msg as Message & { timestamp?: string }).timestamp || undefined,
                  taskId: task.id,
                });
              }
            }
          }
        }
        
        messages.sort((a, b) => {
          if (a.timestamp && b.timestamp) {
            return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
          }
          return 0;
        });
        
        return res.json({ contextId, messages });
      }
      
      // Convert OpenAI messages to A2A format
      const a2aMessages = openaiToA2A(openaiMessages, contextId);
      
      // Convert to the format expected by the frontend
      const messages = a2aMessages.map(msg => {
        const textParts = msg.parts
          ?.filter((p) => p.kind === 'text')
          .map((p) => {
            if (p.kind === 'text') {
              return p.text;
            }
            return '';
          })
          .filter((text) => text.length > 0)
          .join('') || '';
        
        const msgWithToolName = msg as Message & { toolName?: string };
        return {
          role: msg.role,
          content: textParts,
          timestamp: (msg as Message & { timestamp?: string }).timestamp || undefined,
          taskId: msg.taskId,
          toolName: msgWithToolName.toolName || undefined,
        };
      });
      
      return res.json({ contextId, messages });
    } catch (error) {
      console.error('Error loading context messages:', error);
      return res.status(500).json({ error: 'Failed to load context messages' });
    }
  });
  
  // Agent-scoped A2A routes - strip /agents/:agentId prefix and forward to agent handler
  // This must come after all specific routes
  router.use('/:agentId', async (req, res, next) => {
    try {
      // Skip if this is the /agents route itself (shouldn't happen due to route order, but safety check)
      if (req.path === '/agents' || req.path === '/agents/') {
        return next();
      }
      
      const { agentId } = req.params;
      
      // Skip if agentId is empty or undefined
      if (!agentId) {
        return next();
      }
      
      const agentApp = await getAgentHandler(agentId);
      
      if (!agentApp) {
        return res.status(404).json({ error: `Agent not found: ${agentId}` });
      }
      
      // Store original URL
      const originalUrl = req.url;
      
      // Remove /agents/:agentId prefix from URL
      const newUrl = req.url.replace(`/agents/${agentId}`, '') || '/';
      req.url = newUrl;
      
      // Handle with agent app
      agentApp(req, res, (err?: any) => {
        // Restore original URL
        req.url = originalUrl;
        if (err) {
          next(err);
        } else {
          // Don't call next() if response was already sent
          if (!res.headersSent) {
            next();
          }
        }
      });
    } catch (error) {
      console.error('Error handling agent request:', error);
      res.status(500).json({ error: 'Failed to handle agent request' });
    }
  });
  
  return router;
}

export function runServer(
  host: string = '0.0.0.0',
  port: number = 8080,
  tools?: Array<{ name: string; parameters: z.ZodTypeAny }>,
  model: string = 'gpt-4.1'
): void {
  /** Run the A2A server.
   *
   * @param host - Host address to bind to.
   * @param port - Port number to listen on.
   * @param tools - List of tool objects to use. If undefined, uses default tools.
   * @param model - OpenAI model name to use.
   */
  const app = createServer(host, port, tools, model);
  const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : port;
  
  app.listen(PORT, () => {
    console.log(`[Timestep] Server started on http://${host}:${PORT}`);
    console.log(`[Timestep] Agent Card: http://${host}:${PORT}/.well-known/agent-card.json`);
    console.log('[Timestep] Press Ctrl+C to stop the server');
  });
}

