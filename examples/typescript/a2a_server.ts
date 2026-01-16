/**
 * A2A Server using a2a-js SDK with Express.
 * Handles task creation and continuation via A2A protocol.
 *
 * ⚠️ STATUS: MAY NEED UPDATES WHEN MCP SDK v2 IS RELEASED
 *
 * This file uses the A2A SDK which is already published and working.
 * However, it may need minor updates when the MCP SDK v2 is released
 * if there are any integration changes needed.
 *
 * Expected MCP SDK v2 release: Q1 2026
 */

import express from 'express';
import OpenAI from 'openai';
import { v4 as uuidv4 } from 'uuid';
import { AgentCard, AGENT_CARD_PATH } from '@a2a-js/sdk';
import {
  InMemoryTaskStore,
  TaskStore,
  AgentExecutor,
  DefaultRequestHandler,
  RequestContext,
  ExecutionEventBus,
  TaskStatusUpdateEvent,
} from '@a2a-js/sdk/server';
import { agentCardHandler, jsonRpcHandler, UserBuilder } from '@a2a-js/sdk/server/express';
import { Message, Task } from '@a2a-js/sdk';

// Initialize OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Agent IDs
const PERSONAL_ASSISTANT_ID = '00000000-0000-0000-0000-000000000000';
const WEATHER_ASSISTANT_ID = 'FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF';

// Tool definitions (OpenAI function calling format)
const HANDOFF_TOOL = {
  type: 'function',
  function: {
    name: 'handoff',
    description: 'Hand off to another agent to handle a specific task',
    parameters: {
      type: 'object',
      properties: {
        agent_uri: {
          type: 'string',
          description: 'The A2A URI of the agent to hand off to',
        },
        context_id: {
          type: 'string',
          description: 'Optional context ID for the handoff',
        },
        message: {
          type: 'string',
          description: 'The message to send to the other agent',
        },
      },
      required: ['agent_uri', 'message'],
    },
  },
};

const GET_WEATHER_TOOL = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Get the current weather for a specific location',
    parameters: {
      type: 'object',
      properties: {
        location: {
          type: 'string',
          description: 'The location to get weather for (e.g., city name)',
        },
      },
      required: ['location'],
    },
  },
};

// Agent registry mapping agent IDs to tool configurations
const AGENT_TOOLS: Record<string, any[]> = {
  [PERSONAL_ASSISTANT_ID]: [HANDOFF_TOOL],
  [WEATHER_ASSISTANT_ID]: [GET_WEATHER_TOOL],
};

// Agent descriptions
const AGENT_DESCRIPTIONS: Record<string, string> = {
  [PERSONAL_ASSISTANT_ID]: 'Personal Assistant',
  [WEATHER_ASSISTANT_ID]: 'Weather Assistant',
};

function buildSystemMessage(agentId: string, tools: any[]): string {
  /**Build system message explaining who the agent is and what tools are available.*/
  const agentName = AGENT_DESCRIPTIONS[agentId] || 'Assistant';
  const baseUrl = process.env.A2A_BASE_URL || 'http://localhost:8000';
  const systemParts: string[] = [`You are a ${agentName}.`];

  if (tools.length > 0) {
    systemParts.push('\nYou have access to the following tools:');
    for (const tool of tools) {
      const func = tool.function || {};
      const toolName = func.name || 'unknown';
      const toolDesc = func.description || 'No description available';
      const params = func.parameters || {};

      let toolInfo = `- ${toolName}: ${toolDesc}`;

      if (params && params.properties) {
        toolInfo += '\n  Parameters:';
        const properties = params.properties;
        const required = params.required || [];
        for (const [paramName, paramSpec] of Object.entries(properties)) {
          const spec = paramSpec as any;
          const paramType = spec.type || 'string';
          const paramDesc = spec.description || '';
          const requiredMarker = required.includes(paramName) ? ' (required)' : ' (optional)';
          toolInfo += `\n    - ${paramName} (${paramType})${requiredMarker}: ${paramDesc}`;
        }
      }

      if (toolName === 'handoff') {
        toolInfo += '\n  Available agents for handoff:';
        for (const [otherAgentId, otherAgentName] of Object.entries(AGENT_DESCRIPTIONS)) {
          if (otherAgentId !== agentId) {
            const agentUri = `${baseUrl}/agents/${otherAgentId}`;
            toolInfo += `\n    - ${otherAgentName} (ID: ${otherAgentId.substring(0, 8)}...): Use agent_uri="${agentUri}"`;
          }
        }
      }

      systemParts.push(toolInfo);
    }
  } else {
    systemParts.push('\nYou do not have access to any tools.');
  }

  return systemParts.join('\n');
}

// Simple in-memory task storage (messages per task, keyed by agent_id:task_id)
const taskMessages: Record<string, any[]> = {};

// Track all task IDs per agent for listing
const agentTaskIds: Record<string, string[]> = {};

function writeTrace(
  taskId: string,
  agentId: string,
  inputMessages: any[],
  inputTools: any[],
  outputMessage: any
): void {
  /**Write trace to traces/ folder.*/
  // Implementation would write to traces/ folder
  // Similar to Python version
}

class MultiAgentExecutor implements AgentExecutor {
  /**Agent executor that uses OpenAI directly and configures tools based on agent_id.*/
  agent_id: string;
  model: string;
  tools: any[];

  constructor(agent_id: string, model: string = 'gpt-4o-mini') {
    this.agent_id = agent_id;
    this.model = model;
    this.tools = AGENT_TOOLS[agent_id] || [];
  }

  async execute(requestContext: RequestContext, eventBus: ExecutionEventBus): Promise<void> {
    /**Execute agent task using OpenAI.*/
    const taskId = requestContext.taskId;
    const contextId = requestContext.contextId;
    const userMessage = requestContext.userMessage;

    // Get messages for this task (or initialize empty)
    const task_key = `${this.agent_id}:${taskId}`;
    if (!(task_key in taskMessages)) {
      taskMessages[task_key] = [];
      if (!(this.agent_id in agentTaskIds)) {
        agentTaskIds[this.agent_id] = [];
      }
      if (taskId && !agentTaskIds[this.agent_id].includes(taskId)) {
        agentTaskIds[this.agent_id].push(taskId);
      }
    }

    const messages = taskMessages[task_key];

    // Extract text from incoming message for OpenAI processing
    if (userMessage) {
      let text_content = '';
      const tool_results: any[] = [];
      if (userMessage.parts) {
        for (const part of userMessage.parts) {
          if (part.kind === 'text') {
            text_content += part.text;
          } else if (part.kind === 'data' && part.data && typeof part.data === 'object') {
            const data = part.data as any;
            if (Array.isArray(data.tool_results)) {
              tool_results.push(...data.tool_results);
            }
          }
        }
      }

      if (text_content) {
        messages.push({ role: 'user', content: text_content });
      }

      if (tool_results.length > 0) {
        for (const tool_result of tool_results) {
          const tool_call_id = tool_result.tool_call_id || tool_result.id;
          const raw_result =
            tool_result.result ?? tool_result.content ?? tool_result;
          const content =
            raw_result && typeof raw_result === 'object' ? JSON.stringify(raw_result) : String(raw_result ?? '');
          messages.push({
            role: 'tool',
            tool_call_id: tool_call_id,
            content: content,
          });
        }
      }
    }

    // Convert messages to OpenAI format
    const openai_messages: any[] = [];
    let pending_tool_calls: any[] = [];
    let pending_tool_index = 0;

    for (const msg of messages) {
      const role = msg.role || 'user';
      const content = msg.content || '';

      if (role === 'assistant') {
        const openai_msg: any = { role: role, content: content };
        if (msg.tool_calls) {
          openai_msg.tool_calls = msg.tool_calls;
          pending_tool_calls = msg.tool_calls || [];
          pending_tool_index = 0;
        } else {
          pending_tool_calls = [];
          pending_tool_index = 0;
        }
        openai_messages.push(openai_msg);
        continue;
      }

      if (role === 'tool') {
        let tool_call_id = msg.tool_call_id;
        if (!tool_call_id && pending_tool_index < pending_tool_calls.length) {
          tool_call_id = pending_tool_calls[pending_tool_index]?.id;
          pending_tool_index += 1;
          if (pending_tool_index >= pending_tool_calls.length) {
            pending_tool_calls = [];
          }
        }
        openai_messages.push({
          role: 'tool',
          tool_call_id: tool_call_id,
          content: content,
        });
        continue;
      }

      if (pending_tool_calls.length > 0 && pending_tool_index < pending_tool_calls.length) {
        const tool_call_id = msg.tool_call_id || pending_tool_calls[pending_tool_index]?.id;
        pending_tool_index += 1;
        if (pending_tool_index >= pending_tool_calls.length) {
          pending_tool_calls = [];
        }
        openai_messages.push({
          role: 'tool',
          tool_call_id: tool_call_id,
          content: content,
        });
        continue;
      }

      openai_messages.push({ role: role, content: content });
    }

    if (openai_messages.length === 0) {
      throw new Error(
        `No messages to send to OpenAI. Task: ${taskId}, Agent: ${this.agent_id}, Messages count: ${messages.length}`
      );
    }

    const system_message_content = buildSystemMessage(this.agent_id, this.tools || []);
    const openai_messages_with_system = [{ role: 'system', content: system_message_content }, ...openai_messages];

    try {
      const request_params: any = {
        model: this.model,
        messages: openai_messages_with_system,
        temperature: 0.0,
      };

      if (this.tools.length > 0) {
        request_params.tools = this.tools;
        request_params.tool_choice = 'auto';
      }

      const response = await openai.chat.completions.create(request_params);
      const assistant_message = response.choices[0].message;
      const tool_calls = assistant_message.tool_calls || [];
      const assistant_content = assistant_message.content || '';

      const output_message_dict = {
        content: assistant_content,
        tool_calls: tool_calls.map((tc: any) => ({
          id: tc.id,
          type: 'function',
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments,
          },
        })),
      };

      writeTrace(taskId || '', this.agent_id, openai_messages_with_system, this.tools || [], output_message_dict);

      // Build A2A message
      const agentMessage: Message = {
        kind: 'message',
        role: 'agent',
        messageId: uuidv4(),
        parts: [{ kind: 'text', text: assistant_content }],
        taskId: taskId,
        contextId: contextId,
      };

      // Add tool calls as data parts if present
      if (tool_calls.length > 0) {
        const tool_calls_data = {
          tool_calls: tool_calls.map((tc: any) => ({
            id: tc.id,
            type: 'function',
            function: {
              name: tc.function.name,
              arguments: tc.function.arguments,
            },
          })),
        };
        agentMessage.parts.push({ kind: 'data', data: tool_calls_data });
      }

      // Publish status update
      if (tool_calls.length > 0) {
        const statusUpdate: TaskStatusUpdateEvent = {
          kind: 'status-update',
          taskId: taskId || '',
          contextId: contextId || '',
          status: {
            state: 'input-required',
            message: agentMessage,
            timestamp: new Date().toISOString(),
          },
          final: false,
        };
        eventBus.publish(statusUpdate);
      } else {
        const statusUpdate: TaskStatusUpdateEvent = {
          kind: 'status-update',
          taskId: taskId || '',
          contextId: contextId || '',
          status: {
            state: 'completed',
            message: agentMessage,
            timestamp: new Date().toISOString(),
          },
          final: true,
        };
        eventBus.publish(statusUpdate);
      }

      messages.push({
        role: 'assistant',
        content: assistant_content,
        tool_calls: tool_calls.map((tc: any) => ({
          id: tc.id,
          type: 'function',
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments,
          },
        })),
      });
    } catch (e: any) {
      const statusUpdate: TaskStatusUpdateEvent = {
        kind: 'status-update',
        taskId: taskId || '',
        contextId: contextId || '',
        status: {
          state: 'failed',
          message: {
            kind: 'message',
            role: 'agent',
            messageId: uuidv4(),
            parts: [{ kind: 'text', text: String(e) }],
            taskId: taskId,
            contextId: contextId,
          },
          timestamp: new Date().toISOString(),
        },
        final: true,
      };
      eventBus.publish(statusUpdate);
      throw e;
    }
  }

  async cancelTask(_taskId: string, eventBus: ExecutionEventBus): Promise<void> {
    /**Cancel an ongoing task.*/
    const statusUpdate: TaskStatusUpdateEvent = {
      kind: 'status-update',
      taskId: _taskId,
      contextId: '',
      status: {
        state: 'canceled',
        timestamp: new Date().toISOString(),
      },
      final: true,
    };
    eventBus.publish(statusUpdate);
    eventBus.finished();
  }
}

function createAgentCard(agent_id: string, agent_name: string, description: string): AgentCard {
  /**Create an agent card for a specific agent.*/
  const base_url = process.env.A2A_BASE_URL || 'http://localhost:8000';
  return {
    name: agent_name,
    version: '1.0.0',
    description: description,
    url: `${base_url}/agents/${agent_id}`,
    protocolVersion: '0.3.0',
    preferredTransport: 'http_json',
    defaultInputModes: ['text/plain'],
    defaultOutputModes: ['text/plain'],
    capabilities: {
      streaming: true,
      pushNotifications: false,
      stateTransitionHistory: true,
    },
    skills: [
      {
        id: agent_id,
        name: agent_name,
        description: description,
        tags: [],
      },
    ],
    supportsAuthenticatedExtendedCard: false,
  };
}

// Create base Express app
const app = express();

// Agent routing: map agent_id to executor and handler
const agent_handlers: Record<string, DefaultRequestHandler> = {};

function getOrCreateHandler(agent_id: string): DefaultRequestHandler {
  /**Get or create a request handler for an agent.*/
  if (!(agent_id in agent_handlers)) {
    if (!(agent_id in AGENT_TOOLS)) {
      throw new Error(`Agent ${agent_id} not found`);
    }

    const executor = new MultiAgentExecutor(agent_id);
    const task_store = new InMemoryTaskStore();
    const handler = new DefaultRequestHandler(
      createAgentCard(agent_id, AGENT_DESCRIPTIONS[agent_id] || 'Agent', 'Agent description'),
      task_store,
      executor
    );
    agent_handlers[agent_id] = handler;
    return handler;
  }
  return agent_handlers[agent_id];
}

// Create A2A apps for routing
const personal_assistant_card = createAgentCard(
  PERSONAL_ASSISTANT_ID,
  'personal-assistant',
  'Personal Assistant agent with handoff capability'
);
const weather_assistant_card = createAgentCard(
  WEATHER_ASSISTANT_ID,
  'weather-assistant',
  'Weather Assistant agent with weather lookup capability'
);

// Create handlers for each agent
const personal_handler = getOrCreateHandler(PERSONAL_ASSISTANT_ID);
const weather_handler = getOrCreateHandler(WEATHER_ASSISTANT_ID);

// Add agent card endpoints
app.get(`/agents/:agent_id/${AGENT_CARD_PATH}`, async (req, res) => {
  const agent_id = req.params.agent_id;
  let card: AgentCard;
  if (agent_id === PERSONAL_ASSISTANT_ID) {
    card = personal_assistant_card;
  } else if (agent_id === WEATHER_ASSISTANT_ID) {
    card = weather_assistant_card;
  } else {
    res.status(404).json({ detail: `Agent ${agent_id} not found` });
    return;
  }
  res.json(card);
});

// Mount handlers for each agent
app.use(`/agents/${PERSONAL_ASSISTANT_ID}`, agentCardHandler({ agentCardProvider: personal_handler }));
app.use(`/agents/${PERSONAL_ASSISTANT_ID}`, jsonRpcHandler({ requestHandler: personal_handler, userBuilder: UserBuilder.noAuthentication }));

app.use(`/agents/${WEATHER_ASSISTANT_ID}`, agentCardHandler({ agentCardProvider: weather_handler }));
app.use(`/agents/${WEATHER_ASSISTANT_ID}`, jsonRpcHandler({ requestHandler: weather_handler, userBuilder: UserBuilder.noAuthentication }));

app.get('/agents/:agent_id/v1/tasks', async (req, res) => {
  const agent_id = req.params.agent_id;
  if (!(agent_id in AGENT_TOOLS)) {
    res.status(404).json({ detail: `Agent ${agent_id} not found` });
    return;
  }
  const task_ids = agentTaskIds[agent_id] || [];
  res.json({ task_ids, count: task_ids.length });
});

if (import.meta.main) {
  const port = parseInt(process.env.A2A_PORT || '8000', 10);
  app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
  });
}
