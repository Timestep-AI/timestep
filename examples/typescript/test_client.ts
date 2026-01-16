/**
 * Test client that orchestrates A2A and MCP servers.
 * Handles the main loop: send message to A2A, forward tool calls to MCP, handle handoff with sampling.
 *
 * ⚠️ STATUS: INCOMPLETE - PENDING MCP SDK v2 RELEASE
 *
 * This implementation is incomplete because it depends on the MCP SDK v2 which is not yet published.
 * The MCP client imports may need to be updated when v2 is released.
 *
 * Expected MCP SDK v2 release: Q1 2026
 *
 * TODO: When v2 is published, verify imports from @modelcontextprotocol/sdk/client
 * work correctly or need to be updated to @modelcontextprotocol/client
 */

import { ClientFactory } from '@a2a-js/sdk/client';
import { Message, Task, TaskStatusUpdateEvent } from '@a2a-js/sdk';
import { v4 as uuidv4 } from 'uuid';
import {
  Client,
  StreamableHTTPClientTransport,
  CallToolResultSchema,
  CreateMessageRequestSchema,
  CreateMessageResultSchema,
  ErrorCode,
  McpError,
} from '@modelcontextprotocol/sdk/client';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

// Server URLs
const A2A_BASE_URL = process.env.A2A_URL || 'http://localhost:8000';
const MCP_URL = process.env.MCP_URL || 'http://localhost:8080/mcp';

// Agent IDs
const PERSONAL_ASSISTANT_ID = '00000000-0000-0000-0000-000000000000';
const WEATHER_ASSISTANT_ID = 'FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF';

function writeTask(task: any, agent_id: string): void {
  /**Write task to tasks/ folder in proper A2A Task format.*/
  const tasks_dir = 'tasks';
  mkdirSync(tasks_dir, { recursive: true });

  const timestamp = new Date().toISOString().replace(/:/g, '-');
  const task_id_short = task.id ? task.id.substring(0, 8) : 'unknown';
  const agent_id_short = agent_id ? agent_id.substring(0, 8) : 'unknown';
  const task_file = join(tasks_dir, `${timestamp}_${task_id_short}_${agent_id_short}.json`);

  writeFileSync(task_file, JSON.stringify(task, null, 2));
  console.error(`\n[Saved task to ${task_file}]`);
}

function extractTaskFromEvent(event: any): any {
  /**Extract Task object from A2A event.*/
  if (event.kind === 'task') {
    return event;
  } else if (event.kind === 'status-update') {
    // Status update events contain task information
    return event;
  } else if (event && typeof event === 'object' && 'id' in event && 'status' in event) {
    return event;
  } else {
    throw new Error(`Received non-Task event from Task-generating agent: ${typeof event}`);
  }
}

function extractFinalMessage(task: any): string {
  /**Extract final message text from a completed task.*/
  let message_text = '';

  // Extract from task.status.message if available
  if (task.status?.message?.parts) {
    for (const part of task.status.message.parts) {
      if (part.kind === 'text') {
        message_text += part.text;
      }
    }
  }

  // Also check task history for agent messages
  if (task.history) {
    for (const msg of task.history) {
      if (msg.role === 'agent') {
        if (msg.parts) {
          for (const part of msg.parts) {
            if (part.kind === 'text') {
              message_text += part.text;
            }
          }
        }
      }
    }
  }

  return message_text.trim();
}

function extractToolCalls(task: any): any[] | null {
  /**Extract tool calls from task status message or history.*/
  // Check task.status.message parts first
  if (task.status?.message?.parts) {
    for (const part of task.status.message.parts) {
      if (part.kind === 'data' && part.data && typeof part.data === 'object') {
        const tool_calls = (part.data as any).tool_calls;
        if (tool_calls) {
          return tool_calls;
        }
      }
    }
  }

  // Fallback: check last agent message in history
  if (task.history) {
    for (let i = task.history.length - 1; i >= 0; i--) {
      const msg = task.history[i];
      if (msg.role === 'agent') {
        if (msg.parts) {
          for (const part of msg.parts) {
            if (part.kind === 'data' && part.data && typeof part.data === 'object') {
              const tool_calls = (part.data as any).tool_calls;
              if (tool_calls) {
                return tool_calls;
              }
            }
          }
        }
      }
    }
  }

  return null;
}

function parseToolCall(tool_call: Record<string, any>): [string | null, Record<string, any>] {
  /**Parse tool call dict to extract tool name and arguments.*/
  const tool_name = tool_call?.function?.name || null;
  const tool_args_str = tool_call?.function?.arguments;

  let tool_args: Record<string, any> = {};
  try {
    tool_args = typeof tool_args_str === 'string' ? JSON.parse(tool_args_str) : tool_args_str || {};
  } catch {
    tool_args = {};
  }

  return [tool_name, tool_args];
}

async function executeToolCalls(tool_calls: Record<string, any>[]): Promise<Array<Record<string, any>>> {
  /**Execute tool calls concurrently and return structured results.*/
  return Promise.all(
    tool_calls.map(async (tool_call) => {
      const [tool_name, tool_args] = parseToolCall(tool_call);
      const tool_call_id = tool_call?.id;
      try {
        const result = await callMcpTool(tool_name || '', tool_args);
        return { tool_call_id: tool_call_id, name: tool_name, result: result };
      } catch (e: any) {
        return { tool_call_id: tool_call_id, name: tool_name, result: { error: String(e) } };
      }
    })
  );
}

function buildToolResultMessage(tool_results: Array<Record<string, any>>, task_id?: string, context_id?: string): Message {
  /**Build an A2A message containing tool results as DataPart.*/
  return {
    kind: 'message',
    role: 'user',
    messageId: uuidv4(),
    parts: [{ kind: 'data', data: { tool_results: tool_results } }],
    taskId: task_id,
    contextId: context_id,
  };
}

// MCP client for calling tools
let mcpClient: Client | null = null;
let mcpTransport: StreamableHTTPClientTransport | null = null;

async function initializeMcpClient(): Promise<void> {
  if (mcpClient && mcpTransport) {
    return; // Already initialized
  }

  mcpClient = new Client(
    {
      name: 'test-client',
      version: '1.0.0',
    },
    {
      capabilities: {
        sampling: {},
      },
    }
  );

  // Set up sampling callback for handoff
  mcpClient.setRequestHandler(CreateMessageRequestSchema, async (request) => {
    const agent_uri = request.params.metadata?.agent_uri;
    if (!agent_uri) {
      throw new McpError(ErrorCode.InvalidParams, 'agent_uri is required for sampling');
    }

    const message_text =
      request.params.messages?.[0]?.content && typeof request.params.messages[0].content === 'object' && 'text' in request.params.messages[0].content
        ? (request.params.messages[0].content as any).text
        : 'Please help with this task.';

    const result_text = await handleAgentHandoff(agent_uri, message_text);

    return {
      role: 'assistant',
      content: {
        type: 'text',
        text: result_text,
      },
      model: 'a2a-agent',
    };
  });

  mcpTransport = new StreamableHTTPClientTransport(new URL(MCP_URL));
  await mcpClient.connect(mcpTransport);
}

async function callMcpTool(tool_name: string, arguments_: Record<string, any>): Promise<Record<string, any>> {
  /**Call MCP tool using MCP TypeScript SDK client.*/
  if (!mcpClient) {
    await initializeMcpClient();
  }

  if (!mcpClient) {
    return { error: 'Failed to initialize MCP client' };
  }

  try {
    const result = await mcpClient.request(
      {
        method: 'tools/call',
        params: {
          name: tool_name,
          arguments: arguments_,
        },
      },
      CallToolResultSchema
    );

    // Extract text from result content
    const text_parts = result.content
      .filter((item) => item.type === 'text')
      .map((item) => (item as any).text);
    return text_parts.length > 0 ? { result: text_parts.join(' ') } : { result: null };
  } catch (e: any) {
    return { error: String(e) };
  }
}

function extractAgentIdFromUri(agent_uri: string): string {
  /**Extract agent_id from agent_uri for use in write_task.*/
  const match = agent_uri.match(/\/agents\/([^/\s]+)/);
  return match ? match[1] : 'unknown';
}

async function processMessageStream(
  a2a_client: any,
  message_obj: any,
  agent_id: string,
  task_id?: string,
  context_id?: string
): Promise<string> {
  /**Process a message stream, handling tool calls recursively.*/
  let final_message = '';

  for await (const event of a2a_client.sendMessageStream(message_obj)) {
    const task = extractTaskFromEvent(event);
    writeTask(task, agent_id);

    const current_task_id = task.id || task.taskId || task_id;
    const current_context_id = task.contextId || context_id;

    if (task.kind === 'status-update' && task.status?.state === 'completed') {
      final_message = extractFinalMessage(task);
      break;
    }

    if (task.kind === 'status-update' && task.status?.state === 'input-required') {
      const tool_calls = extractToolCalls(task);
      if (tool_calls) {
        const tool_results = await executeToolCalls(tool_calls);
        const tool_result_msg = buildToolResultMessage(tool_results, current_task_id, current_context_id);

        // Recursively process tool result stream
        const result_message = await processMessageStream(
          a2a_client,
          tool_result_msg,
          agent_id,
          current_task_id,
          current_context_id
        );
        if (result_message) {
          final_message = result_message;
        }
      }
    }
    if (final_message) {
      break;
    }
  }

  return final_message.trim() || 'Task completed.';
}

async function handleAgentHandoff(agent_uri: string, message: string): Promise<string> {
  /**Handle agent handoff by calling the A2A agent at agent_uri.*/
  const factory = new ClientFactory();
  const a2a_client = await factory.createFromUrl(agent_uri);

  try {
    const message_obj: Message = {
      kind: 'message',
      role: 'user',
      messageId: uuidv4(),
      parts: [{ kind: 'text', text: message }],
    };
    const agent_id = extractAgentIdFromUri(agent_uri);
    return await processMessageStream(a2a_client, message_obj, agent_id);
  } finally {
    // Note: Client cleanup would be handled by the SDK
  }
}

async function runClientLoop(initial_message: string, agent_id: string = PERSONAL_ASSISTANT_ID): Promise<void> {
  /**Main client loop that orchestrates A2A and MCP (fully async).*/

  // Construct A2A URL with agent path
  const agent_url = `${A2A_BASE_URL}/agents/${agent_id}`;

  // Track all task IDs encountered
  const task_ids: string[] = [];

  // Create A2A client using ClientFactory
  const factory = new ClientFactory();
  const a2a_client = await factory.createFromUrl(agent_url);

  try {
    const message: Message = {
      kind: 'message',
      role: 'user',
      messageId: uuidv4(),
      parts: [{ kind: 'text', text: initial_message }],
    };
    console.error('\n[DEBUG: Starting to send message to A2A server]');

    async function processWithOutput(a2a_client: any, message_obj: any, agent_id: string): Promise<void> {
      /**Process message stream and print output.*/
      for await (const event of a2a_client.sendMessageStream(message_obj)) {
        const task = extractTaskFromEvent(event);
        console.error(`\n[DEBUG: Received task, id=${task.id || task.taskId || 'NO_ID'}, type=${task.kind || typeof task}]`);
        writeTask(task, agent_id);

        const taskId = task.id || task.taskId;
        if (taskId && !task_ids.includes(taskId)) {
          task_ids.push(taskId);
        }

        // Print agent messages
        if (task.history) {
          for (const msg of task.history) {
            if (msg.role === 'agent') {
              if (msg.parts) {
                for (const part of msg.parts) {
                  if (part.kind === 'text') {
                    process.stdout.write(part.text);
                  }
                }
              }
            }
          }
        }

        if (task.kind === 'status-update' && task.status?.state === 'completed') {
          console.log('\n[Task completed]');
          break;
        }

        if (task.kind === 'status-update' && task.status?.state === 'input-required') {
          const tool_calls = extractToolCalls(task);
          if (tool_calls) {
            const tool_results = await executeToolCalls(tool_calls);
            for (const tool_result of tool_results) {
              const tool_name = tool_result.name;
              console.log(`\n[Calling tool: ${tool_name}]`);
              if (tool_name !== 'handoff') {
                console.log(`[Tool result: ${JSON.stringify(tool_result.result)}]`);
              }
            }

            const tool_result_msg = buildToolResultMessage(tool_results, task.id || task.taskId, task.contextId);

            // Recursively process tool result
            await processWithOutput(a2a_client, tool_result_msg, agent_id);
            break;
          }
        }
      }
    }

    await processWithOutput(a2a_client, message, agent_id);
  } catch (e: any) {
    console.error(`\n[Error in client loop: ${e}]`);
    throw e;
  } finally {
    // Cleanup
    if (mcpTransport) {
      try {
        await mcpTransport.close();
      } catch {
        // Ignore errors during cleanup
      }
    }
  }
}

async function main(): Promise<void> {
  /**Main entry point.*/
  const args = process.argv.slice(2);
  if (args.length < 1) {
    console.error('Usage: bun run lib/typescript/test_client.ts <message>');
    process.exit(1);
  }

  const message = args.join(' ');
  await runClientLoop(message, PERSONAL_ASSISTANT_ID);
}

if (import.meta.main) {
  main().catch(console.error);
}
