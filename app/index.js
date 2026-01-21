import { ClientFactory } from '@a2a-js/sdk/client';
import { Message } from '@a2a-js/sdk';
import {
  Client,
  StreamableHTTPClientTransport,
  CreateMessageRequestSchema,
  ErrorCode,
  McpError,
} from '@modelcontextprotocol/sdk/client';

// Server URLs
const A2A_BASE_URL = window.A2A_URL || 'http://localhost:8000';
const MCP_URL = window.MCP_URL || 'http://localhost:8080/mcp';

// Agent IDs
const PERSONAL_ASSISTANT_ID = '00000000-0000-0000-0000-000000000000';
const WEATHER_ASSISTANT_ID = 'FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF';

// Generate UUID v4
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// Helper functions (mirroring test_client.py)
function extractTaskFromEvent(event) {
  if (event.kind === 'task') {
    return event;
  } else if (event.kind === 'status-update') {
    return event;
  } else if (event && typeof event === 'object' && 'id' in event && 'status' in event) {
    return event;
  } else {
    throw new Error(`Received non-Task event: ${typeof event}`);
  }
}

function extractFinalMessage(task) {
  let messageText = '';

  // Extract from task.status.message if available
  if (task.status?.message?.parts) {
    for (const part of task.status.message.parts) {
      if (part.kind === 'text') {
        messageText += part.text || '';
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
              messageText += part.text || '';
            }
          }
        }
      }
    }
  }

  return messageText.trim();
}

function extractToolCalls(task) {
  // Check task.status.message parts first
  if (task.status?.message?.parts) {
    for (const part of task.status.message.parts) {
      if (part.kind === 'data' && part.data && typeof part.data === 'object') {
        const toolCalls = part.data.tool_calls;
        if (toolCalls) {
          return toolCalls;
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
              const toolCalls = part.data.tool_calls;
              if (toolCalls) {
                return toolCalls;
              }
            }
          }
        }
      }
    }
  }

  return null;
}

function parseToolCall(toolCall) {
  const toolName = toolCall?.function?.name || null;
  const toolArgsStr = toolCall?.function?.arguments;

  let toolArgs = {};
  try {
    toolArgs = typeof toolArgsStr === 'string' ? JSON.parse(toolArgsStr) : toolArgsStr || {};
  } catch {
    toolArgs = {};
  }

  return [toolName, toolArgs];
}

function extractAgentIdFromUri(agentUri) {
  const match = agentUri.match(/\/agents\/([^/\s]+)/);
  return match ? match[1] : 'unknown';
}

// MCP client for calling tools
let mcpClient = null;
let mcpTransport = null;

async function initializeMcpClient(chatInterface) {
  if (mcpClient && mcpTransport) {
    return; // Already initialized
  }

  mcpClient = new Client(
    {
      name: 'web-client',
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
    const agentUri = request.params.metadata?.agent_uri;
    if (!agentUri) {
      throw new McpError(ErrorCode.InvalidParams, 'agent_uri is required for sampling');
    }

    const messageText =
      request.params.messages?.[0]?.content && 
      typeof request.params.messages[0].content === 'object' && 
      'text' in request.params.messages[0].content
        ? request.params.messages[0].content.text
        : 'Please help with this task.';

    const resultText = await handleAgentHandoff(agentUri, messageText, chatInterface);

    return {
      role: 'assistant',
      content: {
        type: 'text',
        text: resultText,
      },
      model: 'a2a-agent',
    };
  });

  mcpTransport = new StreamableHTTPClientTransport(new URL(MCP_URL));
  await mcpClient.connect(mcpTransport);
}

async function callMcpTool(toolName, arguments_, chatInterface) {
  if (!mcpClient) {
    await initializeMcpClient(chatInterface);
  }

  if (!mcpClient) {
    return { error: 'Failed to initialize MCP client' };
  }

  try {
    // Call MCP tool - the SDK handles the request/response format
    const result = await mcpClient.request({
      method: 'tools/call',
      params: {
        name: toolName,
        arguments: arguments_,
      },
    });

    // Extract text from result content
    // Result should have a content array with text items
    if (result && result.content && Array.isArray(result.content)) {
      const textParts = result.content
        .filter((item) => item.type === 'text')
        .map((item) => item.text || '');
      return textParts.length > 0 ? { result: textParts.join(' ') } : { result: null };
    }
    return { result: null };
  } catch (e) {
    return { error: String(e) };
  }
}

async function processMessageStream(a2aClient, messageObj, agentId, chatInterface, taskId, contextId) {
  let finalMessage = '';
  let currentMessageElement = null;

  try {
    for await (const event of a2aClient.sendMessageStream(messageObj)) {
      const task = extractTaskFromEvent(event);

      const currentTaskId = task.id || task.taskId || taskId;
      const currentContextId = task.contextId || contextId;

      // Display agent messages as they stream
      if (task.history) {
        for (const msg of task.history) {
          if (msg.role === 'agent') {
            if (msg.parts) {
              for (const part of msg.parts) {
                if (part.kind === 'text' && part.text) {
                  if (!currentMessageElement) {
                    currentMessageElement = chatInterface.appendMessage('agent', '');
                  }
                  currentMessageElement.textContent += part.text;
                  chatInterface.scrollToBottom();
                }
              }
            }
          }
        }
      }

      if (task.kind === 'status-update' && task.status?.state === 'completed') {
        finalMessage = extractFinalMessage(task);
        if (currentMessageElement) {
          currentMessageElement.classList.remove('streaming');
        }
        break;
      }

      if (task.kind === 'status-update' && task.status?.state === 'input-required') {
        const toolCalls = extractToolCalls(task);
        if (toolCalls) {
          if (currentMessageElement) {
            currentMessageElement.classList.remove('streaming');
          }
          for (const toolCall of toolCalls) {
            const [toolName, toolArgs] = parseToolCall(toolCall);
            
            chatInterface.appendMessage('system', `[Calling tool: ${toolName}]`);
            
            const result = await callMcpTool(toolName || '', toolArgs, chatInterface);
            
            if (toolName !== 'handoff') {
              chatInterface.appendMessage('system', `[Tool result: ${JSON.stringify(result)}]`);
            }

            const toolResultMsg = {
              kind: 'message',
              role: 'user',
              messageId: uuidv4(),
              parts: [{ kind: 'text', text: JSON.stringify(result) }],
              taskId: currentTaskId,
              contextId: currentContextId,
            };

            // Recursively process tool result stream
            currentMessageElement = null;
            const resultMessage = await processMessageStream(
              a2aClient,
              toolResultMsg,
              agentId,
              chatInterface,
              currentTaskId,
              currentContextId
            );
            if (resultMessage) {
              finalMessage = resultMessage;
              break;
            }
          }
        }
      }
      if (finalMessage) {
        break;
      }
    }
  } catch (error) {
    if (currentMessageElement) {
      currentMessageElement.classList.remove('streaming');
    }
    throw error;
  }

  return finalMessage.trim() || 'Task completed.';
}

async function handleAgentHandoff(agentUri, message, chatInterface) {
  const factory = new ClientFactory();
  const a2aClient = await factory.createFromUrl(agentUri);

  try {
    const messageObj = {
      kind: 'message',
      role: 'user',
      messageId: uuidv4(),
      parts: [{ kind: 'text', text: message }],
    };
    const agentId = extractAgentIdFromUri(agentUri);
    return await processMessageStream(a2aClient, messageObj, agentId, chatInterface);
  } finally {
    // Client cleanup handled by SDK
  }
}

// Web Component
class ChatInterface extends HTMLElement {
  constructor() {
    super();
    this.messagesContainer = null;
    this.input = null;
    this.sendButton = null;
    this.isProcessing = false;
  }

  connectedCallback() {
    this.innerHTML = `
      <div class="messages"></div>
      <div class="input-area">
        <input type="text" placeholder="Type your message..." />
        <button>Send</button>
      </div>
    `;

    this.messagesContainer = this.querySelector('.messages');
    this.input = this.querySelector('input');
    this.sendButton = this.querySelector('button');

    this.sendButton.addEventListener('click', () => this.handleSend());
    this.input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleSend();
      }
    });
  }

  async handleSend() {
    const message = this.input.value.trim();
    if (!message || this.isProcessing) return;

    this.input.value = '';
    this.input.disabled = true;
    this.sendButton.disabled = true;
    this.isProcessing = true;

    // Display user message
    this.appendMessage('user', message);

    try {
      await this.runClientLoop(message);
    } catch (error) {
      this.appendMessage('error', `Error: ${error.message}`);
      console.error('Error in client loop:', error);
    } finally {
      this.input.disabled = false;
      this.sendButton.disabled = false;
      this.isProcessing = false;
      this.input.focus();
    }
  }

  async runClientLoop(initialMessage, agentId = PERSONAL_ASSISTANT_ID) {
    const agentUrl = `${A2A_BASE_URL}/agents/${agentId}`;

    const factory = new ClientFactory();
    const a2aClient = await factory.createFromUrl(agentUrl);

    try {
      const message = {
        kind: 'message',
        role: 'user',
        messageId: uuidv4(),
        parts: [{ kind: 'text', text: initialMessage }],
      };

      await processMessageStream(a2aClient, message, agentId, this);
    } catch (error) {
      throw error;
    } finally {
      // Cleanup handled by SDK
    }
  }

  appendMessage(role, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    if (role === 'agent' && !text) {
      contentDiv.classList.add('streaming');
    }
    
    messageDiv.appendChild(contentDiv);
    this.messagesContainer.appendChild(messageDiv);
    this.scrollToBottom();
    
    return contentDiv;
  }

  scrollToBottom() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }
}

// Register the custom element
customElements.define('chat-interface', ChatInterface);
