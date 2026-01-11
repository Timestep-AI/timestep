/** Agent harness interface and adapters. */

import { JSON, Message, StreamingAgentFn } from './types';

export function agentBuiltinEcho(messages: Message[], context: JSON): Message {
  /** Builtin agent harness that finishes immediately by echoing the last user message. */
  let lastUser = '';
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') {
      lastUser = String(messages[i].content || '');
      break;
    }
  }
  return { role: 'assistant', content: `Echo: ${lastUser}` };
}

export function _agentCmdFactory(agentCmd: string, timeoutS: number = 120): any {
  /**
   * Internal function for CLI use only - creates an agent harness that shells out to a command.
   * Not exported from the package.
   */
  const { spawn } = require('child_process');
  const { AgentFn } = require('./types');
  
  return async function _agent(messages: Message[], context: JSON): Promise<Message> {
    const payload = {
      messages,
      tools: context.tools_schema || [],
      task: context.task || {},
      seed: context.seed,
      limits: context.limits || {},
    };
    
    const t0 = Date.now() / 1000;
    const [cmd, ...args] = agentCmd.trim().split(/\s+/);
    
    return new Promise<Message>((resolve) => {
      const proc = spawn(cmd, args, {
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      
      let stdout = '';
      let stderr = '';
      
      proc.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });
      
      proc.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });
      
      const timeout = setTimeout(() => {
        proc.kill();
        resolve({ role: 'assistant', content: '', tool_calls: [], error: 'agent_timeout' });
      }, timeoutS * 1000);
      
      proc.on('close', () => {
        clearTimeout(timeout);
        
        if (!stdout.trim()) {
          resolve({ role: 'assistant', content: '', tool_calls: [], error: 'agent_empty_stdout', stderr });
          return;
        }
        
        try {
          const msg = JSON.parse(stdout.trim());
          msg._agent_latency_s = Math.round((Date.now() / 1000 - t0) * 1000) / 1000;
          if (stderr) msg._agent_stderr = stderr;
          resolve(msg);
        } catch {
          resolve({ role: 'assistant', content: stdout.trim() });
        }
      });
      
      proc.stdin.write(JSON.stringify(payload));
      proc.stdin.end();
    });
  };
}

export function createAgent(
  apiKey?: string,
  baseUrl?: string,
  model: string = 'gpt-4o-mini',
  temperature: number = 0.0,
  maxTokens?: number,
  timeoutS: number = 120,
): StreamingAgentFn {
  /**
   * Creates a streaming agent harness function that uses OpenAI-compatible streaming API.
   * 
   * This is the single agent function for the Timestep library. It always uses streaming.
   * Supports any OpenAI-compatible API (OpenAI, Anthropic via proxy, local models, etc.)
   * via the baseUrl parameter.
   * 
   * Yields chunks in Timestep format:
   * - {type: "content", delta: string} - content chunk
   * - {type: "tool_call", delta: {...}} - tool call chunk (partial)
   * - {type: "done"} - agent response complete
   * - {type: "error", error: string} - error occurred
   */
  return async function* _streamingAgent(messages: Message[], context: JSON): AsyncGenerator<JSON> {
    let client: any;
    try {
      // Dynamic import to avoid requiring openai at module load time
      const { OpenAI } = await import('openai');
      const clientOptions: any = {};
      if (apiKey || process.env.OPENAI_API_KEY) {
        clientOptions.apiKey = apiKey || process.env.OPENAI_API_KEY;
      }
      if (baseUrl || process.env.OPENAI_BASE_URL) {
        clientOptions.baseURL = baseUrl || process.env.OPENAI_BASE_URL;
      }
      client = new OpenAI(clientOptions);
    } catch (e) {
      yield { type: 'error', error: 'openai package required. Install with: npm install openai' };
      return;
    }

    // Get tools schema from context
    const tools = context.tools_schema || [];

    try {
      // Prepare request parameters
      const requestParams: any = {
        model,
        messages,
        temperature,
        stream: true,
      };
      if (maxTokens !== undefined) {
        requestParams.max_tokens = maxTokens;
      }
      if (tools.length > 0) {
        requestParams.tools = tools;
        requestParams.tool_choice = 'auto';
      }

      // Call OpenAI with streaming
      const stream = await client.chat.completions.create(requestParams);

      // Track accumulated message state
      let accumulatedContent = '';
      const accumulatedToolCalls: Record<string, any> = {};

      for await (const chunk of stream) {
        // Check for usage information (OpenAI provides this in the final chunk)
        if ((chunk as any).usage) {
          const usage = (chunk as any).usage;
          yield {
            type: 'usage',
            usage: {
              prompt_tokens: usage.prompt_tokens || 0,
              completion_tokens: usage.completion_tokens || 0,
              total_tokens: usage.total_tokens || 0,
            },
          };
        }

        if (!chunk.choices || chunk.choices.length === 0) {
          continue;
        }

        const choice = chunk.choices[0];
        const delta = choice.delta;

        // Content delta
        if (delta.content) {
          accumulatedContent += delta.content;
          yield { type: 'content', delta: delta.content };
        }

        // Tool call deltas
        if (delta.tool_calls) {
          for (const tcDelta of delta.tool_calls) {
            const tcId = tcDelta.id;
            if (!tcId) continue;

            if (!accumulatedToolCalls[tcId]) {
              accumulatedToolCalls[tcId] = {
                id: tcId,
                type: tcDelta.type || 'function',
                function: { name: '', arguments: '' },
              };
            }

            const tc = accumulatedToolCalls[tcId];

            // Function name delta
            if (tcDelta.function?.name) {
              tc.function.name = tcDelta.function.name;
            }

            // Function arguments delta
            if (tcDelta.function?.arguments) {
              tc.function.arguments += tcDelta.function.arguments;
              yield {
                type: 'tool_call',
                delta: {
                  id: tcId,
                  function: {
                    name: tc.function.name,
                    arguments: tcDelta.function.arguments,
                  },
                },
              };
            }
          }
        }
      }

      // Signal completion
      yield { type: 'done' };
    } catch (e: any) {
      yield { type: 'error', error: String(e) };
    }
  };
}
