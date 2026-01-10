/** Agent harness interface and adapters. */

import { spawn } from 'child_process';
import { AgentFn, JSON, Message, StreamingAgentFn } from './types';

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

export function agentCmdFactory(agentCmd: string, timeoutS: number = 120): AgentFn {
  /**
   * Creates an agent harness function that shells out to `agentCmd`.
   * 
   * Protocol:
   *   - send JSON to stdin: {"messages":[...], "tools":[...], "task":{...}, "seed":..., "limits":...}
   *   - expect stdout JSON: assistant message dict
   * 
   * Note: This returns a Promise-based agent for async command execution.
   */
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
          // If agent prints plain text, treat as assistant content
          resolve({ role: 'assistant', content: stdout.trim() });
        }
      });
      
      proc.stdin.write(JSON.stringify(payload));
      proc.stdin.end();
    });
  };
}

export function createOpenAIStreamingAgent(apiKey?: string): StreamingAgentFn {
  /**
   * Creates a streaming agent harness function that uses OpenAI's streaming API.
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
      client = new OpenAI({ apiKey });
    } catch (e) {
      yield { type: 'error', error: 'openai package required. Install with: npm install openai' };
      return;
    }

    // Get tools schema from context
    const tools = context.tools_schema || [];

    try {
      // Call OpenAI with streaming
      const stream = await client.chat.completions.create({
        model: 'gpt-4o-mini',
        messages,
        tools: tools.length > 0 ? tools : undefined,
        tool_choice: tools.length > 0 ? 'auto' : undefined,
        stream: true,
      });

      // Track accumulated message state
      let accumulatedContent = '';
      const accumulatedToolCalls: Record<string, any> = {};

      for await (const chunk of stream) {
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
