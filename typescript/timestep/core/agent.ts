/** Agent harness interface and adapters. */

import { spawn } from 'child_process';
import { AgentFn, JSON, Message } from './types';

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
