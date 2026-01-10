/** Example usage of Timestep core agent-environment loop (without evaluation). */

import { runEpisode, agentBuiltinEcho, DEFAULT_TOOLS } from '@timestep-ai/timestep';
import type { Message } from '@timestep-ai/timestep';

async function main() {
  // Define initial messages
  const messages: Message[] = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Calculate 5 + 3 using the calc tool, then tell me the result.' }
  ];
  
  // Run a single episode
  const [transcript, info] = await runEpisode(
    messages,
    agentBuiltinEcho,
    DEFAULT_TOOLS,
    ['calc'],
    { max_steps: 10, time_limit_s: 30 },
    { id: 'demo' },
    0
  );
  
  // Print results
  console.log('Episode completed!');
  console.log(`Steps: ${info.steps}`);
  console.log(`Tool calls: ${info.tool_calls}`);
  console.log(`Duration: ${info.duration_s.toFixed(2)}s`);
  console.log(`Terminated reason: ${info.terminated_reason}`);
  if (info.input_tokens > 0) {
    console.log(`Tokens: ${info.total_tokens} (input: ${info.input_tokens}, output: ${info.output_tokens})`);
  }
  
  console.log('\nTranscript:');
  transcript.forEach((msg, i) => {
    const role = msg.role || 'unknown';
    const content = String(msg.content || '').substring(0, 100); // Truncate long content
    console.log(`${i + 1}. ${role}: ${content}`);
  });
}

main().catch(console.error);
