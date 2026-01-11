#!/usr/bin/env node
/** Example of creating a streaming agent harness using OpenAI's streaming API. */

import { streamEpisode, createAgent, DEFAULT_TOOLS } from '../timestep';
import type { Message } from '../timestep/core/types';

async function main() {
  // Create streaming agent (requires OpenAI API key)
  let agent;
  try {
    agent = createAgent(process.env.OPENAI_API_KEY);
  } catch (e: any) {
    console.error('OpenAI not available:', e.message);
    console.error('Install with: npm install openai');
    console.error('Setting OPENAI_API_KEY environment variable is required.');
    return;
  }
  
  // Define initial messages
  const messages: Message[] = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Count from 1 to 5, saying each number on a new line.' }
  ];
  
  console.log('Streaming agent response (chunks arrive in real-time):\n');
  
  // Stream the episode - chunks will arrive as the agent generates them
  for await (const event of streamEpisode(
    messages,
    agent, // Streaming agent
    DEFAULT_TOOLS,
    undefined,
    { max_steps: 5, time_limit_s: 30 },
    { id: 'streaming_demo' },
    0,
  )) {
    const eventType = event.type;
    
    if (eventType === 'TextMessageContent') {
      // Content chunks arrive in real-time (like OpenAI streaming)
      process.stdout.write(event.delta as string);
    } else if (eventType === 'ToolCallChunk') {
      console.log(`\n[Tool call chunk:`, event.chunk, ']');
    } else if (eventType === 'TextMessageEnd') {
      console.log('\n\n[Message complete]');
    } else if (eventType === 'RunFinished') {
      const info = event.result.episodeInfo as any;
      console.log('\n\nEpisode complete!');
      console.log(`Steps: ${info.steps}, Duration: ${info.duration_s}s`);
    }
  }
}

main().catch(console.error);
