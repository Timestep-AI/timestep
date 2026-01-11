#!/usr/bin/env node
/** Example of using streamEpisode with Express for real-time agent execution. */

import { streamEpisode, agentBuiltinEcho, DEFAULT_TOOLS } from '../timestep';
import type { Message } from '../timestep/core/types';

async function main() {
  // Define initial messages
  const messages: Message[] = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Calculate 5 + 3 using the calc tool.' }
  ];
  
  // Stream the episode
  for await (const event of streamEpisode(
    messages,
    agentBuiltinEcho, // Can also use streaming agents
    DEFAULT_TOOLS,
    ['calc'],
    { max_steps: 10, time_limit_s: 30 },
    { id: 'demo' },
    0,
  )) {
    const eventType = event.type;
    
    if (eventType === 'RunStarted') {
      console.log(`Run started: ${event.runId}`);
    } else if (eventType === 'StepStarted') {
      console.log(`Step ${event.stepName} started`);
    } else if (eventType === 'TextMessageContent') {
      // Stream content chunks in real-time
      process.stdout.write(event.delta as string);
    } else if (eventType === 'ToolCallChunk') {
      console.log(`\nTool call chunk:`, event.chunk);
    } else if (eventType === 'TextMessageEnd') {
      console.log(`\nMessage complete`);
    } else if (eventType === 'ToolCallStart') {
      console.log(`Tool call started: ${event.name}`);
    } else if (eventType === 'ToolCallResult') {
      console.log(`Tool call result:`, event.result);
    } else if (eventType === 'StepFinished') {
      console.log(`Step ${event.stepName} completed`);
    } else if (eventType === 'RunFinished') {
      const info = event.result.episodeInfo as any;
      console.log(`\nEpisode complete!`);
      console.log(`Steps: ${info.steps}, Tool calls: ${info.tool_calls}`);
      console.log(`Duration: ${info.duration_s}s`);
      if (info.total_tokens > 0) {
        console.log(`Tokens: ${info.total_tokens} (input: ${info.input_tokens}, output: ${info.output_tokens})`);
      }
    }
  }
}

// Example Express integration:
/*
import express from 'express';
import { streamEpisode, agentBuiltinEcho, DEFAULT_TOOLS } from 'timestep';

const app = express();
app.use(express.json());

app.post('/agent/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  
  for await (const event of streamEpisode(
    req.body.messages,
    agentBuiltinEcho,
    DEFAULT_TOOLS,
    req.body.tools_allowed,
    req.body.limits || {},
    req.body.task_meta || {},
    req.body.seed || 0,
  )) {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  }
  
  res.end();
});

app.listen(3000);
*/

main().catch(console.error);
