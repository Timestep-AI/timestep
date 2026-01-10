#!/usr/bin/env tsx
/** Example agent adapter using OpenAI library. */

import OpenAI from 'openai';
import { readFileSync } from 'fs';

// Agent adapter that reads from stdin and writes to stdout
async function main(): Promise<void> {
  /** Agent adapter main function. */
  // Read payload from stdin
  const input = readFileSync(0, 'utf-8');
  const payload = JSON.parse(input);
  const messages = payload.messages || [];
  const toolsSchema = payload.tools || [];
  
  // Initialize OpenAI client
  const client = new OpenAI();
  
  // Call OpenAI API
  const response = await client.chat.completions.create({
    model: 'gpt-4',
    messages: messages as any,
    tools: toolsSchema.length > 0 ? toolsSchema : undefined,
  });
  
  // Extract assistant message
  const message = response.choices[0].message;
  
  // Format as OpenAI-style assistant message
  const assistantMsg: any = {
    role: 'assistant',
    content: message.content || '',
  };
  
  // Add tool calls if present
  if (message.tool_calls) {
    assistantMsg.tool_calls = message.tool_calls.map((tc: any) => ({
      id: tc.id,
      type: 'function',
      function: {
        name: tc.function.name,
        arguments: tc.function.arguments,
      }
    }));
  }
  
  // Write to stdout
  console.log(JSON.stringify(assistantMsg));
}

main().catch(console.error);
