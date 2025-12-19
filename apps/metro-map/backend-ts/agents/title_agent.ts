/**
 * Title generation agent for thread titles.
 */

import { Agent } from "@openai/agents";

export const titleAgent = new Agent({
  model: "gpt-5-nano",
  name: "Title generator",
  instructions: `
    Generate a short conversation title for a metro planning assistant chatting with a user.
    The first user message in the thread is included below to provide context. Use your own
    words, respond with 2-5 words, and avoid punctuation.
  `.trim(),
});

