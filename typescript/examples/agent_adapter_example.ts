/** Example of creating a custom agent harness using OpenAI. */

import { runEpisode, DEFAULT_TOOLS } from '@timestep-ai/timestep';
import type { AgentFn, Message, JSON } from '@timestep-ai/timestep';
import { agentBuiltinEcho } from '@timestep-ai/timestep';

function createOpenAIAgent(apiKey?: string): AgentFn {
  /**
   * Creates an agent harness function that uses OpenAI.
   * 
   * This is a simple example - in production you'd want better error handling,
   * retry logic, streaming support, etc.
   */
  return async function agent(messages: Message[], context: JSON): Promise<Message> {
    /**
     * Agent harness that calls OpenAI API.
     * 
     * Args:
     *   messages: Full conversation history (transcript)
     *   context: Context with tools_schema, task, seed, limits
     * 
     * Returns:
     *   Assistant message (may include tool_calls and usage info)
     */
    try {
      const { OpenAI } = await import('openai');
      const client = new OpenAI({ apiKey });
      
      // Get tools schema from context
      const tools = context.tools_schema || [];
      
      // Call OpenAI
      const response = await client.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: messages as any,
        tools: tools.length > 0 ? (tools as any) : undefined,
        tool_choice: tools.length > 0 ? 'auto' : undefined,
      });
      
      // Extract assistant message
      const choice = response.choices[0];
      const message = choice.message;
      
      // Build assistant message in Timestep format
      const assistantMsg: Message = {
        role: 'assistant',
        content: message.content || '',
      };
      
      // Add tool calls if present
      if (message.tool_calls) {
        assistantMsg.tool_calls = message.tool_calls.map(tc => ({
          id: tc.id,
          type: tc.type,
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments,
          }
        }));
      }
      
      // Add usage info for token tracking
      if (response.usage) {
        assistantMsg.usage = {
          prompt_tokens: response.usage.prompt_tokens,
          completion_tokens: response.usage.completion_tokens,
          total_tokens: response.usage.total_tokens,
        };
      }
      
      return assistantMsg;
    } catch (error) {
      // Fallback to echo agent if OpenAI fails
      console.warn('OpenAI not available, using echo agent:', error);
      return agentBuiltinEcho(messages, context);
    }
  };
}

async function main() {
  // Create agent harness
  // In production, get API key from environment: process.env.OPENAI_API_KEY
  let agent: AgentFn;
  try {
    agent = createOpenAIAgent();
  } catch {
    console.log('OpenAI not available, using builtin echo agent instead');
    agent = agentBuiltinEcho;
  }
  
  // Run a single episode
  const messages: Message[] = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Calculate 5 + 3 using the calc tool.' }
  ];
  
  const [transcript, info] = await runEpisode(
    messages,
    agent,
    DEFAULT_TOOLS,
    ['calc'],
    { max_steps: 10, time_limit_s: 30 },
    { id: 'demo' },
    0
  );
  
  console.log(`Episode completed in ${info.steps} steps`);
  console.log(`Tool calls: ${info.tool_calls}`);
  if (info.total_tokens > 0) {
    console.log(`Tokens used: ${info.total_tokens} (input: ${info.input_tokens}, output: ${info.output_tokens})`);
  }
  console.log(`Final message: ${transcript[transcript.length - 1]?.content || ''}`);
}

main().catch(console.error);
