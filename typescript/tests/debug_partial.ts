import { runAgentTestPartial } from './test_run_agent';
import OpenAI from 'openai';

async function main() {
  const sessionId = await runAgentTestPartial(false, false, undefined, 0, 4);
  console.log('sessionId', sessionId);
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY required');
  const client = new OpenAI({ apiKey });
  const itemsResponse = await client.conversations.items.list(sessionId, { limit: 100, order: 'asc' });
  console.log(JSON.stringify(itemsResponse.data, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
