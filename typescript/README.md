# @timestep-ai/timestep

Streaming agent implementation using OpenAI's streaming API with tool support.

## Installation

```bash
npm install @timestep-ai/timestep
# or
pnpm add @timestep-ai/timestep
# or
yarn add @timestep-ai/timestep
```

## Quick Start

### Basic Usage

```typescript
import { runAgent } from '@timestep-ai/timestep';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';

// Set your OpenAI API key
process.env.OPENAI_API_KEY = 'your-api-key-here';

// Basic usage
const messages: ChatCompletionMessageParam[] = [
  { role: 'system', content: 'You are a helpful AI assistant.' },
  { role: 'user', content: "What's 2+2?" },
];

const response = await runAgent(messages);
console.log(response);  // "2 + 2 = 4"
```

### Using Tools

Tools are Zod schemas that define the tool parameters:

```typescript
import { runAgent, GetWeatherParameters, WebSearchParameters } from '@timestep-ai/timestep';
import { z } from 'zod';

// Use built-in tools
const messages: ChatCompletionMessageParam[] = [
  {
    role: 'system',
    content: 'You are a helpful AI assistant that can answer questions about weather.',
  },
  { role: 'user', content: "What's the weather in Oakland?" },
];

const tools = [{ name: 'get_weather', parameters: GetWeatherParameters }];
const response = await runAgent(messages, tools);
console.log(response);

// Create a custom tool
const MyCustomToolParameters = z.object({
  arg1: z.string().describe('Description of arg1'),
  arg2: z.number().describe('Description of arg2'),
});

// Register your tool's execute function in tools.ts
const customTools = [{ name: 'my_custom_tool', parameters: MyCustomToolParameters }];
const customResponse = await runAgent(messages, customTools);
```

### Conversation Context

The `runAgent` function maintains conversation context by modifying the messages array:

```typescript
const messages: ChatCompletionMessageParam[] = [
  { role: 'system', content: 'You are a helpful AI assistant.' },
  { role: 'user', content: "What's 2+2?" },
];

// First message (runAgent appends assistant response to messages)
const response1 = await runAgent(messages);
console.log(response1);  // "2 + 2 = 4"

// Follow-up message (messages now includes the previous assistant response)
messages.push({ role: 'user', content: "What's three times that number?" });
const response2 = await runAgent(messages);
console.log(response2);  // "Three times 4 is 12"
```

### Streaming and Tool Approval

```typescript
import { runAgent, GetWeatherParameters } from '@timestep-ai/timestep';
import type { ChatCompletionMessageToolCall } from 'openai/resources/chat/completions';

const messages: ChatCompletionMessageParam[] = [
  { role: 'user', content: "What's the weather in San Francisco?" },
];

// Streaming callback
const onDelta = (delta: { content?: string; tool_calls?: any[] }) => {
  if (delta.content) {
    process.stdout.write(delta.content);
  }
};

// Tool approval callback
const onToolApprovalRequired = async (
  toolCall: ChatCompletionMessageToolCall
): Promise<boolean> => {
  console.log(`Tool call requested: ${toolCall.function.name}`);
  // Return true to approve, false to reject
  return true;
};

const response = await runAgent(
  messages,
  [{ name: 'get_weather', parameters: GetWeatherParameters }],
  'gpt-4.1',
  undefined,
  onDelta,
  onToolApprovalRequired
);
```

## API Reference

### `runAgent`

Run agent with streaming OpenAI API and tool support.

**Parameters:**
- `messages` (ChatCompletionMessageParam[], required): List of message objects with 'role' and 'content'
- `tools` (Array<{ name: string; parameters: z.ZodTypeAny }>, optional): Array of tool objects with schema
- `model` (string, optional): OpenAI model name (default: "gpt-4.1")
- `apiKey` (string, optional): OpenAI API key (defaults to OPENAI_API_KEY env var)
- `onDelta` ((delta: { content?: string; tool_calls?: any[] }) => void, optional): Callback for streaming deltas
- `onToolApprovalRequired` ((toolCall: ChatCompletionMessageToolCall) => Promise<boolean>, optional): Async callback for tool approval

**Returns:**
- `Promise<string>`: Final assistant response as string

**Note:** The function modifies the `messages` array in place, appending assistant responses and tool results.

### Tools

Tools are Zod schemas that define the tool parameters. Each tool must have:
- A Zod schema exported (e.g., `GetWeatherParameters`)
- An execute function registered in `timestep/tools.ts` via `callFunction`

```typescript
import { z } from 'zod';

const MyToolParameters = z.object({
  param1: z.string().describe('Parameter description'),
});

// Register execute function in tools.ts
async function myTool(args: { param1: string }): Promise<string> {
  return `Result: ${args.param1}`;
}
```

### Built-in Tools

- **GetWeatherParameters**: Simple weather tool (example implementation)
- **WebSearchParameters**: Web search using Firecrawl (requires `FIRECRAWL_API_KEY` environment variable)

## A2A Protocol Support

Timestep includes full support for the [Agent2Agent (A2A) Protocol](https://a2a-protocol.org/), enabling your agent to communicate with other A2A-compatible agents and systems.

### Running the A2A Server

**Command line (recommended):**

```bash
# From the typescript directory
pnpm a2a --host 0.0.0.0 --port 8080 --model gpt-4.1

# Or use environment variables
HOST=0.0.0.0 PORT=8080 MODEL=gpt-4.1 pnpm a2a

# Show help
pnpm a2a --help
```

**Programmatically:**

```typescript
import { runServer } from '@timestep-ai/timestep/a2a';

// Start the A2A server
runServer('0.0.0.0', 8080, undefined, 'gpt-4.1');
```

**What happens when you run it?**

1. Server starts on the specified host/port
2. Agent Card is available at `http://localhost:8080/.well-known/agent-card.json`
3. You can send tasks via HTTP POST to `/tasks`
4. Task updates stream via Server-Sent Events (SSE)
5. Other A2A agents can discover and interact with your agent

### A2A Features

- **Task-generating Agents**: All agent responses are encapsulated in `Task` objects, following the A2A "Task-generating Agents" philosophy
- **Human-in-the-loop**: Tool calls can require approval using the `input-required` task status
- **Streaming Updates**: Real-time task status updates via Server-Sent Events (SSE)
- **Agent Skills**: Tools are automatically exposed as `AgentSkill` objects in the Agent Card
- **Agent Card**: Discoverable agent metadata at `/.well-known/agent-card.json`

### Agent Card

Once the server is running, access the Agent Card at:
```
http://localhost:8080/.well-known/agent-card.json
```

The Agent Card includes:
- Agent name, description, and version
- Available skills (mapped from tools)
- Supported input/output modes
- Streaming capabilities
- Example queries

### Customizing the A2A Server

```typescript
import { createServer } from '@timestep-ai/timestep/a2a';
import { GetWeatherParameters, WebSearchParameters } from '@timestep-ai/timestep';

// Create server with custom tools
const app = createServer(
  '0.0.0.0',
  8080,
  [
    { name: 'get_weather', parameters: GetWeatherParameters },
    { name: 'web_search', parameters: WebSearchParameters },
  ],
  'gpt-4.1'
);

// Run with Express
app.listen(8080, () => {
  console.log('A2A server running on http://localhost:8080');
});
```

For more information about the A2A Protocol, visit [https://a2a-protocol.org/](https://a2a-protocol.org/).

## Testing

Timestep includes comprehensive test suites for both the core agent functionality and A2A server integration.

### Running Tests

```bash
# Run all TypeScript tests
make test-typescript

# Or run directly with vitest
cd typescript
pnpm test
```

### Test Coverage

The test suite includes:
- **Core Agent Tests** (`test_run_agent.ts`): Tests for the `runAgent` function with tools and streaming
- **A2A Server Tests** (`test_a2a_server.ts`): Tests for A2A server setup and agent card generation
- **A2A Agent Executor Tests** (`test_a2a_agent_executor.ts`): Tests for the A2A agent executor implementation
- **A2A Integration Tests** (`test_a2a_integration.ts`): End-to-end integration tests for A2A protocol

## Requirements

- Node.js >=20
- openai >=6.0.0
- zod >=3.0.0
- @mendable/firecrawl-js >=1.0.0 (for WebSearch tool)
- @a2a-js/sdk >=0.3.5 (for A2A Protocol support)
- express >=4.19.2 (for A2A server)

## License

MIT
