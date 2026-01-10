# @timestep-ai/timestep (TypeScript)

TypeScript bindings for the Timestep AI Agents SDK. See the root `README.md` for the full story; this file highlights TypeScript-specific setup.

## Install

```bash
npm install @timestep-ai/timestep
# or
pnpm add @timestep-ai/timestep
# or
yarn add @timestep-ai/timestep
```

## Prerequisites

- **Node.js 20+**
- **OpenAI API key** (optional, for agents that use OpenAI or LLM-as-judge graders)

## Quick Start: Core Agent-Environment Loop

```typescript
import { runEpisode, agentBuiltinEcho, DEFAULT_TOOLS } from '@timestep-ai/timestep';
import type { Message } from '@timestep-ai/timestep';

// Define initial messages
const messages: Message[] = [
  { role: 'system', content: 'You are helpful.' },
  { role: 'user', content: 'Calculate 5 + 3 using the calc tool.' }
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

console.log(`Steps: ${info.steps}, Tool calls: ${info.tool_calls}`);
console.log(`Final message: ${transcript[transcript.length - 1]?.content}`);
```

## Quick Start: Evaluation Harness

```typescript
import { runSuite, report, agentBuiltinEcho, DEFAULT_TOOLS } from '@timestep-ai/timestep';
import { FinalContains, ForbiddenTools, LLMJudge } from '@timestep-ai/timestep';
import { writeFileSync } from 'fs';

// Create tasks.jsonl
const tasks = [
  {
    id: 'hello_01',
    messages: [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Say hello to Mike.' }
    ],
    expected: { final_contains: 'Mike' },
    limits: { max_steps: 5 }
  }
];

writeFileSync('tasks.jsonl', tasks.map(t => JSON.stringify(t)).join('\n') + '\n');

// Run eval suite
await runSuite(
  'tasks.jsonl',
  'runs/demo',
  agentBuiltinEcho,
  DEFAULT_TOOLS,
  [
    new FinalContains(),  // Code-based grader
    new ForbiddenTools(),  // Tool usage checker
    // new LLMJudge(undefined, 'gpt-4o-mini', 0.0, false),  // LLM-as-judge
  ],
  3,
  0,
  120
);

// Generate report
report('runs/demo');
```

## CLI Usage

```bash
# Build first
cd typescript
pnpm build

# Run eval suite
node dist/eval/cli.js run --tasks tasks.jsonl --outdir runs/demo --agent builtin:echo

# Generate report
node dist/eval/cli.js report --outdir runs/demo
```

## Creating Your Own Agent Harness

```typescript
import type { AgentFn, Message, JSON } from '@timestep-ai/timestep';

const myAgent: AgentFn = async (messages: Message[], context: JSON): Promise<Message> => {
  /**
   * Agent harness function.
   * 
   * Args:
   *   messages: Full conversation history (transcript)
   *   context: Context with tools_schema, task, seed, limits
   * 
   * Returns:
   *   Assistant message (may include tool_calls and usage info)
   */
  // Your agent logic here
  // Use OpenAI library or other model provider
  return {
    role: 'assistant',
    content: 'Response here',
    tool_calls: [...],  // Optional
    usage: {  // Optional: token usage info
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15
    }
  };
};
```

See `examples/agent_adapter_example.ts` for a complete OpenAI integration example.

## Project Structure

```
typescript/timestep/
├── core/              # Core agent-environment loop
│   ├── agent.ts       # Agent harness interface
│   ├── episode.ts     # Episode runner
│   ├── tools.ts        # Tool execution
│   └── types.ts        # Core types
├── eval/               # Evaluation harness
│   ├── suite.ts        # Suite runner
│   ├── graders.ts      # All graders (code-based, LLM-as-judge, outcome)
│   └── cli.ts          # CLI interface
└── utils/              # Utilities (JSONL, hashing, etc.)
```

## Testing

```bash
cd typescript
pnpm test
```

## Documentation

- Full docs: https://timestep-ai.github.io/timestep/
- Root README: `../README.md`
