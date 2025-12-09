# Timestep AI Agents SDK

Durable OpenAI Agents with one API across Python and TypeScript. Pause and resume runs (even across languages), keep state in one place, and route models with simple prefixes.

## What Timestep gives you
- **Durable runs**: Save and resume `RunState` without changing your agent code.
- **DBOS workflows**: Run agents in durable workflows that automatically recover from crashes, with queuing and scheduling support.
- **Cross-language parity**: Same API surface in Python and TypeScript; state stays compatible across languages.
- **Single storage story**: Use `PG_CONNECTION_URI` for PostgreSQL.
- **Model routing**: Prefix models (`ollama/gpt-oss:20b-cloud`) and let `MultiModelProvider` pick the backend.
- **Minimal concepts**: `run_agent` / `runAgent`, `RunStateStore`, `run_agent_workflow` / `runAgentWorkflow`.
- **Organized architecture**: Clean separation of concerns with `core/`, `config/`, `stores/`, `tools/`, `model_providers/`, and `models/` modules.

## Prerequisites
- `OPENAI_API_KEY`
- **PostgreSQL**: Set `PG_CONNECTION_URI=postgresql://user:pass@host/db`

## Quick start

### Python (async)

```python
from timestep import run_agent, RunStateStore
from agents import (
    Agent,
    OpenAIConversationsSession,
    ModelSettings,
    function_tool,
    input_guardrail,
    output_guardrail,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
)

# Define a tool with approval requirement
async def needs_approval_for_weather(ctx, args, call_id):
    """Require approval for sensitive cities."""
    return args.get("city", "").lower() in ["berkeley", "san francisco"]

@function_tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"The weather in {city} is sunny and 72°F"

get_weather.needs_approval = needs_approval_for_weather

# Define guardrails
@input_guardrail(run_in_parallel=True)
async def content_filter(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """Block inappropriate content."""
    input_text = input if isinstance(input, str) else str(input)
    blocked = any(word in input_text.lower() for word in ["spam", "scam"])
    return GuardrailFunctionOutput(
        output_info={"blocked": blocked},
        tripwire_triggered=blocked,
    )

@output_guardrail
async def output_safety(
    ctx: RunContextWrapper,
    agent: Agent,
    output: str
) -> GuardrailFunctionOutput:
    """Ensure safe output."""
    unsafe = "password" in output.lower() or "api key" in output.lower()
    return GuardrailFunctionOutput(
        output_info={"unsafe": unsafe},
        tripwire_triggered=unsafe,
    )

# Create specialized weather agent
weather_agent = Agent(
    instructions="You are a weather assistant. Use get_weather for all weather queries.",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0),
    name="Weather Assistant",
    tools=[get_weather],
)

# Create main assistant with handoffs and guardrails
assistant = Agent(
    handoffs=[weather_agent],
    instructions="You are a helpful personal assistant.",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0),
    name="Personal Assistant",
    input_guardrails=[content_filter],
    output_guardrails=[output_safety],
)

# Initialize session and state store
session = OpenAIConversationsSession()
session_id = await session._get_session_id()
state_store = RunStateStore(agent=assistant, session_id=session_id)

# Run multiple conversation turns
conversations = [
    [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's 2+2?"}]}],
    [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's the weather in Oakland?"}]}],
    [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's the weather in Berkeley?"}]}],
]

for input_items in conversations:
    result = await run_agent(assistant, input_items, session, stream=False)
    
    # Handle interruptions (e.g., tool approval needed)
    if result.interruptions:
        state = result.to_state()
        await state_store.save(state)
        
        # Load, approve, and resume
        loaded_state = await state_store.load()
        for interruption in loaded_state.get_interruptions():
            loaded_state.approve(interruption)
        
        result = await run_agent(assistant, loaded_state, session, stream=False)
```

### TypeScript

```typescript
import { runAgent, RunStateStore } from '@timestep-ai/timestep';
import {
  Agent,
  OpenAIConversationsSession,
  tool,
  InputGuardrail,
  OutputGuardrail,
} from '@openai/agents';
import type { AgentInputItem } from '@openai/agents-core';

// Define a tool with approval requirement
const getWeather = tool({
  name: 'get_weather',
  description: 'Get weather information for a city.',
  parameters: {
    type: 'object',
    properties: {
      city: { type: 'string' }
    },
    required: ['city'],
    additionalProperties: false
  } as any,
  needsApproval: async (_ctx: any, args: any) => {
    // Require approval for sensitive cities
    return ['berkeley', 'san francisco'].includes(args.city?.toLowerCase() || '');
  },
  execute: async (args: any): Promise<string> => {
    return `The weather in ${args.city} is sunny and 72°F`;
  }
});

// Define guardrails
const contentFilter: InputGuardrail = {
  name: 'Content Filter',
  runInParallel: true,
  execute: async ({ input }) => {
    const inputText = typeof input === 'string' ? input : JSON.stringify(input);
    const blocked = ['spam', 'scam'].some(word => 
      inputText.toLowerCase().includes(word)
    );
    return {
      outputInfo: { blocked },
      tripwireTriggered: blocked,
    };
  },
};

const outputSafety: OutputGuardrail<any> = {
  name: 'Output Safety',
  execute: async ({ agentOutput }) => {
    const outputText = typeof agentOutput === 'string' 
      ? agentOutput 
      : JSON.stringify(agentOutput);
    const unsafe = outputText.toLowerCase().includes('password') || 
                   outputText.toLowerCase().includes('api key');
    return {
      outputInfo: { unsafe },
      tripwireTriggered: unsafe,
    };
  },
};

// Create specialized weather agent
const weatherAgent = new Agent({
  instructions: 'You are a weather assistant. Use get_weather for all weather queries.',
  model: 'gpt-4.1',
  modelSettings: { temperature: 0 },
  name: 'Weather Assistant',
  tools: [getWeather],
});

// Create main assistant with handoffs and guardrails
const assistant = new Agent({
  handoffs: [weatherAgent],
  instructions: 'You are a helpful personal assistant.',
  model: 'gpt-4.1',
  modelSettings: { temperature: 0 },
  name: 'Personal Assistant',
  inputGuardrails: [contentFilter],
  outputGuardrails: [outputSafety],
});

// Initialize session and state store
const session = new OpenAIConversationsSession();
const sessionId = await session.getSessionId();
const stateStore = new RunStateStore({ agent: assistant, sessionId });

// Run multiple conversation turns
const conversations: AgentInputItem[][] = [
  [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: "What's 2+2?" }] }],
  [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: "What's the weather in Oakland?" }] }],
  [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: "What's the weather in Berkeley?" }] }],
];

for (const inputItems of conversations) {
  let result = await runAgent(assistant, inputItems, session, false);
  
  // Handle interruptions (e.g., tool approval needed)
  if (result.interruptions?.length) {
    await stateStore.save(result.state);
    
    // Load, approve, and resume
    const loadedState = await stateStore.load();
    for (const interruption of loadedState.getInterruptions()) {
      loadedState.approve(interruption);
    }
    
    result = await runAgent(assistant, loadedState, session, false);
  }
}
```

## Cross-language resume

1) Start in Python, save state on interruption:
```python
state = result.to_state()
await state_store.save(state)
```
2) Load and continue in TypeScript:
```typescript
const saved = await stateStore.load();
for (const interruption of saved.getInterruptions()) saved.approve(interruption);
await runAgent(agent, saved, session, false);
```

## DBOS Workflows (New!)

Timestep now supports durable agent execution via DBOS workflows. Run agents in workflows that automatically recover from crashes, with built-in queuing and scheduling.

### Durable Execution

```python
from timestep import run_agent_workflow, configure_dbos, ensure_dbos_launched
from agents import Agent, OpenAIConversationsSession

configure_dbos()
ensure_dbos_launched()

agent = Agent(model="gpt-4.1")
session = OpenAIConversationsSession()

# Run in a durable workflow - automatically saves state and recovers from crashes
result = await run_agent_workflow(
    agent=agent,
    input_items=input_items,
    session=session,
    stream=False,
    workflow_id="unique-id"  # Idempotency key
)
```

### Queued Execution with Rate Limiting

```python
from timestep import queue_agent_workflow

# Enqueue agent runs with automatic rate limiting (50 requests per 60 seconds)
handle = queue_agent_workflow(
    agent=agent,
    input_items=input_items,
    session=session,
    priority=1,  # Higher priority
    deduplication_id="unique-queue-id"
)

result = await handle.get_result()
```

### Scheduled Execution

```python
from timestep import create_scheduled_agent_workflow

# Schedule agent to run every 6 hours
create_scheduled_agent_workflow(
    crontab="0 */6 * * *",  # Every 6 hours
    agent=agent,
    input_items=input_items,
    session=session
)
```

See the [DBOS Workflows documentation](docs/docs/dbos-workflows.md) for more details.

## Routing models
- `gpt-4.1` or `openai/gpt-4.1` → OpenAI
- `ollama/gpt-oss:20b` → Ollama (local)
- `ollama/gpt-oss:20b-cloud` → [Ollama Cloud](https://ollama.com/cloud) (note: `-cloud` suffix determines cloud usage)
- Add your own prefixes via `MultiModelProviderMap`.

## Docs
- Full docs: https://timestep-ai.github.io/timestep/
- Python notes: python/README.md
- TypeScript notes: typescript/README.md

## License
MIT License - see `LICENSE`.
