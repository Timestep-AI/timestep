# Architecture

Timestep provides a custom agent execution loop with human-in-the-loop, guardrails, handoffs, and sessions.

## System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Agent Execution Loop                     │
│                  (Custom Implementation)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              OpenAI API Integration                         │
│  - Direct API calls (no SDK wrapper)                      │
│  - Tool call handling                                      │
│  - Streaming support                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Guardrail System                               │
│  - Pre-execution validation                                │
│  - Post-execution validation                               │
│  - Human approval requests                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Tool Execution                                 │
│  - Regular tools                                            │
│  - Handoff tools (agent delegation)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Session Management                            │
│  - FileSession (JSONL storage)                             │
│  - Conversation persistence                                │
└─────────────────────────────────────────────────────────────┘
```

## Execution Flow

1. **Agent Initialization**: Agent is configured with tools, handoffs, and guardrails
2. **Handoff Tool Generation**: Agent handoffs are converted to tools (e.g., `transfer_to_weather_agent`)
3. **Message Processing**: Messages are sent to OpenAI API with tools
4. **Tool Execution**: Tools are executed with guardrails applied
5. **Human-in-the-Loop**: Guardrails can request approval via `GuardrailInterrupt`
6. **Session Persistence**: All messages are saved to FileSession

## Guardrails

Guardrails provide validation and modification of tool execution:

### Input Guardrails

Run before tool execution:
- Can **block** execution
- Can **modify** input arguments
- Can **request approval** via `GuardrailInterrupt`

### Output Guardrails

Run after tool execution:
- Can **modify** output results
- Can **block** results

### Human-in-the-Loop

When a guardrail raises `GuardrailInterrupt`, execution pauses and `request_approval()` is called. The user can approve or deny the operation.

## Handoffs

Handoffs enable agent-to-agent delegation:

1. Agent A has `handoffs=[Agent B]`
2. Tool `transfer_to_agent_b` is automatically created
3. When called, Agent B receives the message in its own session
4. Agent B processes and returns result to Agent A

Each agent manages its own conversation session independently.

## Sessions

`FileSession` provides simple file-based storage:
- One JSONL file per agent+conversation
- Messages stored as JSON lines
- No database required
- Persists across runs

## Design Principles

- **No SDK Dependencies**: Direct OpenAI API integration
- **Simple Storage**: File-based sessions (no database required)
- **Human-in-the-Loop**: Guardrails can pause for approval
- **Agent Delegation**: Handoffs enable multi-agent workflows
- **Minimal Surface Area**: Core concepts only
