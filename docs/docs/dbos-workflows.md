# DBOS Workflows

Timestep integrates with [DBOS](https://www.dbos.dev/) to provide durable agent execution, automatic crash recovery, queuing with rate limiting, and scheduled agent runs.

## Overview

DBOS workflows make your agent executions **durable** - if your process crashes or restarts, workflows automatically resume from the last completed step. This is especially valuable for:

- Long-running agent conversations
- Production deployments where reliability is critical
- Managing LLM API rate limits
- Scheduled periodic agent tasks

## Prerequisites

DBOS uses the same database as Timestep's `RunStateStore`. If you're using PostgreSQL, set `PG_CONNECTION_URI`. Otherwise, DBOS will use SQLite by default.

## Quick Start

### Python

```python
from timestep import (
    run_agent_workflow,
    configure_dbos,
    ensure_dbos_launched
)
from agents import Agent, OpenAIConversationsSession

# Configure DBOS (uses PG_CONNECTION_URI if set)
configure_dbos()
ensure_dbos_launched()

agent = Agent(model="gpt-4.1")
session = OpenAIConversationsSession()

input_items = [
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}
]

# Run in a durable workflow
result = await run_agent_workflow(
    agent=agent,
    input_items=input_items,
    session=session,
    stream=False,
    workflow_id="my-workflow-1"  # Optional: idempotency key
)

print(result.output)
```

### TypeScript

```typescript
import {
  runAgentWorkflow,
  configureDBOS,
  ensureDBOSLaunched
} from '@timestep-ai/timestep';
import { Agent, OpenAIConversationsSession } from '@openai/agents';

// Configure DBOS (uses PG_CONNECTION_URI if set)
configureDBOS();
await ensureDBOSLaunched();

const agent = new Agent({ model: 'gpt-4.1' });
const session = new OpenAIConversationsSession();

const inputItems = [
  { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Hello!' }] }
];

// Run in a durable workflow
const result = await runAgentWorkflow(
  agent,
  inputItems,
  session,
  false,
  undefined,
  undefined,
  undefined,
  'my-workflow-1'  // Optional: workflow ID
);

console.log(result.output);
```

## Durable Execution

The `run_agent_workflow` function wraps your agent execution in a DBOS workflow. If the process crashes, the workflow automatically resumes from the last completed step.

### Features

- **Automatic state saving**: State is saved on interruptions automatically
- **Crash recovery**: Workflows resume after process restarts
- **Idempotency**: Use `workflow_id` to prevent duplicate executions
- **Timeouts**: Set `timeout_seconds` to cancel long-running workflows

### Example

```python
from timestep import run_agent_workflow

result = await run_agent_workflow(
    agent=agent,
    input_items=input_items,
    session=session,
    stream=False,
    workflow_id="unique-id",  # Idempotency key
    timeout_seconds=300  # 5 minute timeout
)
```

## Queued Execution

Use `queue_agent_workflow` to enqueue agent runs with rate limiting. This is perfect for managing LLM API rate limits.

### Features

- **Rate limiting**: Default queue limits to 50 requests per 60 seconds
- **Priority**: Set priority to control execution order
- **Deduplication**: Prevent duplicate runs with `deduplication_id`
- **Concurrency control**: Manage how many workflows run simultaneously

### Example

```python
from timestep import queue_agent_workflow

# Enqueue with priority and deduplication
handle = queue_agent_workflow(
    agent=agent,
    input_items=input_items,
    session=session,
    stream=False,
    priority=1,  # Lower number = higher priority
    deduplication_id="user-123-task-1"  # Prevents duplicates
)

# Get result when ready
result = await handle.get_result()
```

### Custom Queues

Create custom queues with different rate limits:

```python
from timestep import queue_agent_workflow

# Use a custom queue name (will be created automatically)
handle = queue_agent_workflow(
    agent=agent,
    input_items=input_items,
    session=session,
    queue_name="my-custom-queue"
)
```

## Scheduled Execution

Schedule agents to run periodically using cron syntax.

### Example

```python
from timestep import create_scheduled_agent_workflow

# Schedule agent to run every 6 hours
create_scheduled_agent_workflow(
    crontab="0 */6 * * *",  # Every 6 hours
    agent=agent,
    input_items=input_items,
    session=session,
    stream=False
)
```

### Crontab Syntax

- `*/5 * * * *` - Every 5 minutes
- `0 */6 * * *` - Every 6 hours
- `0 0 * * *` - Every day at midnight
- `0 0 * * 1` - Every Monday at midnight

**Note**: Keep your process running for scheduled workflows to execute. In production, use a process manager like systemd or a container orchestrator.

## State Persistence

DBOS workflows automatically integrate with Timestep's `RunStateStore`. When an agent run is interrupted (e.g., tool approval needed), the state is automatically saved and can be resumed later.

```python
# State is automatically saved on interruptions
result = await run_agent_workflow(agent, input_items, session)

# If interrupted, state is already saved
# You can load and resume later
if result.interruptions:
    state_store = RunStateStore(agent=agent, session_id=session_id)
    loaded_state = await state_store.load()
    # Approve interruptions and resume
    for interruption in loaded_state.get_interruptions():
        loaded_state.approve(interruption)
    result = await run_agent_workflow(agent, loaded_state, session)
```

## Configuration

### Database

DBOS uses the same database as Timestep:

- **PostgreSQL**: Set `PG_CONNECTION_URI=postgresql://user:pass@host/db`
- **SQLite**: Default (no configuration needed)

### Application Name

Customize the DBOS application name:

```python
from timestep import configure_dbos

configure_dbos(name="my-timestep-app")
```

## Best Practices

1. **Use workflow IDs for idempotency**: Prevent duplicate executions with unique workflow IDs
2. **Set timeouts**: Use `timeout_seconds` to prevent runaway workflows
3. **Use queues for rate limiting**: Respect LLM API rate limits with queued execution
4. **Monitor workflows**: Use DBOS admin server (runs on port 3001 by default) to monitor workflow status
5. **Handle interruptions**: Workflows automatically save state, but you still need to handle approvals

## Troubleshooting

### Workflows not resuming

- Ensure DBOS is configured and launched before using workflows
- Check that the database is accessible
- Verify workflow IDs are unique

### Rate limiting issues

- Adjust queue rate limits if needed
- Use priority to control execution order
- Consider using multiple queues for different use cases

### Scheduled workflows not running

- Ensure the process is kept running
- Check crontab syntax
- Verify DBOS is launched

## API Reference

### Python

- `run_agent_workflow()` - Run agent in durable workflow
- `queue_agent_workflow()` - Enqueue agent run with rate limiting
- `create_scheduled_agent_workflow()` - Schedule periodic agent runs
- `configure_dbos()` - Configure DBOS
- `ensure_dbos_launched()` - Ensure DBOS is launched

### TypeScript

- `runAgentWorkflow()` - Run agent in durable workflow
- `queueAgentWorkflow()` - Enqueue agent run with rate limiting
- `createScheduledAgentWorkflow()` - Schedule periodic agent runs
- `configureDBOS()` - Configure DBOS
- `ensureDBOSLaunched()` - Ensure DBOS is launched

See the [API reference](../api-reference/) for detailed function signatures.

