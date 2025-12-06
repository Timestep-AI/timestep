# Utilities

Timestep provides several utility functions and classes to help with agent execution, state management, and result consumption.

## run_agent / runAgent

A convenience function for running agents with session management and error handling.

### Function Signature

=== "Python"

    ```python
    async def run_agent(
        agent: Agent,
        run_input: list[TResponseInputItem] | RunState,
        session: SessionABC,
        stream: bool
    ) -> Any
    ```

=== "TypeScript"

    ```typescript
    export async function runAgent(
      agent: Agent,
      runInput: AgentInputItem[] | RunState<any, any>,
      session: Session,
      stream: boolean
    ): Promise<any>
    ```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Agent` | The agent to run. |
| `run_input` / `runInput` | `list[TResponseInputItem] \| RunState` / `AgentInputItem[] \| RunState` | The input for the agent run. Can be a list of input items or a RunState. |
| `session` | `SessionABC` / `Session` | The session to use for maintaining conversation context. |
| `stream` | `bool` / `boolean` | Whether to use streaming mode. |

### Returns

| Type | Description |
|------|-------------|
| `RunResult` / `RunResultStreaming` | The result of the agent run. |

### Features

- **Session Management**: Automatically handles session input callbacks
- **Error Handling**: Catches and logs common agent errors (MaxTurnsExceeded, ModelBehaviorError, UserError, AgentsException)
- **Streaming Support**: Supports both streaming and non-streaming modes
- **Configuration**: Uses default RunConfig settings (nest_handoff_history=False)

### Example

=== "Python"

    ```python
    from timestep import run_agent
    from agents import Agent, Session

    agent = Agent(model="gpt-4")
    session = Session()

    # Non-streaming
    result = await run_agent(
        agent,
        [{"role": "user", "content": "Hello"}],
        session,
        stream=False
    )

    # Streaming
    result = await run_agent(
        agent,
        [{"role": "user", "content": "Hello"}],
        session,
        stream=True
    )
    ```

=== "TypeScript"

    ```typescript
    import { runAgent } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4' });
    const session = new Session();

    // Non-streaming
    const result = await runAgent(
      agent,
      [{ role: 'user', content: 'Hello' }],
      session,
      false
    );

    // Streaming
    const result = await runAgent(
      agent,
      [{ role: 'user', content: 'Hello' }],
      session,
      true
    );
    ```

## consume_result / consumeResult

Consumes all events from a result (streaming or non-streaming), ensuring all background operations complete.

### Function Signature

=== "Python"

    ```python
    async def consume_result(result: Any) -> Any
    ```

=== "TypeScript"

    ```typescript
    export async function consumeResult(result: any): Promise<any>
    ```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `result` | `RunResult \| RunResultStreaming` | The result from `run_agent` or `Runner.run()`. |

### Returns

| Type | Description |
|------|-------------|
| Same as input | The same result object after consuming all stream events. |

### Purpose

This function ensures that:

- All streaming events are consumed
- All background operations (like session updates) complete
- The result is ready for final access

This is particularly important for streaming results, where background operations may continue after the stream completes.

### Example

=== "Python"

    ```python
    from timestep import run_agent, consume_result
    from agents import Agent, Session

    agent = Agent(model="gpt-4")
    session = Session()

    result = await run_agent(
        agent,
        [{"role": "user", "content": "Hello"}],
        session,
        stream=True
    )

    # Consume all events and wait for completion
    result = await consume_result(result)

    # Now safe to access final result
    print(result.output)
    ```

=== "TypeScript"

    ```typescript
    import { runAgent, consumeResult } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4' });
    const session = new Session();

    const result = await runAgent(
      agent,
      [{ role: 'user', content: 'Hello' }],
      session,
      true
    );

    // Consume all events and wait for completion
    const finalResult = await consumeResult(result);

    // Now safe to access final result
    console.log(finalResult.output);
    ```

## RunStateStore

A utility class for persisting and loading agent run state to/from files. Useful for saving and resuming agent conversations.

### Class Definition

=== "Python"

    ```python
    class RunStateStore:
        def __init__(self, file_path: str, agent: Agent)
        async def save(self, state: Any) -> None
        async def load(self) -> Any
        async def clear(self) -> None
    ```

=== "TypeScript"

    ```typescript
    export class RunStateStore {
      constructor(filePath: string, agent: Agent)
      async save(state: any): Promise<void>
      async load(): Promise<any>
      async clear(): Promise<void>
    }
    ```

### Constructor

=== "Python"

    ```python
    def __init__(self, file_path: str, agent: Agent)
    ```

=== "TypeScript"

    ```typescript
    constructor(filePath: string, agent: Agent)
    ```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` / `filePath` | `string` | The path to the file where state will be stored. |
| `agent` | `Agent` | The agent instance. Required for loading state. |

### Methods

#### `save()`

Saves the run state to a file.

=== "Python"

    ```python
    async def save(self, state: Any) -> None
    ```

=== "TypeScript"

    ```typescript
    async save(state: any): Promise<void>
    ```

##### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `RunState` | The run state to save. |

##### Example

=== "Python"

    ```python
    from timestep import RunStateStore
    from agents import Agent, RunState

    agent = Agent(model="gpt-4")
    store = RunStateStore("state.json", agent)

    # Save state
    await store.save(run_state)
    ```

=== "TypeScript"

    ```typescript
    import { RunStateStore } from '@timestep-ai/timestep';
    import { Agent, RunState } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4' });
    const store = new RunStateStore('state.json', agent);

    // Save state
    await store.save(runState);
    ```

#### `load()`

Loads run state from a file.

=== "Python"

    ```python
    async def load(self) -> Any
    ```

=== "TypeScript"

    ```typescript
    async load(): Promise<any>
    ```

##### Returns

| Type | Description |
|------|-------------|
| `RunState` | The loaded run state. |

##### Example

=== "Python"

    ```python
    from timestep import RunStateStore
    from agents import Agent

    agent = Agent(model="gpt-4")
    store = RunStateStore("state.json", agent)

    # Load state
    run_state = await store.load()
    ```

=== "TypeScript"

    ```typescript
    import { RunStateStore } from '@timestep-ai/timestep';
    import { Agent } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4' });
    const store = new RunStateStore('state.json', agent);

    // Load state
    const runState = await store.load();
    ```

#### `clear()`

Deletes the state file.

=== "Python"

    ```python
    async def clear(self) -> None
    ```

=== "TypeScript"

    ```typescript
    async clear(): Promise<void>
    ```

##### Example

=== "Python"

    ```python
    from timestep import RunStateStore
    from agents import Agent

    agent = Agent(model="gpt-4")
    store = RunStateStore("state.json", agent)

    # Clear saved state
    await store.clear()
    ```

=== "TypeScript"

    ```typescript
    import { RunStateStore } from '@timestep-ai/timestep';
    import { Agent } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4' });
    const store = new RunStateStore('state.json', agent);

    // Clear saved state
    await store.clear();
    ```

### Complete Example

=== "Python"

    ```python
    from timestep import run_agent, consume_result, RunStateStore
    from agents import Agent, Session

    agent = Agent(model="gpt-4")
    session = Session()
    store = RunStateStore("conversation.json", agent)

    # Try to load existing state
    try:
        run_state = await store.load()
        print("Resuming conversation")
    except FileNotFoundError:
        run_state = [{"role": "user", "content": "Hello"}]
        print("Starting new conversation")

    # Run agent
    result = await run_agent(agent, run_state, session, stream=False)
    result = await consume_result(result)

    # Save state for next time
    await store.save(result.state)
    ```

=== "TypeScript"

    ```typescript
    import { runAgent, consumeResult, RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4' });
    const session = new Session();
    const store = new RunStateStore('conversation.json', agent);

    // Try to load existing state
    let runInput;
    try {
      runInput = await store.load();
      console.log('Resuming conversation');
    } catch {
      runInput = [{ role: 'user', content: 'Hello' }];
      console.log('Starting new conversation');
    }

    // Run agent
    let result = await runAgent(agent, runInput, session, false);
    result = await consumeResult(result);

    // Save state for next time
    await store.save(result.state);
    ```

## InterruptionException

An exception class for handling agent execution interruptions (e.g., for approval workflows).

### Class Definition

=== "Python"

    ```python
    class InterruptionException(Exception):
        def __init__(self, message: str = "Agent execution interrupted for approval")
    ```

=== "TypeScript"

    ```typescript
    export class InterruptionException extends Error {
      constructor(message: string = 'Agent execution interrupted for approval')
    }
    ```

### Usage

This exception can be raised when you need to interrupt agent execution for user approval or other reasons.

=== "Python"

    ```python
    from timestep import InterruptionException

    # Raise interruption
    raise InterruptionException("Waiting for user approval")
    ```

=== "TypeScript"

    ```typescript
    import { InterruptionException } from '@timestep-ai/timestep';

    // Throw interruption
    throw new InterruptionException('Waiting for user approval');
    ```

### Example

=== "Python"

    ```python
    from timestep import InterruptionException

    def check_approval():
        if needs_approval:
            raise InterruptionException("Action requires approval")

    try:
        check_approval()
        # Continue execution
    except InterruptionException as e:
        print(f"Interrupted: {e}")
        # Handle interruption
    ```

=== "TypeScript"

    ```typescript
    import { InterruptionException } from '@timestep-ai/timestep';

    function checkApproval() {
      if (needsApproval) {
        throw new InterruptionException('Action requires approval');
      }
    }

    try {
      checkApproval();
      // Continue execution
    } catch (e) {
      if (e instanceof InterruptionException) {
        console.error(`Interrupted: ${e.message}`);
        // Handle interruption
      }
    }
    ```

## See Also

- [Use Cases](../use-cases.md) - For examples of using these utilities
- [Getting Started](../getting-started.md) - For basic agent setup

