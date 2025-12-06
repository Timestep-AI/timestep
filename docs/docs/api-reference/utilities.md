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
        stream: bool,
        result_processor: Optional[Callable[[Any], Awaitable[Any]]] = default_result_processor
    ) -> Any
    ```

=== "TypeScript"

    ```typescript
    export async function runAgent(
      agent: Agent,
      runInput: AgentInputItem[] | RunState<any, any>,
      session: Session,
      stream: boolean,
      resultProcessor?: (result: any) => Promise<any>
    ): Promise<any>
    ```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Agent` | The agent to run. |
| `run_input` / `runInput` | `list[TResponseInputItem] \| RunState` / `AgentInputItem[] \| RunState` | The input for the agent run. Can be a list of input items or a RunState. |
| `session` | `SessionABC` / `Session` | The session to use for maintaining conversation context. |
| `stream` | `bool` / `boolean` | Whether to use streaming mode. |
| `result_processor` / `resultProcessor` | `Optional[Callable[[Any], Awaitable[Any]]]` / `(result: any) => Promise<any> \| undefined` | Optional function to process the result. Defaults to `default_result_processor`/`defaultResultProcessor` which consumes all streaming events and waits for completion. Pass `None`/`undefined` to skip processing. |

### Returns

| Type | Description |
|------|-------------|
| `RunResult` / `RunResultStreaming` | The result of the agent run. |

### Features

- **Session Management**: Automatically handles session input callbacks
- **Error Handling**: Catches and logs common agent errors (MaxTurnsExceeded, ModelBehaviorError, UserError, AgentsException)
- **Streaming Support**: Supports both streaming and non-streaming modes
- **Configuration**: Uses default RunConfig settings (nest_handoff_history=False)
- **Result Processing**: Automatically processes results by default (consumes streaming events, waits for completion). Can be customized via `result_processor`/`resultProcessor` parameter.

### Example

=== "Python"

    ```python
    from timestep import run_agent
    from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
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
    
    # Result is automatically processed (streaming events consumed, completion awaited)
    print(result.output)
    ```

=== "TypeScript"

    ```typescript
    import { runAgent } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
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
    
    // Result is automatically processed (streaming events consumed, completion awaited)
    console.log(result.output);
    ```

## default_result_processor / defaultResultProcessor

The default result processor function that consumes all events from a result (streaming or non-streaming), ensuring all background operations complete. This is used automatically by `run_agent`/`runAgent` unless a custom processor is provided.

### Function Signature

=== "Python"

    ```python
    async def default_result_processor(result: Any) -> Any
    ```

=== "TypeScript"

    ```typescript
    export async function defaultResultProcessor(result: any): Promise<any>
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

### Custom Result Processors

You can provide a custom result processor to `run_agent`/`runAgent` to handle results differently:

=== "Python"

    ```python
    from timestep import run_agent, default_result_processor
    from agents import Agent, Session

    async def custom_processor(result):
        # Process events incrementally
        if hasattr(result, 'stream_events'):
            async for event in result.stream_events():
                # Handle each event
                print(f"Event: {event}")
        return result

    agent = Agent(model="gpt-4.1")
    session = Session()

    # Use custom processor
    result = await run_agent(
        agent,
        [{"role": "user", "content": "Hello"}],
        session,
        stream=True,
        result_processor=custom_processor
    )

    # Or skip processing entirely
    result = await run_agent(
        agent,
        [{"role": "user", "content": "Hello"}],
        session,
        stream=True,
        result_processor=None
    )
    ```

=== "TypeScript"

    ```typescript
    import { runAgent, defaultResultProcessor } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    async function customProcessor(result: any) {
      // Process events incrementally
      if ('toTextStream' in result) {
        const stream = result.toTextStream({ compatibleWithNodeStreams: true });
        for await (const chunk of stream) {
          // Handle each chunk
          console.log(`Chunk: ${chunk}`);
        }
        await result.completed;
      }
      return result;
    }

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();

    // Use custom processor
    const result = await runAgent(
      agent,
      [{ role: 'user', content: 'Hello' }],
      session,
      true,
      customProcessor
    );

    // Or skip processing entirely
    const result2 = await runAgent(
      agent,
      [{ role: 'user', content: 'Hello' }],
      session,
      true,
      undefined
    );
    ```

## RunStateStore

A utility class for persisting and loading agent run state to/from a database (PGLite by default, or PostgreSQL). Useful for saving and resuming agent conversations with cross-language compatibility.

### Class Definition

=== "Python"

    ```python
    class RunStateStore:
        def __init__(
            self,
            agent: Agent,
            session_id: Optional[str] = None,
            connection_string: Optional[str] = None,
            use_pglite: Optional[bool] = None,
            pglite_path: Optional[str] = None
        )
        async def save(self, state: Any) -> None
        async def load(self) -> Any
        async def clear(self) -> None
        async def close(self) -> None
    ```

=== "TypeScript"

    ```typescript
    export class RunStateStore {
      constructor(options: {
        agent: Agent;
        sessionId?: string;
        connectionString?: string;
        usePglite?: boolean;
        pglitePath?: string;
      })
      async save(state: any): Promise<void>
      async load(): Promise<any>
      async clear(): Promise<void>
      async close(): Promise<void>
    }
    ```

### Constructor

=== "Python"

    ```python
    def __init__(
        self,
        agent: Agent,
        session_id: Optional[str] = None,
        connection_string: Optional[str] = None,
        use_pglite: Optional[bool] = None,
        pglite_path: Optional[str] = None
    )
    ```

=== "TypeScript"

    ```typescript
    constructor(options: {
      agent: Agent;
      sessionId?: string;
      connectionString?: string;
      usePglite?: boolean;
      pglitePath?: string;
    })
    ```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Agent` | The agent instance. Required for loading state. |
| `session_id` / `sessionId` | `string \| undefined` | Session ID to use as identifier. If not provided, will be generated automatically. |
| `connection_string` / `connectionString` | `string \| undefined` | PostgreSQL connection string. If not provided, uses PGLite (default). |
| `use_pglite` / `usePglite` | `boolean \| undefined` | Whether to use PGLite. Defaults to `True` if no connection string is provided. |
| `pglite_path` / `pglitePath` | `string \| undefined` | Path for PGLite data directory. Defaults to platform-specific app directory. |

### Methods

#### `save()`

Saves the run state to the database.

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
    from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()
    state_store = RunStateStore(
        agent=agent,
        session_id=await session._get_session_id()
    )

    # Save state
    await state_store.save(run_state)
    ```

=== "TypeScript"

    ```typescript
    import { RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();
    const stateStore = new RunStateStore({
      agent,
      sessionId: await session.getSessionId()
    });

    // Save state
    await stateStore.save(runState);
    ```

#### `load()`

Loads run state from the database.

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
    from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()
    state_store = RunStateStore(
        agent=agent,
        session_id=await session._get_session_id()
    )

    # Load state
    run_state = await state_store.load()
    ```

=== "TypeScript"

    ```typescript
    import { RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();
    const stateStore = new RunStateStore({
      agent,
      sessionId: await session.getSessionId()
    });

    // Load state
    const runState = await stateStore.load();
    ```

#### `clear()`

Marks the state as inactive (soft delete) in the database.

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
    from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()
    state_store = RunStateStore(
        agent=agent,
        session_id=await session._get_session_id()
    )

    # Clear saved state
    await state_store.clear()
    ```

=== "TypeScript"

    ```typescript
    import { RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();
    const stateStore = new RunStateStore({
      agent,
      sessionId: await session.getSessionId()
    });

    // Clear saved state
    await stateStore.clear();
    ```

#### `close()`

Closes the database connection.

=== "Python"

    ```python
    async def close(self) -> None
    ```

=== "TypeScript"

    ```typescript
    async close(): Promise<void>
    ```

##### Example

=== "Python"

    ```python
    # Close connection when done
    await state_store.close()
    ```

=== "TypeScript"

    ```typescript
    // Close connection when done
    await stateStore.close();
    ```

### Complete Example

=== "Python"

    ```python
    from timestep import run_agent, RunStateStore
    from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()
    state_store = RunStateStore(
        agent=agent,
        session_id=await session._get_session_id()
    )

    # Try to load existing state
    try:
        run_state = await state_store.load()
        print("Resuming conversation")
    except FileNotFoundError:
        run_state = [{"role": "user", "content": "Hello"}]
        print("Starting new conversation")

    # Run agent
    result = await run_agent(agent, run_state, session, stream=False)

    # Handle interruptions
    if result.interruptions:
        # Save state for later resume
        state = result.to_state()
        await state_store.save(state)
        
        # Load and approve interruptions
        loaded_state = await state_store.load()
        for interruption in loaded_state.get_interruptions():
            loaded_state.approve(interruption)
        
        # Resume execution
        result = await run_agent(agent, loaded_state, session, stream=False)

    # Save state for next time
    await state_store.save(result.to_state())
    
    # Close connection when done
    await state_store.close()
    ```

=== "TypeScript"

    ```typescript
    import { runAgent, RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();
    const stateStore = new RunStateStore({
      agent,
      sessionId: await session.getSessionId()
    });

    // Try to load existing state
    let runInput;
    try {
      runInput = await stateStore.load();
      console.log('Resuming conversation');
    } catch {
      runInput = [{ role: 'user', content: 'Hello' }];
      console.log('Starting new conversation');
    }

    // Run agent
    let result = await runAgent(agent, runInput, session, false);

    // Handle interruptions
    if (result.interruptions?.length) {
      // Save state for later resume
      await stateStore.save(result.state);
      
      // Load and approve interruptions
      const loadedState = await stateStore.load();
      for (const interruption of loadedState.getInterruptions()) {
        loadedState.approve(interruption);
      }
      
      // Resume execution
      result = await runAgent(agent, loadedState, session, false);
    }

    // Save state for next time
    await stateStore.save(result.state);
    
    // Close connection when done
    await stateStore.close();
    ```

### Cross-Language State Transfer Example

`RunStateStore` enables seamless state transfer between Python and TypeScript using a shared database (PGLite or PostgreSQL):

=== "Python → TypeScript"

    ```python
    # Python: Save state
    from timestep import run_agent, RunStateStore
    from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()
    session_id = await session._get_session_id()
    state_store = RunStateStore(agent=agent, session_id=session_id)

    result = await run_agent(agent, input_items, session, stream=False)

    if result.interruptions:
        # Save state - can be loaded in TypeScript!
        state = result.to_state()
        await state_store.save(state)
        print(f"State saved. Resume in TypeScript with session_id: {session_id}")
    ```

    ```typescript
    // TypeScript: Load Python state and resume
    import { runAgent, RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();
    const sessionId = await session.getSessionId();
    const stateStore = new RunStateStore({ agent, sessionId });

    // Load state saved from Python (using same session_id)
    const savedState = await stateStore.load();

    // Approve interruptions
    for (const interruption of savedState.getInterruptions()) {
      savedState.approve(interruption);
    }

    // Resume execution
    const result = await runAgent(agent, savedState, session, false);
    ```

=== "TypeScript → Python"

    ```typescript
    // TypeScript: Save state
    import { runAgent, RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();
    const sessionId = await session.getSessionId();
    const stateStore = new RunStateStore({ agent, sessionId });

    let result = await runAgent(agent, inputItems, session, false);

    if (result.interruptions?.length) {
      // Save state - can be loaded in Python!
      await stateStore.save(result.state);
      console.log(`State saved. Resume in Python with session_id: ${sessionId}`);
    }
    ```

    ```python
    # Python: Load TypeScript state and resume
    from timestep import run_agent, RunStateStore
    from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()
    session_id = await session._get_session_id()
    state_store = RunStateStore(agent=agent, session_id=session_id)

    # Load state saved from TypeScript (using same session_id)
    saved_state = await state_store.load()

    # Approve interruptions
    for interruption in saved_state.get_interruptions():
        saved_state.approve(interruption)

    # Resume execution
    result = await run_agent(agent, saved_state, session, False)
    ```

## See Also

- [Use Cases](../use-cases.md) - For examples of using these utilities
- [Getting Started](../getting-started.md) - For basic agent setup
