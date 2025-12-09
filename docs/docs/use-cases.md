# Use Cases

This document covers common patterns and use cases for Timestep, with practical examples in both Python and TypeScript, focusing on durable execution and cross-language state persistence.

## Durable Execution with Interruptions

One of Timestep's core features is durable execution with built-in state persistence. This enables resumable workflows and human-in-the-loop patterns.

=== "Python"

    ```python
    from timestep import run_agent, RunStateStore,     from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()
    state_store = RunStateStore(
        agent=agent,
        session_id=await session._get_session_id()
    )

    # Run agent
    result = await run_agent(agent, input_items, session, stream=False)
    result = await (result)

    # Handle interruptions (e.g., tool calls requiring approval)
    if result.interruptions:
        # Save state for later resume
        state = result.to_state()
        await state_store.save(state)
        
        # Load state and approve interruptions
        loaded_state = await state_store.load()
        for interruption in loaded_state.get_interruptions():
            loaded_state.approve(interruption)
        
        # Resume execution
        result = await run_agent(agent, loaded_state, session, stream=False)
        result = await (result)
    ```

=== "TypeScript"

    ```typescript
    import { runAgent, RunStateStore,  } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();
    const stateStore = new RunStateStore({
      agent,
      sessionId: await session.getSessionId()
    });

    // Run agent
    let result = await runAgent(agent, inputItems, session, false);
    result = await (result);

    // Handle interruptions (e.g., tool calls requiring approval)
    if (result.interruptions?.length) {
      // Save state for later resume
      await stateStore.save(result.state);
      
      // Load state and approve interruptions
      const loadedState = await stateStore.load();
      for (const interruption of loadedState.getInterruptions()) {
        loadedState.approve(interruption);
      }
      
      // Resume execution
      result = await runAgent(agent, loadedState, session, false);
      result = await (result);
    }
    ```

## Cross-Language State Transfer

Timestep's unique feature is the ability to start execution in one language and resume in another, enabling flexible deployment architectures.

### Python → TypeScript

Start execution in Python, interrupt for tool approval, and resume in TypeScript:

=== "Python: Start and Save"

    ```python
    from timestep import run_agent, RunStateStore
    from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()
    state_store = RunStateStore(
        agent=agent,
        session_id=await session._get_session_id()
    )

    # Run until interruption
    result = await run_agent(agent, input_items, session, stream=False)
    result = await (result)

    if result.interruptions:
        # Save state - can be loaded in TypeScript!
        state = result.to_state()
        await state_store.save(state)
        session_id = await session._get_session_id()
        print(f"State saved. Resume in TypeScript with session_id: {session_id}")
    ```

=== "TypeScript: Resume"

    ```typescript
    import { runAgent, RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();
    const stateStore = new RunStateStore({
      agent,
      sessionId: await session.getSessionId()
    });

    // Load state saved from Python
    const savedState = await stateStore.load();

    // Approve interruptions
    for (const interruption of savedState.getInterruptions()) {
      savedState.approve(interruption);
    }

    // Resume execution
    const result = await runAgent(agent, savedState, session, false);
    ```

### TypeScript → Python

Start execution in TypeScript, interrupt for tool approval, and resume in Python:

=== "TypeScript: Start and Save"

    ```typescript
    import { runAgent, RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();
    const stateStore = new RunStateStore({
      agent,
      sessionId: await session.getSessionId()
    });

    // Run until interruption
    let result = await runAgent(agent, inputItems, session, false);
    result = await (result);

    if (result.interruptions?.length) {
      // Save state - can be loaded in Python!
      await stateStore.save(result.state);
      const sessionId = await session.getSessionId();
      console.log(`State saved. Resume in Python with session_id: ${sessionId}`);
    }
    ```

=== "Python: Resume"

    ```python
    from timestep import run_agent, RunStateStore
    from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()
    state_store = RunStateStore(
        agent=agent,
        session_id=await session._get_session_id()
    )

    # Load state saved from TypeScript
    saved_state = await state_store.load()

    # Approve interruptions
    for interruption in saved_state.get_interruptions():
        saved_state.approve(interruption)

    # Resume execution
    result = await run_agent(agent, saved_state, session, False)
    ```

## Switching Between OpenAI and Ollama

One of the most common use cases is switching between different model providers based on your needs (cost, performance, privacy, etc.).

=== "Python"

    ```python
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
    from agents import Agent, Runner, RunConfig
    import os

    # Setup provider with both OpenAI and Ollama
    model_provider_map = MultiModelProviderMap()
    if os.environ.get("OLLAMA_API_KEY"):
        model_provider_map.add_provider(
            "ollama",
            OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
        )

    model_provider = MultiModelProvider(
        provider_map=model_provider_map,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    # Use OpenAI for complex tasks
    openai_agent = Agent(model="gpt-4.1")
    
    # Use Ollama for simpler tasks or when privacy is important
    # Note: -cloud suffix uses Ollama Cloud (https://ollama.com/cloud)
    ollama_agent = Agent(model="ollama/gpt-oss:20b-cloud")
    
    # Both use the same provider
    run_config = RunConfig(model_provider=model_provider)
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    // Setup provider with both OpenAI and Ollama
    const modelProviderMap = new MultiModelProviderMap();
    if (Deno.env.get('OLLAMA_API_KEY')) {
      modelProviderMap.addProvider(
        'ollama',
        new OllamaModelProvider({ apiKey: Deno.env.get('OLLAMA_API_KEY') })
      );
    }

    const modelProvider = new MultiModelProvider({
      provider_map: modelProviderMap,
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });

    // Use OpenAI for complex tasks
    const openaiAgent = new Agent({ model: 'gpt-4.1' });
    
    // Use Ollama for simpler tasks or when privacy is important
    // Note: -cloud suffix uses Ollama Cloud (https://ollama.com/cloud)
    const ollamaAgent = new Agent({ model: 'ollama/gpt-oss:20b-cloud' });
    
    // Both use the same provider
    const runner = new Runner({ modelProvider });
    ```

## Custom Provider Setup

Create custom provider mappings for specialized use cases, such as multiple Ollama instances or custom endpoints.

=== "Python"

    ```python
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider

    # Create custom mapping with multiple Ollama instances
    provider_map = MultiModelProviderMap()
    
    # Local Ollama for development
    provider_map.add_provider(
        "local",
        OllamaModelProvider(base_url="http://localhost:11434")
    )
    
    # Remote Ollama for production
    provider_map.add_provider(
        "remote",
        OllamaModelProvider(base_url="http://ollama-server:11434")
    )
    
    # Ollama Cloud for specific use cases
    provider_map.add_provider(
        "cloud",
        OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
    )
    
    model_provider = MultiModelProvider(provider_map=provider_map)
    
    # Use different instances based on prefix
    dev_agent = Agent(model="local/llama3")
    prod_agent = Agent(model="remote/llama3")
    cloud_agent = Agent(model="cloud/llama3")
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';

    // Create custom mapping with multiple Ollama instances
    const providerMap = new MultiModelProviderMap();
    
    // Local Ollama for development
    providerMap.addProvider(
      'local',
      new OllamaModelProvider({ baseURL: 'http://localhost:11434' })
    );
    
    // Remote Ollama for production
    providerMap.addProvider(
      'remote',
      new OllamaModelProvider({ baseURL: 'http://ollama-server:11434' })
    );
    
    // Ollama Cloud for specific use cases
    providerMap.addProvider(
      'cloud',
      new OllamaModelProvider({ apiKey: Deno.env.get('OLLAMA_API_KEY') })
    );
    
    const modelProvider = new MultiModelProvider({ provider_map: providerMap });
    
    // Use different instances based on prefix
    const devAgent = new Agent({ model: 'local/llama3' });
    const prodAgent = new Agent({ model: 'remote/llama3' });
    const cloudAgent = new Agent({ model: 'cloud/llama3' });
    ```

## Streaming vs Non-Streaming

Timestep supports both streaming and non-streaming responses. Choose based on your UX requirements.

=== "Python - Streaming"

    ```python
    from timestep import run_agent,     from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()

    # Streaming response
    result = await run_agent(agent, input_items, session, stream=True)
    
    # Process stream events
    async for event in result.stream_events():
        # Handle streaming events
        print(event)
    
    # Ensure all events are consumed
    result = await (result)
    ```

=== "Python - Non-Streaming"

    ```python
    from timestep import run_agent,     from agents import Agent, Session

    agent = Agent(model="gpt-4.1")
    session = Session()

    # Non-streaming response
    result = await run_agent(agent, input_items, session, stream=False)
    result = await (result)
    
    # Access final result
    print(result.final_output)
    ```

=== "TypeScript - Streaming"

    ```typescript
    import { runAgent,  } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();

    // Streaming response
    const result = await runAgent(agent, inputItems, session, true);
    
    // Process stream
    for await (const event of result.toTextStream()) {
      // Handle streaming events
      console.log(event);
    }
    
    // Ensure all events are consumed
    await (result);
    ```

=== "TypeScript - Non-Streaming"

    ```typescript
    import { runAgent,  } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();

    // Non-streaming response
    let result = await runAgent(agent, inputItems, session, false);
    result = await (result);
    
    // Access final result
    console.log(result.finalOutput);
    ```

## Error Handling Patterns

Handle errors gracefully when providers are unavailable or models fail.

=== "Python"

    ```python
    from timestep import run_agent
    from agents import Agent, Session
    from agents.exceptions import AgentsException, ModelBehaviorError

    agent = Agent(model="gpt-4.1")
    session = Session()

    try:
        result = await run_agent(agent, input_items, session, stream=False)
    except ModelBehaviorError as e:
        # Handle model-specific errors
        print(f"Model error: {e}")
        # Fallback to different model (Ollama Cloud, note: -cloud suffix)
        fallback_agent = Agent(model="ollama/gpt-oss:20b-cloud")
        result = await run_agent(fallback_agent, input_items, session, stream=False)
    except AgentsException as e:
        # Handle general agent errors
        print(f"Agent error: {e}")
    ```

=== "TypeScript"

    ```typescript
    import { runAgent } from '@timestep-ai/timestep';
    import { Agent, Session, ModelBehaviorError, AgentsError } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4.1' });
    const session = new Session();

    try {
      const result = await runAgent(agent, inputItems, session, false);
    } catch (e) {
      if (e instanceof ModelBehaviorError) {
        // Handle model-specific errors
        console.error('Model error:', e.message);
        // Fallback to different model (Ollama Cloud, note: -cloud suffix)
        const fallbackAgent = new Agent({ model: 'ollama/gpt-oss:20b-cloud' });
        const result = await runAgent(fallbackAgent, inputItems, session, false);
      } else if (e instanceof AgentsError) {
        // Handle general agent errors
        console.error('Agent error:', e.message);
      }
    }
    ```

## Session Management

Use sessions to maintain conversation context across multiple agent runs.

=== "Python"

    ```python
    from timestep import run_agent,     from agents import Agent, Session

    # Create agent with model provider
    agent = Agent(model="gpt-4.1")
    
    # Create session for conversation context
    session = Session()
    
    # First message
    result1 = await run_agent(
        agent,
        [{"role": "user", "content": "Hello, my name is Alice"}],
        session,
        stream=False
    )
    await (result1)
    
    # Second message - session maintains context
    result2 = await run_agent(
        agent,
        [{"role": "user", "content": "What's my name?"}],
        session,
        stream=False
    )
    await (result2)
    
    # Agent remembers: "Alice"
    ```

=== "TypeScript"

    ```typescript
    import { runAgent,  } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    // Create agent with model provider
    const agent = new Agent({ model: 'gpt-4.1' });
    
    // Create session for conversation context
    const session = new Session();
    
    // First message
    const result1 = await runAgent(
      agent,
      [{ role: 'user', content: 'Hello, my name is Alice' }],
      session,
      false
    );
    await (result1);
    
    // Second message - session maintains context
    const result2 = await runAgent(
      agent,
      [{ role: 'user', content: "What's my name?" }],
      session,
      false
    );
    await (result2);
    
    // Agent remembers: "Alice"
    ```

## Using Tools

Timestep includes built-in tools like web search. Here's how to use them:

=== "Python"

    ```python
    from timestep import web_search, run_agent
    from agents import Agent, Session

    # Create agent with tools
    tools = [web_search]
    agent = Agent(model="gpt-4.1", tools=tools)
    session = Session()

    result = await run_agent(agent, input_items, session, stream=False)
    result = await (result)
    ```

=== "TypeScript"

    ```typescript
    import { webSearch, runAgent } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    // Create agent with tools
    const tools = [webSearch];
    const agent = new Agent({ model: 'gpt-4.1', tools });
    const session = new Session();

    const result = await runAgent(agent, inputItems, session, false);
    const finalResult = await (result);
    ```

## Cross-Language Compatibility

Timestep maintains feature parity between Python and TypeScript, allowing teams to share patterns and code structure.

### Shared Patterns

Both implementations support:
- Same model naming conventions
- Same provider mapping approach
- Same error handling patterns
- Same session management
- **Same state format for cross-language persistence**

### Language-Specific Notes

**Python:**
- Uses `run_agent()` for execution
- Uses `RunStateStore` for state persistence
- Uses `()` for result handling

**TypeScript:**
- Uses `runAgent()` for execution
- Uses `RunStateStore` for state persistence
- Uses `()` for result handling

## Best Practices

1. **Environment Variables**: Always use environment variables for API keys
2. **Error Handling**: Implement fallback strategies for production
3. **Session Management**: Use sessions for multi-turn conversations
4. **State Persistence**: Save state at interruption points for resumability
5. **Cross-Language**: Leverage cross-language state transfer for flexible deployments
6. **Provider Selection**: Choose providers based on task requirements (cost, latency, privacy)
7. **Streaming**: Use streaming for better UX in interactive applications
8. **Testing**: Test with both OpenAI and Ollama to ensure compatibility
