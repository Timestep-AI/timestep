# Use Cases

This document covers common patterns and use cases for Timestep, with practical examples in both Python and TypeScript.

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
    openai_agent = Agent(model="gpt-4")
    
    # Use Ollama for simpler tasks or when privacy is important
    ollama_agent = Agent(model="ollama/llama3")
    
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
    const openaiAgent = new Agent({ model: 'gpt-4' });
    
    // Use Ollama for simpler tasks or when privacy is important
    const ollamaAgent = new Agent({ model: 'ollama/llama3' });
    
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
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
    from agents import Agent, Runner, RunConfig
    import os

    model_provider = MultiModelProvider(
        provider_map=MultiModelProviderMap(),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    agent = Agent(model="gpt-4")
    run_config = RunConfig(model_provider=model_provider)

    # Streaming response
    result = Runner.run_streamed(agent, agent_input, run_config=run_config)
    
    # Process stream events
    async for event in result.stream_events():
        # Handle streaming events
        print(event)
    ```

=== "Python - Non-Streaming"

    ```python
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
    from agents import Agent, Runner, RunConfig
    import os

    model_provider = MultiModelProvider(
        provider_map=MultiModelProviderMap(),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    agent = Agent(model="gpt-4")
    run_config = RunConfig(model_provider=model_provider)

    # Non-streaming response
    result = await Runner.run(agent, agent_input, run_config=run_config)
    
    # Access final result
    print(result.output)
    ```

=== "TypeScript - Streaming"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    const modelProvider = new MultiModelProvider({
      provider_map: new MultiModelProviderMap(),
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });

    const agent = new Agent({ model: 'gpt-4' });
    const runner = new Runner({ modelProvider });

    // Streaming response
    const result = await runner.run(agent, agentInput, { stream: true });
    
    // Process stream
    for await (const event of result.toTextStream()) {
      // Handle streaming events
      console.log(event);
    }
    ```

=== "TypeScript - Non-Streaming"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    const modelProvider = new MultiModelProvider({
      provider_map: new MultiModelProviderMap(),
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });

    const agent = new Agent({ model: 'gpt-4' });
    const runner = new Runner({ modelProvider });

    // Non-streaming response
    const result = await runner.run(agent, agentInput);
    
    // Access final result
    console.log(result.output);
    ```

## Error Handling Patterns

Handle errors gracefully when providers are unavailable or models fail.

=== "Python"

    ```python
    from timestep import MultiModelProvider, OllamaModelProvider
    from agents import Agent, Runner, RunConfig
    from agents.exceptions import AgentsException, ModelBehaviorError

    model_provider = MultiModelProvider(
        provider_map=MultiModelProviderMap(),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    agent = Agent(model="gpt-4")
    run_config = RunConfig(model_provider=model_provider)

    try:
        result = await Runner.run(agent, agent_input, run_config=run_config)
    except ModelBehaviorError as e:
        # Handle model-specific errors
        print(f"Model error: {e}")
        # Fallback to different model
        fallback_agent = Agent(model="ollama/llama3")
        result = await Runner.run(fallback_agent, agent_input, run_config=run_config)
    except AgentsException as e:
        # Handle general agent errors
        print(f"Agent error: {e}")
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap } from '@timestep-ai/timestep';
    import { Agent, Runner, ModelBehaviorError, AgentsError } from '@openai/agents';

    const modelProvider = new MultiModelProvider({
      provider_map: new MultiModelProviderMap(),
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });

    const agent = new Agent({ model: 'gpt-4' });
    const runner = new Runner({ modelProvider });

    try {
      const result = await runner.run(agent, agentInput);
    } catch (e) {
      if (e instanceof ModelBehaviorError) {
        // Handle model-specific errors
        console.error('Model error:', e.message);
        // Fallback to different model
        const fallbackAgent = new Agent({ model: 'ollama/llama3' });
        const result = await runner.run(fallbackAgent, agentInput);
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
    from timestep import run_agent, consume_result
    from agents import Agent, Session
    import os

    # Create agent with model provider
    agent = Agent(model="gpt-4")
    
    # Create session for conversation context
    session = Session()
    
    # First message
    result1 = await run_agent(
        agent,
        [{"role": "user", "content": "Hello, my name is Alice"}],
        session,
        stream=False
    )
    await consume_result(result1)
    
    # Second message - session maintains context
    result2 = await run_agent(
        agent,
        [{"role": "user", "content": "What's my name?"}],
        session,
        stream=False
    )
    await consume_result(result2)
    
    # Agent remembers: "Alice"
    ```

=== "TypeScript"

    ```typescript
    import { runAgent, consumeResult } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    // Create agent with model provider
    const agent = new Agent({ model: 'gpt-4' });
    
    // Create session for conversation context
    const session = new Session();
    
    // First message
    const result1 = await runAgent(
      agent,
      [{ role: 'user', content: 'Hello, my name is Alice' }],
      session,
      false
    );
    await consumeResult(result1);
    
    // Second message - session maintains context
    const result2 = await runAgent(
      agent,
      [{ role: 'user', content: "What's my name?" }],
      session,
      false
    );
    await consumeResult(result2);
    
    // Agent remembers: "Alice"
    ```

## Using Tools

Timestep includes built-in tools like web search. Here's how to use them:

=== "Python"

    ```python
    from timestep import web_search
    from agents import Agent, Runner, RunConfig, Tool
    from timestep import MultiModelProvider

    # Create agent with tools
    tools = [web_search]
    agent = Agent(model="gpt-4", tools=tools)

    model_provider = MultiModelProvider(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    run_config = RunConfig(model_provider=model_provider)
    result = await Runner.run(agent, agent_input, run_config=run_config)
    ```

=== "TypeScript"

    ```typescript
    import { webSearch } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';
    import { MultiModelProvider } from '@timestep-ai/timestep';

    // Create agent with tools
    const tools = [webSearch];
    const agent = new Agent({ model: 'gpt-4', tools });

    const modelProvider = new MultiModelProvider({
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });

    const runner = new Runner({ modelProvider });
    const result = await runner.run(agent, agentInput);
    ```

## Cross-Language Compatibility

Timestep maintains feature parity between Python and TypeScript, allowing teams to share patterns and code structure.

### Shared Patterns

Both implementations support:
- Same model naming conventions
- Same provider mapping approach
- Same error handling patterns
- Same session management

### Language-Specific Notes

**Python:**
- Uses `RunConfig` for configuration
- Uses `Runner.run_streamed()` for streaming
- Uses `await Runner.run()` for non-streaming

**TypeScript:**
- Uses `Runner` constructor options
- Uses `runner.run()` with `{ stream: true }` for streaming
- Uses `await runner.run()` for non-streaming

## Best Practices

1. **Environment Variables**: Always use environment variables for API keys
2. **Error Handling**: Implement fallback strategies for production
3. **Session Management**: Use sessions for multi-turn conversations
4. **Provider Selection**: Choose providers based on task requirements (cost, latency, privacy)
5. **Streaming**: Use streaming for better UX in interactive applications
6. **Testing**: Test with both OpenAI and Ollama to ensure compatibility

