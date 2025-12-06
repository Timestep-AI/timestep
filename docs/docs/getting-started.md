# Getting Started

This guide will help you get up and running with Timestep in both Python and TypeScript, focusing on durable execution and cross-language state persistence.

## Prerequisites

- `OPENAI_API_KEY`
- Python default storage: Node.js + `@electric-sql/pglite` on PATH (Python shells out to Node today).
- Better performance: set `TIMESTEP_DB_URL` to Postgres. If you must stay on PGLite with Python, keep a long-lived Node/Deno sidecar that holds a `PGlite` connection instead of spawning Node per query.

## Installation

=== "Python"

    Install Timestep using pip:

    ```bash
    pip install timestep
    ```

    Or using uv:

    ```bash
    uv add timestep
    ```

=== "TypeScript"

    Install Timestep using npm:

    ```bash
    npm install @timestep-ai/timestep
    ```

    Or using pnpm:

    ```bash
    pnpm add @timestep-ai/timestep
    ```

    Or using yarn:

    ```bash
    yarn add @timestep-ai/timestep
    ```

## Prerequisites

### OpenAI API Key

You'll need an OpenAI API key to use OpenAI models. Set it as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Ollama Setup (Optional)

Timestep supports both **local Ollama instances** and **Ollama Cloud**. You can use either or both!

#### Option 1: Ollama Cloud (Recommended for Production)

Ollama Cloud provides managed Ollama instances with no local setup required:

1. Get an API key from [Ollama Cloud](https://ollama.com/cloud)
2. Set it as an environment variable:

```bash
export OLLAMA_API_KEY="your-ollama-cloud-key"
```

That's it! No installation or local setup needed. Timestep will automatically use Ollama Cloud when the API key is set.

#### Option 2: Local Ollama

For local development or when you want to run models on your own infrastructure:

1. Install [Ollama](https://ollama.com/)
2. Start the Ollama service (usually runs on `http://localhost:11434`)
3. Pull the models you want to use: `ollama pull llama3`

Timestep will automatically detect and use your local Ollama instance if no API key is provided.

## Quick Start: Durable Execution

The core feature of Timestep is durable execution with cross-language state persistence. Let's start with a simple example:

=== "Python"

    ```python
    from timestep import run_agent, RunStateStore, consume_result
    from agents import Agent, Session

    # Create agent
    agent = Agent(model="gpt-4")
    session = Session()
    # RunStateStore uses PGLite by default (stored in ~/.config/timestep/pglite/)
    state_store = RunStateStore(agent=agent, session_id=await session._get_session_id())

    # Run agent
    result = await run_agent(agent, input_items, session, stream=False)
    result = await consume_result(result)

    # Handle interruptions
    if result.interruptions:
        # Save state for later resume (even in TypeScript!)
        state = result.to_state()
        await state_store.save(state)
        
        # Load state and approve interruptions
        loaded_state = await state_store.load()
        for interruption in loaded_state.get_interruptions():
            loaded_state.approve(interruption)
        
        # Resume execution
        result = await run_agent(agent, loaded_state, session, stream=False)
        result = await consume_result(result)
    ```

=== "TypeScript"

    ```typescript
    import { runAgent, RunStateStore, consumeResult } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    // Create agent
    const agent = new Agent({ model: 'gpt-4' });
    const session = new Session();
    // RunStateStore uses PGLite by default (stored in ~/.config/timestep/pglite/)
    const stateStore = new RunStateStore({ 
      agent, 
      sessionId: await session.getSessionId() 
    });

    // Run agent
    let result = await runAgent(agent, inputItems, session, false);
    result = await consumeResult(result);

    // Handle interruptions
    if (result.interruptions?.length) {
      // Save state for later resume (even in Python!)
      await stateStore.save(result.state);
      
      // Load state and approve interruptions
      const loadedState = await stateStore.load();
      for (const interruption of loadedState.getInterruptions()) {
        loadedState.approve(interruption);
      }
      
      // Resume execution
      result = await runAgent(agent, loadedState, session, false);
      result = await consumeResult(result);
    }
    ```

## Cross-Language State Transfer

One of Timestep's unique features is the ability to start execution in one language and resume in another:

=== "Python → TypeScript"

    ```python
    # Python: Start agent and save state at interruption
    from timestep import run_agent, RunStateStore
    from agents import Agent, Session

    agent = Agent(model="gpt-4")
    session = Session()
    # Uses PGLite in shared app directory (~/.config/timestep)
    state_store = RunStateStore(agent=agent, session_id=await session._get_session_id())

    # Run until interruption
    result = await run_agent(agent, input_items, session, stream=False)
    result = await consume_result(result)

    if result.interruptions:
        # Save state - can be loaded in TypeScript!
        state = result.to_state()
        await state_store.save(state)
        session_id = await session._get_session_id()
        print(f"State saved. Resume in TypeScript with session_id: {session_id}")
    ```

    ```typescript
    // TypeScript: Load Python state and resume
    import { runAgent, RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4' });
    const session = new Session();
    // Uses PGLite in shared app directory (~/.config/timestep)
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

=== "TypeScript → Python"

    ```typescript
    // TypeScript: Start agent and save state at interruption
    import { runAgent, RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4' });
    const session = new Session();
    // Uses PGLite in shared app directory (~/.config/timestep)
    const stateStore = new RunStateStore({ 
      agent, 
      sessionId: await session.getSessionId() 
    });

    // Run until interruption
    let result = await runAgent(agent, inputItems, session, false);
    result = await consumeResult(result);

    if (result.interruptions?.length) {
      // Save state - can be loaded in Python!
      await stateStore.save(result.state);
      const sessionId = await session.getSessionId();
      console.log(`State saved. Resume in Python with session_id: ${sessionId}`);
    }
    ```

    ```python
    # Python: Load TypeScript state and resume
    from timestep import run_agent, RunStateStore
    from agents import Agent, Session

    agent = Agent(model="gpt-4")
    session = Session()
    # Uses PGLite in shared app directory (~/.config/timestep)
    state_store = RunStateStore(agent=agent, session_id=await session._get_session_id())

    # Load state saved from TypeScript
    saved_state = await state_store.load()

    # Approve interruptions
    for interruption in saved_state.get_interruptions():
        saved_state.approve(interruption)

    # Resume execution
    result = await run_agent(agent, saved_state, session, False)
    ```

## Multi-Model Provider Support

Timestep also provides multi-model provider support for OpenAI and Ollama:

=== "Python"

    ```python
    from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider
    from agents import Agent, Runner, RunConfig
    import os

    # Create a provider map and add Ollama support (works with both local and cloud)
    model_provider_map = MultiModelProviderMap()

    # Add Ollama Cloud support (if API key is set)
    if os.environ.get("OLLAMA_API_KEY"):
        model_provider_map.add_provider(
            "ollama",
            OllamaModelProvider(api_key=os.environ.get("OLLAMA_API_KEY"))
        )
    # Or use local Ollama by omitting the API key:
    # model_provider_map.add_provider("ollama", OllamaModelProvider())

    # Create MultiModelProvider with OpenAI fallback
    model_provider = MultiModelProvider(
        provider_map=model_provider_map,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    # Create agent with model name
    agent = Agent(model="gpt-4")  # Uses OpenAI by default
    # Or: agent = Agent(model="ollama/llama3")  # Uses Ollama

    # Run agent with RunConfig
    run_config = RunConfig(model_provider=model_provider)
    result = Runner.run_streamed(agent, agent_input, run_config=run_config)
    ```

=== "TypeScript"

    ```typescript
    import { MultiModelProvider, MultiModelProviderMap, OllamaModelProvider } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    // Create a provider map and add Ollama support (works with both local and cloud)
    const modelProviderMap = new MultiModelProviderMap();

    // Add Ollama Cloud support (if API key is set)
    if (Deno.env.get('OLLAMA_API_KEY')) {
      modelProviderMap.addProvider(
        'ollama',
        new OllamaModelProvider({
          apiKey: Deno.env.get('OLLAMA_API_KEY'),
        })
      );
    }
    // Or use local Ollama by omitting the API key:
    // modelProviderMap.addProvider('ollama', new OllamaModelProvider());

    // Create MultiModelProvider with OpenAI fallback
    const modelProvider = new MultiModelProvider({
      provider_map: modelProviderMap,
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });

    // Create agent with model name
    const agent = new Agent({ model: 'gpt-4' }); // Uses OpenAI by default
    // Or: new Agent({ model: 'ollama/llama3' }) // Uses Ollama

    // Run agent with Runner
    const runner = new Runner({ modelProvider });
    const result = await runner.run(agent, agentInput, { stream: true });
    ```

### Using OllamaModelProvider Directly

If you only need Ollama models, you can use `OllamaModelProvider` directly. It supports both **Ollama Cloud** and **local Ollama instances**:

=== "Python"

    ```python
    from timestep import OllamaModelProvider
    from agents import Agent, Runner, RunConfig

    # Option 1: Use Ollama Cloud (recommended for production)
    cloud_provider = OllamaModelProvider(api_key="your-ollama-cloud-key")

    # Option 2: Use local Ollama instance (for development)
    local_provider = OllamaModelProvider()  # Defaults to localhost:11434

    # Create agent and run with Ollama Cloud
    agent = Agent(model="llama3")
    run_config = RunConfig(model_provider=cloud_provider)
    result = Runner.run_streamed(agent, agent_input, run_config=run_config)
    ```

=== "TypeScript"

    ```typescript
    import { OllamaModelProvider } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';

    // Option 1: Use Ollama Cloud (recommended for production)
    const cloudProvider = new OllamaModelProvider({
      apiKey: 'your-ollama-cloud-key',
    });

    // Option 2: Use local Ollama instance (for development)
    const localProvider = new OllamaModelProvider(); // Defaults to localhost:11434

    // Create agent and run with Ollama Cloud
    const agent = new Agent({ model: 'llama3' });
    const runner = new Runner({ modelProvider: cloudProvider });
    const result = await runner.run(agent, agentInput, { stream: true });
    ```

## Model Naming Conventions

Timestep uses model name prefixes to determine which provider to use:

- **No prefix** (e.g., `gpt-4`): Defaults to OpenAI
- **`openai/` prefix** (e.g., `openai/gpt-4`): Explicitly uses OpenAI
- **`ollama/` prefix** (e.g., `ollama/llama3`): Uses Ollama

### Examples

=== "Python"

    ```python
    # OpenAI models
    agent1 = Agent(model="gpt-4")           # Uses OpenAI
    agent2 = Agent(model="openai/gpt-4")    # Also uses OpenAI

    # Ollama models
    agent3 = Agent(model="ollama/llama3")    # Uses Ollama (local or cloud)
    agent4 = Agent(model="ollama/llama3-cloud")  # Uses Ollama Cloud
    ```

=== "TypeScript"

    ```typescript
    // OpenAI models
    const agent1 = new Agent({ model: 'gpt-4' });        // Uses OpenAI
    const agent2 = new Agent({ model: 'openai/gpt-4' }); // Also uses OpenAI

    // Ollama models
    const agent3 = new Agent({ model: 'ollama/llama3' });      // Uses Ollama (local or cloud)
    const agent4 = new Agent({ model: 'ollama/llama3-cloud' }); // Uses Ollama Cloud
    ```

## Environment Variables

Here's a summary of the environment variables you might need:

| Variable | Description | Required For |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | Your OpenAI API key | OpenAI models |
| `OLLAMA_API_KEY` | Your Ollama Cloud API key | **Ollama Cloud** (highly recommended - no local setup needed!) |
| `FIRECRAWL_API_KEY` | Firecrawl API key for web search tool | Web search functionality |

!!! tip "Ollama Cloud vs Local"
    - **Ollama Cloud**: Set `OLLAMA_API_KEY` - no installation or local setup required, perfect for production
    - **Local Ollama**: Omit `OLLAMA_API_KEY` - requires local Ollama installation, great for development and offline use

## Durable Execution with PGLite

Timestep uses **PGLite** (PostgreSQL in WebAssembly) as the default storage backend for durable execution:

- **PGLite by Default**: No setup required - works out of the box
- **Cross-Platform App Directory**: State stored in platform-appropriate directories (shared between Python and TypeScript):
  - Linux: `~/.config/timestep/pglite/`
  - macOS: `~/Library/Application Support/timestep/pglite/`
  - Windows: `%APPDATA%/timestep/pglite/`
- **PostgreSQL Option**: Use full PostgreSQL by setting `TIMESTEP_DB_URL` environment variable
- **State Persistence**: Agent run states are stored in the database
- **Cross-Language Compatibility**: State format is identical between Python and TypeScript
- **Resumable Workflows**: Load any saved state and continue execution

### Python PGLite Setup

For Python, PGLite runs via Node.js subprocess. You need:

1. **Install Node.js**: Download from [nodejs.org](https://nodejs.org/)
2. **Install PGLite**:
   ```bash
   npm install -g @electric-sql/pglite
   ```
   Or install locally in your project:
   ```bash
   npm install @electric-sql/pglite
   ```

The database schema supports:
- Agent definitions and configurations
- Run state snapshots for resumability
- Session history and conversation items
- Tool calls and interruptions
- Usage metrics and cost tracking

## Next Steps

- Learn about the [Architecture](architecture.md) to understand how Timestep works, including RISC design principles
- Explore [Use Cases](use-cases.md) for common patterns and examples, including durable execution workflows
- Check out the [API Reference](api-reference/utilities.md) for detailed documentation on `run_agent`, `RunStateStore`, and other utilities
