# Timestep Documentation

Welcome to the Timestep documentation! Timestep is a clean, RISC-style abstraction layer over the OpenAI Agents SDK, providing durable execution and cross-language state persistence for AI agent workflows.

## Architecture Philosophy: RISC vs CISC

Timestep is built on the principle that the OpenAI Agents SDK is like a **CISC (Complex Instruction Set Computer)** architecture—powerful and feature-rich, but with many abstractions and layers that can be complex to work with. Timestep provides a **RISC (Reduced Instruction Set Computer)** approach—a simpler, cleaner interface that exposes the fundamental components needed for building robust agent systems.

### What This Means

- **Simpler API**: Clean abstractions that focus on essential operations
- **Durable Execution**: Built-in support for state persistence and resumable workflows
- **Cross-Language Parity**: Identical behavior and APIs in Python and TypeScript
- **Composable**: Fundamental building blocks that can be combined as needed

## Packages

- **[Python Package (`timestep`)](https://pypi.org/project/timestep/)** - Available on PyPI
- **[TypeScript Package (`@timestep-ai/timestep`)](https://www.npmjs.com/package/@timestep-ai/timestep)** - Available on npm

## Quick Navigation

### Getting Started
- [Installation and Quick Start](getting-started.md) - Get up and running with Timestep in minutes

### Core Concepts
- [Architecture](architecture.md) - Understand how Timestep works under the hood, including RISC design principles
- [Use Cases](use-cases.md) - Common patterns and real-world examples, including durable execution

### API Reference
- [Utilities](api-reference/utilities.md) - Core utilities: `run_agent`, `RunStateStore`, `consume_result`
- [MultiModelProvider](api-reference/multi-model-provider.md) - Automatic model routing
- [OllamaModelProvider](api-reference/ollama-model-provider.md) - Ollama integration
- [MultiModelProviderMap](api-reference/multi-model-provider-map.md) - Custom provider mappings
- [Tools](api-reference/tools.md) - Built-in tools like web search

## Core Features

### 🔄 Durable Execution

Timestep enables durable, resumable agent workflows with built-in state persistence:

- **Interrupt and Resume**: Pause agent execution at any point and resume later
- **Cross-Language State Transfer**: Start in Python, interrupt for tool approval, and resume in TypeScript (or vice versa)
- **Persistent Sessions**: Agent state is stored in a database (DBOS/PGLite) for durability
- **Human-in-the-Loop**: Seamlessly handle interruptions for approvals, tool calls, or user input

### 🌐 Cross-Language State Persistence

Timestep provides identical implementations in Python and TypeScript with full state compatibility:

=== "Python"

    ```python
    from timestep import run_agent, RunStateStore
    from agents import Agent, Session

    agent = Agent(model="gpt-4")
    session = Session()
    state_store = RunStateStore("state.json", agent)

    # Run agent and handle interruption
    result = await run_agent(agent, input_items, session, stream=False)
    if result.interruptions:
        # Save state for cross-language resume
        await state_store.save(result.state)
    ```

=== "TypeScript"

    ```typescript
    import { runAgent, RunStateStore } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    const agent = new Agent({ model: 'gpt-4' });
    const session = new Session();
    const stateStore = new RunStateStore('state.json', agent);

    // Load state saved from Python
    const savedState = await stateStore.load();

    // Approve interruptions and continue
    for (const interruption of savedState.getInterruptions()) {
      savedState.approve(interruption);
    }

    // Resume execution
    const result = await runAgent(agent, savedState, session, false);
    ```

### 🔌 Multi-Model Provider Support

Timestep also supports multiple AI model providers through a unified interface:

- **OpenAI**: Full support for all OpenAI models
- **Ollama**: Support for both local Ollama instances and Ollama Cloud
- **Automatic Routing**: Model names with prefixes (e.g., `ollama/llama3`) automatically route to the correct provider
- **Extensible**: Add custom providers using `MultiModelProviderMap`

## Quick Example

=== "Python"

    ```python
    from timestep import run_agent, RunStateStore, consume_result
    from agents import Agent, Session

    # Create agent
    agent = Agent(model="gpt-4")
    session = Session()
    state_store = RunStateStore("agent_state.json", agent)

    # Run with durable state
    result = await run_agent(agent, input_items, session, stream=False)
    result = await consume_result(result)

    # Handle interruptions
    if result.interruptions:
        # Save state for later resume (even in TypeScript!)
        await state_store.save(result.state)
    ```

=== "TypeScript"

    ```typescript
    import { runAgent, RunStateStore, consumeResult } from '@timestep-ai/timestep';
    import { Agent, Session } from '@openai/agents';

    // Create agent
    const agent = new Agent({ model: 'gpt-4' });
    const session = new Session();
    const stateStore = new RunStateStore('agent_state.json', agent);

    // Run with durable state
    let result = await runAgent(agent, inputItems, session, false);
    result = await consumeResult(result);

    // Handle interruptions
    if (result.interruptions?.length) {
      // Save state for later resume (even in Python!)
      await stateStore.save(result.state);
    }
    ```

## Durable Execution with DBOS/PGLite

Timestep uses **DBOS** (Database Operating System) and **PGLite** (PostgreSQL in WebAssembly) to provide durable execution:

- **State Persistence**: Agent run states are stored in PostgreSQL
- **Cross-Language Compatibility**: State format is identical between Python and TypeScript
- **Resumable Workflows**: Load any saved state and continue execution
- **Database Schema**: See [database/README.md](../../database/README.md) for the complete schema

## Requirements

### Python
- Python >=3.11
- ollama >=0.6.0
- openai-agents >=0.4.2

### TypeScript
- Node.js >=20
- @openai/agents-core ^0.2.1
- @openai/agents-openai ^0.2.1
- ollama ^0.6.2

## License

MIT License - see [LICENSE](https://github.com/Timestep-AI/timestep/blob/main/LICENSE) file for details.
