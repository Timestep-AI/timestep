# @timestep-ai/timestep

TypeScript library and CLI for AI agent systems with A2A and MCP protocol support. Addresses discrepancies between OpenAI's Agents SDK Python and TypeScript libraries, providing unified tool-calling, state management, and third-party model integration.

**Free & Open Source** - MIT licensed. See [LICENSE](../../LICENSE) for details.

This library powers [Timestep AI](https://timestep.ai) - the open core for our upcoming public platform focused on personal AI control with a default multi-agent assistant system.

## Features

- **A2A Protocol** - Agent-to-Agent communication with streaming
- **OpenAI Agents SDK** - Built on OpenAI's Agents SDK with unified Python/TypeScript compatibility
- **Human-in-the-Loop** - Default approval for every tool call with fine-grained control - step back as you gain comfort
- **MCP Integration** - Built-in tools + easy connection to any MCP server (Rube, Beeper Desktop, etc.)
- **Multi-Runtime** - Node.js, Deno, and Supabase Edge Functions
- **Flexible Storage** - Default JSONL files, overrideable with custom repositories
- **CLI Interface** - Terminal UI for agent interaction
- **API Server** - RESTful endpoints for programmatic access
- **Context Management** - Persistent conversation contexts
- **Built-in Tools** - Weather, document processing, reasoning
- **Unified Model Support** - Working Ollama integration and consistent third-party model handling
- **Your API Keys** - Add your own API keys for any model provider to power your agents
- **Chat App Integration** - Connect to [Rube](https://rube.app/) and [Beeper Desktop](https://developers.beeper.com/desktop-api/mcp) for access to all your messaging apps

## Installation

```bash
# Global CLI
npm install --global @timestep-ai/timestep

# Library
npm install @timestep-ai/timestep
```

## Quick Start

```bash
# Start server
timestep server

# Chat with agents
timestep chat

# List agents
timestep list-agents

# List tools
timestep list-tools
```

## Library Usage

```typescript
import {TimestepAIAgentExecutor, listAgents} from '@timestep-ai/timestep';

// List agents
const agents = await listAgents();

// Create executor
const executor = new TimestepAIAgentExecutor();
```

## API Endpoints

- `GET /agents` - List agents
- `GET /models` - List models
- `GET /tools` - List tools
- `GET /mcp_servers` - List MCP servers
- `/agents/{agentId}/*` - A2A protocol endpoints

## Configuration

By default, configuration is stored in `~/.config/timestep/` as JSONL files:

- `agents.jsonl` - Agent configurations
- `model_providers.jsonl` - Model provider settings
- `mcp_servers.jsonl` - MCP server configurations
- `contexts.jsonl` - Chat contexts and history

You can override the default JSONL storage with custom repositories. See `examples/supabase-edge-function.ts` for a complete PostgreSQL/Supabase implementation.

## Deno Support

```bash
# Run with Deno
deno task deno-server

# Development mode
deno task deno-server:dev
```

See `examples/supabase-edge-function.ts` for Supabase Edge Functions example.

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Start server
node dist/server.js

# Development mode
npm run dev

# Run tests
npm test

# Run tests with coverage
npm run test:coverage
```

## Testing

The project includes comprehensive test coverage with 80% minimum coverage requirements:

### Test Coverage

- **80% Minimum Coverage** across the entire library
- **Comprehensive Test Suite** covering all major functionality
- **Real Code Paths** - Tests exercise actual methods and logic
- **Proper Mocking** - All dependencies properly mocked

### Key Test Areas

- **Agent Executor** - Core execution logic and A2A protocol handling
- **Context Management** - Conversation contexts and task history
- **Tool Integration** - Tool calling and approval workflows
- **Error Handling** - Edge cases and error scenarios
- **Streaming Logic** - Real-time event processing
- **Type Mapping** - A2A ↔ Agents SDK format conversion

### Coverage Reports

Coverage reports are generated in multiple formats:

- **Text format**: Displayed in terminal
- **HTML format**: `coverage/index.html` (open in browser)
- **JSON format**: `coverage/coverage-final.json`

## Architecture

- **Agent Executor** - A2A protocol execution and tool calling
- **Context Service** - Conversation contexts and task history
- **Model Provider** - AI model API abstraction
- **Agent Factory** - Agent creation from JSON definitions
- **MCP Integration** - External tool server connections

## Built-in Tools

- **`get-alerts`** - Weather alerts
- **`get-forecast`** - Weather forecast
- **`think`** - Advanced reasoning
- **`markdownToPdf`** - Convert markdown to PDF

## License

MIT © [Timestep AI](https://github.com/timestep-ai)
