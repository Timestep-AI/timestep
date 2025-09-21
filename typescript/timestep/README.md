# @timestep-ai/timestep

A comprehensive TypeScript library and CLI for building AI agent systems with A2A (Agent-to-Agent) protocol support. Features built-in MCP (Model Context Protocol) server integration, tool management, and works seamlessly in Node.js, Deno, and Supabase Edge Functions.

## Features

- 🤖 **Agent Management** - Create, configure, and manage AI agents with JSONL-based configuration
- 🔄 **A2A Protocol** - Full Agent-to-Agent protocol implementation with streaming support
- ⚡ **AG-UI Protocol** - Real-time streaming frontend interface with Server-Sent Events
- 🛠️ **Tool Integration** - Built-in MCP server with weather, document, and thinking tools plus external MCP server support
- 🌐 **Multi-Runtime** - Works in Node.js, Deno, and Supabase Edge Functions
- 📊 **Model Provider Support** - Supports OpenAI, Anthropic, Ollama, and custom providers
- 🎛️ **CLI Interface** - Beautiful terminal UI for agent interaction and management
- 📚 **API Server** - RESTful API server with Express.js
- 💾 **Context Management** - Persistent conversation contexts and task tracking
- 🔍 **Tracing & Debugging** - Built-in execution tracing and logging
- ⚡ **Built-in Tools** - Weather forecasting, document processing, and advanced reasoning capabilities

## Installation

### Global CLI Installation

```bash
npm install --global @timestep-ai/timestep
```

### Library Installation

```bash
# Node.js/npm
npm install @timestep-ai/timestep

# Deno
deno add npm:@timestep-ai/timestep
```

## Quick Start

### CLI Usage

```bash
# Start the server
timestep server

# Stop the server
timestep stop

# List available agents
timestep list-agents

# Start an interactive chat
timestep chat

# List available tools
timestep list-tools

# List MCP servers
timestep list-mcp-servers

# List model providers
timestep list-model-providers

# List available models
timestep list-models

# List chat contexts
timestep list-chats

# List execution traces
timestep list-traces

# Get version information
timestep get-version

# List all available commands
timestep --help
```

### AG-UI Protocol (Real-time Frontend Interaction)

The AG-UI (Agent User Interaction Protocol) provides real-time streaming interfaces for frontend applications using the **official AG-UI TypeScript SDK** (`@ag-ui/core`, `@ag-ui/client`):

```bash
# Start AG-UI interactive client
node dist/agUiClient.js

# Direct message to specific agent
node dist/agUiClient.js --agentId 00000000-0000-0000-0000-000000000000 --user-input "Hello!"

# AG-UI REST endpoints (using official @ag-ui/core types and events)
curl http://localhost:8080/ag-ui/agents/discover
curl -X POST http://localhost:8080/ag-ui/agents/AGENT_ID/run \
  -H "Content-Type: application/json" \
  -d '{"runId":"123","threadId":"thread","messages":[{"role":"user","parts":[{"kind":"text","text":"Hello!"}],"messageId":"msg1"}]}'
```

The AG-UI protocol features:
- **Real-time streaming** with Server-Sent Events using official @ag-ui/core events
- **Agent discovery** with capabilities and status information
- **Event-driven communication** using official EventType enum (RUN_STARTED, TEXT_MESSAGE_START/CONTENT/END, RUN_FINISHED, RUN_ERROR)
- **Official TypeScript SDK integration** with @ag-ui/core types and interfaces
- **Side-by-side operation** with existing A2A protocol

### Library Usage (Node.js)

```typescript
import {
	listAgents,
	TimestepAIAgentExecutor,
	startDenoServer,
} from '@timestep-ai/timestep';

// List configured agents
const agents = await listAgents();
console.log(agents.data);

// Create an agent executor for A2A protocol
const executor = new TimestepAIAgentExecutor();

// Start the API server
startDenoServer(3000);
```

### Library Usage (Deno/Supabase)

```typescript
// Use individual library functions for full control
import {
	listAgents,
	listModels,
	handleAgentRequest,
	TimestepAIAgentExecutor,
} from 'npm:@timestep-ai/timestep@latest';

const agentExecutor = new TimestepAIAgentExecutor();

Deno.serve({port}, async (request: Request) => {
	const url = new URL(request.url);

	if (url.pathname === '/agents') {
		const result = await listAgents();
		return new Response(JSON.stringify(result.data));
	}

	if (url.pathname === '/models') {
		const result = await listModels();
		return new Response(JSON.stringify(result.data));
	}

	// Handle dynamic agent A2A endpoints
	const agentMatch = url.pathname.match(/^\/agents\/([^\/]+)(?:\/.*)?$/);
	if (agentMatch) {
		const mockReq = {
			method: request.method,
			path: url.pathname,
			params: {agentId: agentMatch[1]},
			body: await request.json().catch(() => ({})),
		};
		const result = await handleAgentRequest(
			mockReq,
			null,
			null,
			taskStore,
			agentExecutor,
			port,
		);
		return new Response(JSON.stringify(result));
	}

	return new Response('Not found', {status: 404});
});
```

## API Endpoints

When running the server, the following endpoints are available:

### Core Endpoints

- `GET /agents` - List available agents
- `GET /models` - List available models
- `GET /tools` - List available tools
- `GET /traces` - List execution traces
- `GET /chats` - List chat contexts
- `GET /model_providers` - List model providers
- `GET /mcp_servers` - List MCP servers

### MCP Server Endpoints

- `GET /mcp_servers/{serverId}` - Health check for specific MCP server
- `POST /mcp_servers/{serverId}` - Execute MCP server requests
- `DELETE /mcp_servers/{serverId}` - Disable MCP server

### A2A Protocol Endpoints

- `/agents/{agentId}/*` - Dynamic agent endpoints following A2A protocol

### AG-UI Protocol Endpoints

- `GET /ag-ui/agents/discover` - Discover available agents with capabilities
- `POST /ag-ui/agents/{agentId}/run` - Stream real-time agent interactions (Server-Sent Events)

### Utility Endpoints

- `GET /version` - Get version information

## Configuration

Timestep uses a configuration directory structure:

```
~/.config/timestep/           # Linux/macOS
%APPDATA%/timestep/           # Windows
~/Library/Application Support/timestep/  # macOS

├── app.json                  # App configuration (ports, etc.)
├── agents.jsonl             # Agent configurations
├── model_providers.jsonl    # Model provider settings
├── mcp_servers.jsonl       # MCP server configurations
└── contexts.jsonl          # Chat contexts and history
```

### Example Agent Configuration

```json
{
	"id": "assistant",
	"name": "General Assistant",
	"instructions": "You are a helpful AI assistant.",
	"toolIds": ["get-alerts", "get-forecast", "think", "markdownToPdf"],
	"model": "gpt-4",
	"modelSettings": {"temperature": 0.7}
}
```

### Example Model Provider Configuration

```json
{
	"id": "openai",
	"provider": "openai",
	"api_key": "your-api-key",
	"base_url": "https://api.openai.com/v1",
	"models_url": "https://api.openai.com/v1/models"
}
```

## Deno Support

Timestep has first-class Deno support with optimized imports and compatibility:

### Running in Deno

```bash
# Add dependencies
deno add npm:express npm:@a2a-js/sdk

# Run the Deno server
deno task deno-server

# Development mode with auto-reload
deno task deno-server:dev
```

### Supabase Edge Functions

Deploy Timestep in Supabase Edge Functions using individual library functions:

```typescript
import {
	listAgents,
	listModels,
	handleAgentRequest,
	TimestepAIAgentExecutor,
} from 'npm:@timestep-ai/timestep@latest';

// Custom task store for Supabase environment
class SupabaseTaskStore {
	// Implementation for your specific storage needs
}

const agentExecutor = new TimestepAIAgentExecutor();
const taskStore = new SupabaseTaskStore();

Deno.serve({port}, async (request: Request) => {
	// Handle all Timestep endpoints with full control
	// Integrate with Supabase auth, database, etc.
});
```

See `examples/supabase-edge-function.ts` for a complete working example.

## Development

### Node.js Development

```bash
# Clone the repository
git clone https://github.com/timestep-ai/timestep.git
cd timestep/typescript/timestep

# Install dependencies
npm install

# Build the project
npm run build

# Run the CLI locally
node dist/cli.js --help

# Start the server
node dist/server.js
```

### Deno Development

```bash
# Run with Deno
deno task deno-server

# Development mode
deno task deno-server:dev

# Run original server with Deno
deno task server
```

### Available Scripts

- `npm run build` - Build TypeScript to JavaScript
- `npm run dev` - Development mode with file watching
- `deno task deno-server` - Start Deno-optimized server
- `deno task deno-server:dev` - Deno server with file watching

## Architecture

### Core Components

- **Agent Executor** - Handles A2A protocol execution and tool calling
- **Context Service** - Manages conversation contexts and task history
- **Model Provider** - Abstracts different AI model APIs
- **Agent Factory** - Creates and configures agents from JSON definitions
- **MCP Integration** - Connects to external tool servers via MCP protocol

### File Structure

```
src/
├── api/                     # API endpoints and handlers
│   ├── agentsApi.ts        # Agent management endpoints
│   ├── modelsApi.ts        # Model listing endpoints
│   └── settings/           # Settings API endpoints
├── core/                   # Core business logic
│   ├── agent_executor.ts   # A2A protocol implementation
│   └── context_service.ts  # Context management
├── services/               # Service layer
│   ├── agent_factory.ts    # Agent creation and configuration
│   └── model_provider.ts   # Model provider abstraction
├── types/                  # TypeScript type definitions
├── utils.ts               # Utility functions
├── server.ts              # Node.js/Express server
├── denoServer.ts          # Deno-optimized server
├── app.tsx                # CLI React interface
└── cli.ts                 # CLI entry point
```

## Examples

The `examples/` directory contains:

- `supabase-edge-function.ts` - Complete Supabase Edge Function implementation with custom request handling and A2A protocol support

### Built-in Tools

Timestep includes a built-in MCP server with the following tools:

- **`get-alerts`** - Get weather alerts for a specific location
- **`get-forecast`** - Get weather forecast for coordinates
- **`think`** - Advanced reasoning and problem-solving tool
- **`markdownToPdf`** - Convert markdown content to PDF

These tools are automatically available to all agents and can be used alongside external MCP servers.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT © [Timestep AI](https://github.com/timestep-ai)
