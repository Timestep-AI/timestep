# AGENTS.md - Development Guide for AI Coding Agents

This document provides guidance for AI coding agents working on the Timestep project, including development environment setup, testing instructions, PR guidelines, and project-specific conventions.

## Development Environment

### Prerequisites

- **Python 3.11+** for Python development
- **Node.js 20+** for TypeScript development
- **Bun** for TypeScript examples (uses inline dependencies)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd timestep
   ```

2. **Python setup** (for examples):
   ```bash
   cd examples/python
   # Dependencies are inline in Python files using PEP 723
   ```

3. **TypeScript setup** (for examples):
   ```bash
   cd examples/typescript
   # Dependencies are auto-installed by Bun from import statements
   ```

4. **Environment variables**:
   ```bash
   export OPENAI_API_KEY="your-key-here"  # Required for A2A agents using OpenAI
   ```

## Project Structure

The codebase is organized around A2A and MCP protocols:

```
timestep/
├── lib/                    # Future library code (to be extracted)
│   ├── python/             # Python library (reserved)
│   └── typescript/         # TypeScript library (reserved)
├── examples/               # Working examples
│   ├── python/             # Python A2A/MCP examples
│   │   ├── a2a_server.py   # A2A server (Task-generating Agent)
│   │   ├── mcp_server.py   # MCP server with handoff tool
│   │   ├── test_client.py  # Client orchestrating A2A/MCP
│   │   └── compose.yml     # Docker Compose setup
│   └── typescript/         # TypeScript examples (pending v2 SDK)
│       ├── a2a_server.ts
│       ├── mcp_server.ts
│       ├── test_client.ts
│       └── compose.yml
└── app/                    # Web UI
    ├── index.html
    ├── index.css
    └── index.js
```

**Important**: The library code will be extracted from the working examples in `examples/` once the API surface and patterns are well-established.

## Core Concepts

### A2A Protocol

The [Agent-to-Agent (A2A) Protocol](https://a2a-protocol.org/latest/specification/) standardizes how independent AI agents communicate and collaborate as peers. Key concepts:

- **Agent Card**: Discovery mechanism for agent capabilities and endpoints
- **Task**: Stateful interaction object that tracks progress
- **Task States**: `created`, `input-required`, `completed`, `canceled`, etc.
- **Context ID**: Groups related interactions across multiple tasks

### Task-Generating Agents

Following the [A2A Task-generating Agents philosophy](https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task), our agents always respond with Task objects (never just Messages). This provides:

- Clear state management
- Progress visibility
- Support for multi-turn interactions
- Structured tool call communication

### A2A input-required with DataPart

When an agent needs to call tools:

1. Agent sets task state to `input-required`
2. Agent includes a `DataPart` in the task status message
3. The `DataPart.data` contains tool calls in structured format:
   ```json
   {
     "tool_calls": [
       {
         "function": {
           "name": "handoff",
           "arguments": "{\"agent_uri\": \"...\", \"message\": \"...\"}"
         }
       }
     ]
   }
   ```
4. Client extracts tool calls from `DataPart` and executes via MCP

### MCP Protocol

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/specification/latest) provides:

- **Tools**: Functions for agents to execute
- **Sampling**: Server-initiated LLM interactions (used for handoffs)
- **Resources**: Context and data for agents
- **Prompts**: Templated messages and workflows

### MCP Sampling for Handoffs

MCP's sampling feature enables server-initiated agentic behaviors. We use it for handoffs:

1. Agent calls MCP `handoff` tool with target agent URI
2. MCP server calls client's sampling callback via `ctx.session.create_message()`
3. Sampling callback makes A2A request to target agent
4. Target agent processes and responds
5. Response returned to MCP server
6. MCP server returns result to original agent

This allows agents to delegate work to specialized peers without the original agent needing direct A2A client capabilities.

## Architecture

### A2A Server

The A2A server implements a Task-generating Agent:

- Uses A2A SDK to handle protocol operations
- Always responds with Task objects
- Manages task lifecycle and state transitions
- Includes tool calls in `DataPart` when `input-required`
- Supports streaming for real-time updates

### MCP Server

The MCP server provides:

- Tool registration (e.g., `handoff`, `get_weather`)
- Sampling callback registration (for handoffs)
- HTTP transport for client connections
- Tool execution and result formatting

### Client

The client orchestrates A2A and MCP:

- Connects to A2A server to send messages
- Monitors task state transitions
- Extracts tool calls from `DataPart` when `input-required`
- Calls MCP tools and sends results back
- Implements MCP sampling callback for handoffs

## Testing

### Running Examples

**Python:**
```bash
make test-example-python
```

**TypeScript:**
```bash
make test-example-typescript  # Currently shows pending v2 message
```

### Test Organization

- Examples are in `examples/` directories
- Each example includes A2A server, MCP server, and test client
- Docker Compose files orchestrate the services

## Code Style and Conventions

### Python

- Follow PEP 8 style guide
- Use type hints for all function signatures
- Docstrings should follow Google style
- Use inline dependencies (PEP 723) in script files

### TypeScript

- Use TypeScript strict mode
- Prefer `async`/`await` over promises
- Use JSDoc comments for documentation
- Follow the existing code style (2-space indentation)
- Use Bun's auto-install for dependencies (no package.json)

### Cross-Language Parity

**Critical**: When adding features, ensure both Python and TypeScript implementations:
- Have the same API surface
- Use the same function/class names
- Follow the same parameter naming conventions
- Produce compatible A2A Task and MCP message formats

## Pull Request Guidelines

### Before Submitting

1. **Run examples**: Ensure Python examples work end-to-end
2. **Check protocol compliance**: Verify A2A and MCP protocol adherence
3. **Update documentation**: Update relevant docs if adding features
4. **Test handoffs**: Verify handoff flow works correctly

### PR Checklist

- [ ] Examples run successfully (Python)
- [ ] Code follows project conventions
- [ ] A2A and MCP protocol compliance verified
- [ ] Documentation updated (if needed)
- [ ] Handoff functionality tested
- [ ] No breaking changes (or clearly documented)

### Commit Messages

Use clear, descriptive commit messages:
- `feat: Add new MCP tool for X`
- `fix: Resolve issue with A2A handoff`
- `refactor: Reorganize MCP server structure`
- `docs: Update README with A2A/MCP details`

## Common Tasks

### Adding a New MCP Tool

1. Add tool function to `examples/python/mcp_server.py` (or TypeScript equivalent)
2. Register tool with MCP server
3. Update A2A server to include tool in agent's tool list
4. Test tool execution via client
5. Update documentation

### Adding a New Agent

1. Create agent executor in A2A server
2. Define agent card with capabilities
3. Register agent with A2A server
4. Add agent-specific tools if needed
5. Test agent via client
6. Update documentation

### Implementing Handoffs

1. Ensure MCP server has `handoff` tool
2. Implement sampling callback in client
3. Sampling callback should make A2A request to target agent
4. Test handoff flow end-to-end
5. Verify task state transitions

## Resources

- **A2A Protocol Specification**: https://a2a-protocol.org/latest/specification/
- **A2A Task-generating Agents**: https://a2a-protocol.org/latest/topics/life-of-a-task/#agent-response-message-or-task
- **MCP Protocol Specification**: https://modelcontextprotocol.io/specification/latest
- **Documentation**: https://timestep-ai.github.io/timestep/

## Notes for AI Agents

- **Always maintain cross-language parity**: Changes in one language should be reflected in the other
- **Follow A2A and MCP specifications**: Ensure protocol compliance
- **Test handoffs**: Verify handoff functionality when making changes
- **Keep it simple**: The library is intentionally minimal - avoid over-engineering
- **Do NOT create documentation files unless explicitly requested**: Only create markdown documentation files (like README, architecture docs, etc.) when the user explicitly asks for them. Do not create evaluation documents, mapping documents, or other analysis documents unless specifically requested.
