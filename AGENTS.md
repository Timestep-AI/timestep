# AGENTS.md - Development Guide for AI Coding Agents

This document provides guidance for AI coding agents working on the Timestep project, including development environment setup, testing instructions, PR guidelines, and project-specific conventions.

## Development Environment

### Prerequisites

- **Python 3.11+** for Python development
- **Node.js 20+** for TypeScript development

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd timestep
   ```

2. **Python setup**:
   ```bash
   cd python
   pip install -e .
   pip install -e ".[dev]"  # For development dependencies
   ```

3. **TypeScript setup**:
   ```bash
   cd typescript
   pnpm install
   ```

4. **Environment variables** (optional):
   ```bash
   export OPENAI_API_KEY="your-key-here"  # For agents that use OpenAI or LLM-as-judge graders
   ```

## Project Structure

The codebase is organized into core and eval modules:

```
timestep/
├── python/timestep/
│   ├── core/              # Core agent-environment loop
│   │   ├── agent.py       # Agent harness interface
│   │   ├── episode.py     # Episode runner (agent-environment loop)
│   │   ├── tools.py        # Tool execution
│   │   └── types.py        # Core types
│   ├── eval/               # Evaluation harness
│   │   ├── suite.py        # Suite runner
│   │   ├── graders.py      # All graders (code-based, LLM-as-judge, outcome)
│   │   └── cli.py          # CLI interface
│   └── utils/              # Utilities (JSONL, hashing, etc.)
└── typescript/timestep/
    ├── core/               # Same structure as Python
    ├── eval/
    └── utils/
```

**Important**: Both Python and TypeScript follow the same structure for cross-language parity.

## Core Concepts

### Agent Harness

An **agent harness** (or scaffold) is a system that enables a model to act as an agent. In Timestep, this is the `AgentFn` interface:

```python
AgentFn = Callable[[List[Message], JSON], Message]
```

The agent harness:
- Takes messages (transcript) and context
- Returns an assistant message (may include `tool_calls`)
- Processes inputs, orchestrates tool calls, and returns results
- Can use any model provider (OpenAI, Anthropic, local models, etc.)

### Agent-Environment Loop

The core execution pattern implemented by `run_episode()` that orchestrates the agent harness:

1. The loop calls the agent harness with messages and context
2. Agent harness returns assistant message
3. If assistant has `tool_calls`: environment executes them and appends tool messages
4. Loop continues (returns to step 1) until final answer (no tool calls) or limits reached

The agent-environment loop orchestrates the agent harness, executing tools and managing the conversation flow. Together, the loop and harness form the complete agent system.

### Evaluation Harness

The evaluation harness builds on the core to:
- Run evaluation suites on multiple tasks
- Grade agent performance using graders
- Generate reports

### Transcript vs Outcome

- **Transcript**: Complete record of an episode (all messages)
- **Outcome**: Final state in environment (separate from transcript)

## Testing

### Running Tests

**Python**:
```bash
cd python
pytest tests/ -v
```

**TypeScript**:
```bash
cd typescript
pnpm test
```

### Test Organization

- Tests are in `tests/` directories
- Test files follow `test_*.py` or `test_*.ts` naming
- Cross-language tests verify task format compatibility

### Writing Tests

1. **Use async/await** for all async operations (if needed)
2. **Clean up resources** (temporary files, etc.)
3. **Test both success and error cases**
4. **Verify cross-language compatibility** when testing task formats

## Code Style and Conventions

### Python

- Follow PEP 8 style guide
- Use type hints for all function signatures
- Docstrings should follow Google style

### TypeScript

- Use TypeScript strict mode
- Prefer `async`/`await` over promises
- Use JSDoc comments for documentation
- Follow the existing code style (2-space indentation)

### Cross-Language Parity

**Critical**: When adding features, ensure both Python and TypeScript implementations:
- Have the same API surface
- Use the same function/class names
- Follow the same parameter naming conventions
- Produce compatible task and result formats

## Pull Request Guidelines

### Before Submitting

1. **Run all tests**: Ensure both Python and TypeScript tests pass
2. **Check imports**: Verify all imports use correct module paths (core vs eval)
3. **Update documentation**: Update relevant docs if adding features
4. **Cross-language testing**: Test that tasks work in both languages

### PR Checklist

- [ ] All tests pass (Python and TypeScript)
- [ ] Code follows project conventions
- [ ] Imports use correct module paths (core.* or eval.*)
- [ ] Documentation updated (if needed)
- [ ] Cross-language compatibility verified (if task-related)
- [ ] No breaking changes (or clearly documented)

### Commit Messages

Use clear, descriptive commit messages:
- `feat: Add new grader for X`
- `fix: Resolve issue with tool execution`
- `refactor: Reorganize core modules`
- `docs: Update README with new examples`

## Common Tasks

### Adding a New Grader

1. Create grader class in `eval/graders.py` (Python) or `eval/graders.ts` (TypeScript)
2. Extend the `Grader` base class
3. Implement `grade()` method
4. Add to `BUILTIN_GRADERS` dictionary
5. Update `parse_grader_spec()` if needed
6. Add tests
7. Update documentation

### Adding a New Tool

1. Create tool function in `core/tools.py` (Python) or `core/tools.ts` (TypeScript)
2. Add to `DEFAULT_TOOLS` dictionary
3. Add tests
4. Update documentation

### Creating an Agent Harness

Agent harnesses are functions that take `(messages, context)` and return an assistant message:

```python
def my_agent(messages: list[Message], context: JSON) -> Message:
    # Use OpenAI library or other model provider
    # Return OpenAI-style assistant message
    # Optionally include usage info for token tracking
    return {
        "role": "assistant",
        "content": "...",
        "tool_calls": [...],
        "usage": {  # Optional
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
```

For command-based agents, use `agent_cmd_factory()`.

## Architecture Overview

### Core Module

The core module provides the agent-environment loop:

1. **Agent harness interface**: `AgentFn` - function that enables a model to act as an agent (takes messages and context, returns assistant message)
2. **Episode runner**: `run_episode()` - orchestrates the agent harness, executes the agent-environment loop
3. **Tool execution**: Deterministic functions with automatic indexing
4. **Episode info**: Tracks steps, tool calls, duration, tokens, costs

The agent-environment loop (`run_episode()`) orchestrates the agent harness (`AgentFn`), executing tools and managing the conversation flow.

### Eval Module

The eval module builds on core to provide evaluation:

1. **Suite runner**: `run_suite()` - runs evaluation suites on multiple tasks
2. **Graders**: Code-based, LLM-as-judge, and outcome verification graders
3. **Reporting**: `report()` - generates summary reports

### Task Format

Tasks are JSON objects with:
- `id`: Unique identifier
- `messages`: List of OpenAI-style messages
- `tools_allowed`: Optional tool allowlist
- `expected`: Optional expected values for graders
- `limits`: Optional episode limits

### Message Protocol

Messages follow OpenAI chat completion format:
- `role`: "system" | "user" | "assistant" | "tool"
- `content`: String content
- `tool_calls`: Array of tool call objects (assistant messages)
- `tool_call_id`: String ID (tool messages)
- `usage`: Optional token usage info (assistant messages)

## Debugging Tips

### Task Format Issues

1. Check JSONL file is valid
2. Verify messages array structure
3. Check tool names match available tools
4. Verify expected values match grader requirements

### Agent Harness Issues

1. Check agent returns valid assistant message
2. Verify tool_calls format if using tools
3. Check agent handles context correctly
4. Test with builtin:echo first

### Cross-Language Issues

1. Verify task JSON is valid in both languages
2. Check result formats match
3. Test with same seed for reproducibility

## Resources

- **Documentation**: https://timestep-ai.github.io/timestep/
- **OpenAI API Reference**: https://platform.openai.com/docs/api-reference

## Notes for AI Agents

- **Always maintain cross-language parity**: Changes in one language should be reflected in the other
- **Test task compatibility**: When modifying task format or results, verify cross-language compatibility
- **Follow the module structure**: Core is independent, eval builds on core
- **Keep it simple**: The SDK is intentionally minimal - avoid over-engineering
- **Do NOT create documentation files unless explicitly requested**: Only create markdown documentation files (like README, architecture docs, etc.) when the user explicitly asks for them. Do not create evaluation documents, mapping documents, or other analysis documents unless specifically requested.
