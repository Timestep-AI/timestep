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
   export OPENAI_API_KEY="your-key-here"  # For agents that use OpenAI
   ```

## Project Structure

The codebase is organized into clear modules:

```
timestep/
├── python/timestep/
│   ├── eval/              # Eval framework core
│   │   ├── agent.py       # Agent function interface
│   │   ├── episode.py     # Episode runner
│   │   ├── tools.py        # Tool execution
│   │   ├── graders.py      # Built-in graders
│   │   ├── suite.py        # Suite runner
│   │   └── cli.py          # CLI interface
│   └── utils/              # Utilities (JSONL, hashing, etc.)
└── typescript/timestep/
    ├── eval/               # Same structure as Python
    └── utils/
```

**Important**: Both Python and TypeScript follow the same structure for cross-language parity.

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
2. **Check imports**: Verify all imports use correct module paths
3. **Update documentation**: Update relevant docs if adding features
4. **Cross-language testing**: Test that tasks work in both languages

### PR Checklist

- [ ] All tests pass (Python and TypeScript)
- [ ] Code follows project conventions
- [ ] Imports use correct module paths
- [ ] Documentation updated (if needed)
- [ ] Cross-language compatibility verified (if task-related)
- [ ] No breaking changes (or clearly documented)

### Commit Messages

Use clear, descriptive commit messages:
- `feat: Add new grader for X`
- `fix: Resolve issue with tool execution`
- `refactor: Reorganize eval modules`
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

1. Create tool function in `eval/tools.py` (Python) or `eval/tools.ts` (TypeScript)
2. Add to `DEFAULT_TOOLS` dictionary
3. Add tests
4. Update documentation

### Creating an Agent Adapter

Agents are functions that take `(messages, context)` and return an assistant message:

```python
def my_agent(messages: list[Message], context: JSON) -> Message:
    # Use OpenAI library or other model provider
    # Return OpenAI-style assistant message
    return {"role": "assistant", "content": "...", "tool_calls": [...]}
```

For command-based agents, use `agent_cmd_factory()`.

## Eval Framework Architecture

### Core Concepts

1. **Agent Function**: `(messages, context) => assistant_message`
   - Takes list of messages and context
   - Returns one assistant message
   - May include `tool_calls` for tool-using agents

2. **Episode Runner**: `run_episode()`
   - Runs agent-environment loop
   - Executes tool calls automatically
   - Tracks steps, tool calls, duration
   - Returns transcript and episode info

3. **Tool Execution**: Deterministic functions
   - Tools are simple functions `(args) => result`
   - Results are JSON-serialized
   - Tool calls are automatically indexed

4. **Graders**: Evaluation functions
   - Take transcript, tool index, task, episode info
   - Return `{name, passed, score, details}`
   - Can be aggregated

5. **Suite Runner**: `run_suite()`
   - Runs multiple trials per task
   - Persists transcripts, tool indices, grades
   - Generates results.jsonl

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

## Debugging Tips

### Task Format Issues

1. Check JSONL file is valid
2. Verify messages array structure
3. Check tool names match available tools
4. Verify expected values match grader requirements

### Agent Issues

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
- **Follow the module structure**: Use the organized folder structure; don't create files at the root level
- **Keep it simple**: The eval framework is intentionally minimal - avoid over-engineering
- **Documentation matters**: Update docs when adding features or changing behavior
