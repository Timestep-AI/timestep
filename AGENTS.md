# AGENTS.md - Development Guide for AI Coding Agents

This document provides guidance for AI coding agents working on the Timestep project, including development environment setup, testing instructions, PR guidelines, and project-specific conventions.

## Development Environment

### Prerequisites

- **Python 3.11+** for Python development
- **Node.js 20+** for TypeScript development
- **PostgreSQL** (optional, for production-like testing)
- **Docker** (optional, for testcontainers)

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
   pip install -r requirements-docs.txt  # For documentation
   ```

3. **TypeScript setup**:
   ```bash
   cd typescript
   pnpm install
   ```

4. **Environment variables**:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export PG_CONNECTION_URI="postgresql://user:pass@host/db"  # Optional
   export FIRECRAWL_API_KEY="your-key-here"  # For web search tool
   ```

### Database Setup

For testing, you can use:
- **PGLite** (default): Automatically managed, no setup required
- **PostgreSQL**: Set `PG_CONNECTION_URI` environment variable
- **Testcontainers**: Used in tests for isolated PostgreSQL instances

## Project Structure

The codebase is organized into clear modules:

```
timestep/
├── core/              # Core agent execution functions
├── config/            # Configuration utilities
├── stores/            # Data access layer
│   ├── agent_store/   # Agent persistence
│   ├── session_store/ # Session persistence
│   ├── run_state_store/ # Run state persistence
│   └── shared/        # Shared DB utilities
├── tools/             # Agent tools
├── model_providers/   # Model provider implementations
└── models/            # Model implementations
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
- Cross-language tests verify state compatibility between Python and TypeScript

### Writing Tests

1. **Use async/await** for all async operations
2. **Clean up resources** (close connections, clear state)
3. **Test both success and error cases**
4. **Verify cross-language compatibility** when testing state persistence

## Code Style and Conventions

### Python

- Follow PEP 8 style guide
- Use type hints for all function signatures
- Use `async`/`await` for async operations
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
- Produce compatible state formats

## Pull Request Guidelines

### Before Submitting

1. **Run all tests**: Ensure both Python and TypeScript tests pass
2. **Check imports**: Verify all imports use the new module structure
3. **Update documentation**: Update relevant docs if adding features
4. **Cross-language testing**: Test that state can be saved in one language and loaded in the other

### PR Checklist

- [ ] All tests pass (Python and TypeScript)
- [ ] Code follows project conventions
- [ ] Imports use correct module paths
- [ ] Documentation updated (if needed)
- [ ] Cross-language compatibility verified (if state-related)
- [ ] No breaking changes (or clearly documented)

### Commit Messages

Use clear, descriptive commit messages:
- `feat: Add new tool for X`
- `fix: Resolve issue with state loading`
- `refactor: Reorganize stores module`
- `docs: Update README with new structure`

## Common Tasks

### Adding a New Tool

1. Create tool file in `tools/` directory (e.g., `tools/my_tool.py` or `tools/my_tool.ts`)
2. Export from `tools/__init__.py` or `tools/index.ts`
3. Update main `__init__.py` or `index.ts` to export
4. Add tests in `tests/test_tools.py` or `tests/test_tools.ts`
5. Update documentation

### Adding a New Model Provider

1. Create provider in `model_providers/` directory
2. Create model implementation in `models/` directory
3. Update `MultiModelProvider` to support new prefix
4. Add tests
5. Update documentation

### Modifying Store Functions

**Important**: Store functions manage their own database connections internally. Do not pass `db` parameters to public store functions.

- Public functions: `save_agent(agent)`, `load_agent(agent_id)`, etc.
- Internal functions: `_save_agent_internal(agent, db)`, `_load_agent_internal(agent_id, db)`, etc.

### Working with DBOS Workflows

- Workflows are in `core/agent_workflow.py`/`core/agent_workflow.ts`
- Use `@DBOS.workflow()` decorator (Python) or `DBOS.registerWorkflow()` (TypeScript)
- Steps use `@DBOS.step()` decorator (Python) or `DBOS.registerStep()` (TypeScript)
- Always ensure DBOS is configured before using workflows

## Debugging Tips

### State Persistence Issues

1. Check database connection string
2. Verify schema is initialized
3. Check that state is being serialized correctly
4. Verify cross-language state format compatibility

### Import Errors

- Ensure imports use the new module structure
- Check that `__init__.py` files export correctly
- Verify TypeScript exports in `index.ts`

### DBOS Workflow Issues

1. Ensure `configure_dbos()`/`configureDBOS()` is called
2. Check that `ensure_dbos_launched()`/`ensureDBOSLaunched()` is called
3. Verify connection string is available
4. Check workflow registration

## Resources

- **Documentation**: https://timestep-ai.github.io/timestep/
- **OpenAI Agents SDK**: See `_vendored/` directory for vendored SDK code
- **DBOS Documentation**: https://dbos.dev/

## Notes for AI Agents

- **Always maintain cross-language parity**: Changes in one language should be reflected in the other
- **Test state compatibility**: When modifying state-related code, verify cross-language compatibility
- **Follow the module structure**: Use the organized folder structure; don't create files at the root level
- **Store functions are self-contained**: They manage their own DB connections; don't pass `db` parameters
- **Documentation matters**: Update docs when adding features or changing behavior

