# Database Schema for OpenAI Agents Libraries

This directory contains the database schema for backing the OpenAI Agents Python and JavaScript libraries.

## Overview

The schema supports both `openai-agents-python` and `openai-agents-js` libraries, providing a comprehensive data model for:
- Agent definitions and configurations
- Agent execution runs and state management
- Conversation sessions and history
- Tools, handoffs, and guardrails
- Usage metrics and cost tracking
- Run state persistence for resumability

## Database Requirements

- **PostgreSQL 12+** (required for JSONB, UUID, and other features)
- **UUID extension** (`uuid-ossp`) - automatically enabled in migrations

## Schema Structure

### Core Tables

1. **agents** - Agent definitions and configurations
2. **model_settings** - Model-specific configuration parameters
3. **runs** - Agent execution instances
4. **sessions** - Conversation history sessions
5. **session_items** - Individual items in session history
6. **run_items** - Items generated during runs
7. **model_responses** - Raw LLM responses
8. **tools** - Tool definitions
9. **agent_tools** - Agent-tool relationships
10. **handoff_configs** - Handoff configurations
11. **agent_handoffs** - Agent handoff relationships
12. **guardrails** - Guardrail definitions
13. **agent_guardrails** - Agent-guardrail relationships
14. **guardrail_results** - Guardrail execution results
15. **tool_guardrails** - Tool-specific guardrail definitions
16. **tool_guardrail_results** - Tool guardrail execution results
17. **tool_calls** - Tool call executions
18. **interruptions** - Run interruptions (approval requests, etc.)
19. **mcp_servers** - MCP server configurations
20. **agent_mcp_servers** - Agent-MCP server relationships
21. **usage_metrics** - Aggregated usage metrics
22. **request_usage_entries** - Per-request usage breakdown
23. **run_states** - Serialized RunState snapshots for resumability

## Installation

### Run the Migration

```bash
psql -U your_user -d your_database -f migrations/001_initial_schema.sql
```

Or using a connection string:

```bash
psql postgresql://user:password@localhost/dbname -f migrations/001_initial_schema.sql
```

## Design Features

### Data Integrity
- Explicit FOREIGN KEY constraints with appropriate ON DELETE policies
- CHECK constraints for business logic validation
- NOT NULL constraints on critical fields
- UNIQUE constraints where needed

### Performance
- Comprehensive indexing strategy:
  - Indexes on all foreign keys
  - Composite indexes for common query patterns
  - Partial indexes for filtered queries (active runs, unresolved interruptions)
  - GIN indexes for full-text search on JSONB columns
  - JSONB path indexes for querying nested fields

### Flexibility
- JSONB columns for extensible data (item_data, metadata, state_data)
- Support for both static and dynamic instructions/prompts
- Soft delete support via `deleted_at` columns

### Scalability
- Designed for partitioning (runs, run_items, session_items by date)
- Efficient query patterns with composite indexes
- Support for high-volume operations

## Key Relationships

- **agents** → **model_settings** (many-to-one, SET NULL on delete)
- **agents** → **tools** (many-to-many via agent_tools, CASCADE)
- **agents** → **agents** (many-to-many via agent_handoffs, CASCADE)
- **runs** → **agents** (many-to-one, RESTRICT on delete)
- **runs** → **sessions** (many-to-one, optional, SET NULL on delete)
- **runs** → **run_items**, **tool_calls**, **interruptions** (one-to-many, CASCADE)

## Usage Examples

### Query Active Runs for an Agent

```sql
SELECT r.*, a.name as agent_name
FROM runs r
JOIN agents a ON r.agent_id = a.id
WHERE r.agent_id = 'agent-uuid-here'
  AND r.status IN ('running', 'interrupted')
ORDER BY r.started_at DESC;
```

### Get Run State for Resuming

```sql
SELECT state_data
FROM run_states
WHERE run_id = 'run-uuid-here'
  AND is_active = true;
```

### Query Usage Metrics

```sql
SELECT 
  a.name as agent_name,
  SUM(um.input_tokens) as total_input_tokens,
  SUM(um.output_tokens) as total_output_tokens,
  SUM(um.estimated_cost) as total_cost
FROM usage_metrics um
JOIN runs r ON um.run_id = r.id
JOIN agents a ON r.agent_id = a.id
WHERE r.started_at >= NOW() - INTERVAL '7 days'
GROUP BY a.id, a.name;
```

## Future Enhancements

- Partitioning strategies for large-scale deployments
- Materialized views for common aggregations
- Data retention/archiving policies
- Multi-user support (created_by, updated_by columns)
- Full agent versioning history

## See Also

- [ERD Plan](../ERD_PLAN.md) - Detailed ERD documentation
- [OpenAI Agents Python](https://github.com/openai/openai-agents-python)
- [OpenAI Agents JS](https://github.com/openai/openai-agents-js)

