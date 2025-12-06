-- Database Schema for OpenAI Agents Libraries
-- Supports both Python (openai-agents-python) and JavaScript (openai-agents-js) libraries
-- PostgreSQL 12+ required (for JSONB, UUID, and other features)

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- ENUMS
-- ============================================================================

CREATE TYPE agent_type_enum AS ENUM ('standard', 'realtime');
CREATE TYPE instructions_type_enum AS ENUM ('static', 'dynamic');
CREATE TYPE prompt_type_enum AS ENUM ('instructions', 'prompt', 'both');
CREATE TYPE run_status_enum AS ENUM ('running', 'completed', 'failed', 'interrupted', 'cancelled');
CREATE TYPE guardrail_type_enum AS ENUM ('input', 'output', 'tool_input', 'tool_output');
CREATE TYPE tool_guardrail_type_enum AS ENUM ('tool_input', 'tool_output');
CREATE TYPE tool_call_status_enum AS ENUM ('pending', 'approved', 'rejected', 'executed', 'failed');
CREATE TYPE interruption_type_enum AS ENUM ('tool_approval', 'mcp_approval', 'guardrail');
CREATE TYPE run_state_type_enum AS ENUM ('interrupted', 'checkpoint', 'final');

-- ============================================================================
-- TABLE: model_settings
-- ============================================================================

CREATE TABLE model_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    temperature DECIMAL(5,2),
    top_p DECIMAL(5,2),
    frequency_penalty DECIMAL(5,2),
    presence_penalty DECIMAL(5,2),
    max_tokens INTEGER,
    tool_choice VARCHAR(100),
    parallel_tool_calls BOOLEAN,
    truncation VARCHAR(20),
    reasoning_effort VARCHAR(50),
    seed INTEGER,
    response_format JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_temperature CHECK (temperature IS NULL OR (temperature >= 0 AND temperature <= 2)),
    CONSTRAINT chk_top_p CHECK (top_p IS NULL OR (top_p >= 0 AND top_p <= 1)),
    CONSTRAINT chk_frequency_penalty CHECK (frequency_penalty IS NULL OR (frequency_penalty >= -2 AND frequency_penalty <= 2)),
    CONSTRAINT chk_presence_penalty CHECK (presence_penalty IS NULL OR (presence_penalty >= -2 AND presence_penalty <= 2)),
    CONSTRAINT chk_max_tokens CHECK (max_tokens IS NULL OR max_tokens > 0)
);

-- ============================================================================
-- TABLE: agents
-- ============================================================================

CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    agent_type agent_type_enum NOT NULL DEFAULT 'standard',
    instructions TEXT,
    instructions_type instructions_type_enum,
    prompt_id VARCHAR(255),
    prompt_version VARCHAR(50),
    prompt_variables JSONB,
    prompt_type prompt_type_enum,
    handoff_description TEXT,
    model VARCHAR(255),
    model_settings_id UUID,
    output_type VARCHAR(100),
    output_schema JSONB,
    tool_use_behavior VARCHAR(50),
    reset_tool_choice BOOLEAN NOT NULL DEFAULT true,
    mcp_config JSONB,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP,
    metadata JSONB,
    
    CONSTRAINT fk_agents_model_settings FOREIGN KEY (model_settings_id) 
        REFERENCES model_settings(id) ON DELETE SET NULL,
    CONSTRAINT chk_prompt_type CHECK (prompt_type IS NULL OR prompt_type IN ('instructions', 'prompt', 'both'))
);

-- Indexes for agents
CREATE INDEX idx_agents_name ON agents(name);
CREATE INDEX idx_agents_model ON agents(model);
CREATE INDEX idx_agents_type ON agents(agent_type);
CREATE INDEX idx_agents_deleted ON agents(deleted_at) WHERE deleted_at IS NULL;

-- Unique constraint for active agent names
CREATE UNIQUE INDEX idx_agents_name_unique_active ON agents(name) 
    WHERE deleted_at IS NULL;

-- ============================================================================
-- TABLE: sessions
-- ============================================================================

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB,
    
    CONSTRAINT uq_sessions_session_id UNIQUE (session_id)
);

CREATE INDEX idx_sessions_session_id ON sessions(session_id);

-- ============================================================================
-- TABLE: runs
-- ============================================================================

CREATE TABLE runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL,
    session_id UUID,
    input TEXT NOT NULL,
    input_type VARCHAR(20) NOT NULL DEFAULT 'text',
    final_output TEXT,
    status run_status_enum NOT NULL DEFAULT 'running',
    current_turn INTEGER NOT NULL DEFAULT 0,
    max_turns INTEGER NOT NULL DEFAULT 10,
    previous_response_id VARCHAR(255),
    conversation_id VARCHAR(255),
    trace_id VARCHAR(255),
    group_id VARCHAR(255),
    workflow_name VARCHAR(255),
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_message TEXT,
    error_details JSONB,
    metadata JSONB,
    
    CONSTRAINT fk_runs_agent FOREIGN KEY (agent_id) 
        REFERENCES agents(id) ON DELETE RESTRICT,
    CONSTRAINT fk_runs_session FOREIGN KEY (session_id) 
        REFERENCES sessions(id) ON DELETE SET NULL,
    CONSTRAINT chk_current_turn CHECK (current_turn >= 0),
    CONSTRAINT chk_max_turns CHECK (max_turns > 0),
    CONSTRAINT chk_turn_limit CHECK (current_turn <= max_turns)
);

-- Indexes for runs
CREATE INDEX idx_runs_agent_id ON runs(agent_id);
CREATE INDEX idx_runs_session_id ON runs(session_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_trace_id ON runs(trace_id);
CREATE INDEX idx_runs_conversation_id ON runs(conversation_id);
CREATE INDEX idx_runs_started_at ON runs(started_at);
CREATE INDEX idx_runs_agent_status ON runs(agent_id, status);
CREATE INDEX idx_runs_agent_created ON runs(agent_id, started_at DESC);
CREATE INDEX idx_runs_active ON runs(status, started_at) 
    WHERE status IN ('running', 'interrupted');

-- ============================================================================
-- TABLE: session_items
-- ============================================================================

CREATE TABLE session_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    item_data JSONB NOT NULL,
    item_type VARCHAR(50),
    sequence_number INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_session_items_session FOREIGN KEY (session_id) 
        REFERENCES sessions(id) ON DELETE CASCADE,
    CONSTRAINT chk_sequence_number CHECK (sequence_number > 0)
);

CREATE INDEX idx_session_items_session_id ON session_items(session_id);
CREATE INDEX idx_session_items_sequence ON session_items(session_id, sequence_number);
CREATE INDEX idx_session_items_gin ON session_items USING GIN (item_data);

-- ============================================================================
-- TABLE: run_items
-- ============================================================================

CREATE TABLE run_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    item_type VARCHAR(50) NOT NULL,
    item_data JSONB NOT NULL,
    sequence_number INTEGER NOT NULL,
    turn_number INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_run_items_run FOREIGN KEY (run_id) 
        REFERENCES runs(id) ON DELETE CASCADE,
    CONSTRAINT fk_run_items_agent FOREIGN KEY (agent_id) 
        REFERENCES agents(id) ON DELETE RESTRICT,
    CONSTRAINT chk_run_items_sequence CHECK (sequence_number > 0),
    CONSTRAINT chk_run_items_turn CHECK (turn_number IS NULL OR turn_number >= 0)
);

CREATE INDEX idx_run_items_run_id ON run_items(run_id);
CREATE INDEX idx_run_items_agent_id ON run_items(agent_id);
CREATE INDEX idx_run_items_sequence ON run_items(run_id, sequence_number);
CREATE INDEX idx_run_items_turn ON run_items(run_id, turn_number);
CREATE INDEX idx_run_items_gin ON run_items USING GIN (item_data);

-- ============================================================================
-- TABLE: model_responses
-- ============================================================================

CREATE TABLE model_responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    response_id VARCHAR(255),
    model VARCHAR(255),
    response_data JSONB NOT NULL,
    turn_number INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_model_responses_run FOREIGN KEY (run_id) 
        REFERENCES runs(id) ON DELETE CASCADE,
    CONSTRAINT fk_model_responses_agent FOREIGN KEY (agent_id) 
        REFERENCES agents(id) ON DELETE RESTRICT,
    CONSTRAINT uq_model_responses_response_id UNIQUE (response_id),
    CONSTRAINT chk_model_responses_turn CHECK (turn_number IS NULL OR turn_number >= 0)
);

CREATE INDEX idx_model_responses_run_id ON model_responses(run_id);
CREATE INDEX idx_model_responses_response_id ON model_responses(response_id);
CREATE INDEX idx_model_responses_agent_id ON model_responses(agent_id);
CREATE INDEX idx_model_responses_gin ON model_responses USING GIN (response_data);

-- ============================================================================
-- TABLE: tools
-- ============================================================================

CREATE TABLE tools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL,
    parameters_schema JSONB,
    strict_json_schema BOOLEAN NOT NULL DEFAULT true,
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    needs_approval BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB,
    
    CONSTRAINT uq_tools_name UNIQUE (name)
);

CREATE INDEX idx_tools_name ON tools(name);
CREATE INDEX idx_tools_type ON tools(type);

-- ============================================================================
-- TABLE: agent_tools
-- ============================================================================

CREATE TABLE agent_tools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL,
    tool_id UUID NOT NULL,
    "order" INTEGER,
    is_enabled_override BOOLEAN,
    needs_approval_override BOOLEAN,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_agent_tools_agent FOREIGN KEY (agent_id) 
        REFERENCES agents(id) ON DELETE CASCADE,
    CONSTRAINT fk_agent_tools_tool FOREIGN KEY (tool_id) 
        REFERENCES tools(id) ON DELETE CASCADE,
    CONSTRAINT uq_agent_tools UNIQUE (agent_id, tool_id)
);

CREATE INDEX idx_agent_tools_agent_id ON agent_tools(agent_id);
CREATE INDEX idx_agent_tools_tool_id ON agent_tools(tool_id);

-- ============================================================================
-- TABLE: handoff_configs
-- ============================================================================

CREATE TABLE handoff_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tool_name VARCHAR(255),
    tool_description TEXT,
    input_json_schema JSONB,
    strict_json_schema BOOLEAN NOT NULL DEFAULT true,
    input_filter_config JSONB,
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- TABLE: agent_handoffs
-- ============================================================================

CREATE TABLE agent_handoffs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_agent_id UUID NOT NULL,
    target_agent_id UUID NOT NULL,
    handoff_config_id UUID,
    "order" INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_agent_handoffs_source FOREIGN KEY (source_agent_id) 
        REFERENCES agents(id) ON DELETE CASCADE,
    CONSTRAINT fk_agent_handoffs_target FOREIGN KEY (target_agent_id) 
        REFERENCES agents(id) ON DELETE CASCADE,
    CONSTRAINT fk_agent_handoffs_config FOREIGN KEY (handoff_config_id) 
        REFERENCES handoff_configs(id) ON DELETE SET NULL,
    CONSTRAINT uq_agent_handoffs UNIQUE (source_agent_id, target_agent_id),
    CONSTRAINT chk_no_self_handoff CHECK (source_agent_id != target_agent_id)
);

CREATE INDEX idx_agent_handoffs_source ON agent_handoffs(source_agent_id);
CREATE INDEX idx_agent_handoffs_target ON agent_handoffs(target_agent_id);

-- ============================================================================
-- TABLE: guardrails
-- ============================================================================

CREATE TABLE guardrails (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type guardrail_type_enum NOT NULL,
    guardrail_function_ref VARCHAR(255),
    run_in_parallel BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB,
    
    CONSTRAINT uq_guardrails_name_type UNIQUE (name, type)
);

CREATE INDEX idx_guardrails_name ON guardrails(name);
CREATE INDEX idx_guardrails_type ON guardrails(type);

-- ============================================================================
-- TABLE: agent_guardrails
-- ============================================================================

CREATE TABLE agent_guardrails (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL,
    guardrail_id UUID NOT NULL,
    "order" INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_agent_guardrails_agent FOREIGN KEY (agent_id) 
        REFERENCES agents(id) ON DELETE CASCADE,
    CONSTRAINT fk_agent_guardrails_guardrail FOREIGN KEY (guardrail_id) 
        REFERENCES guardrails(id) ON DELETE CASCADE,
    CONSTRAINT uq_agent_guardrails UNIQUE (agent_id, guardrail_id)
);

CREATE INDEX idx_agent_guardrails_agent_id ON agent_guardrails(agent_id);
CREATE INDEX idx_agent_guardrails_guardrail_id ON agent_guardrails(guardrail_id);

-- ============================================================================
-- TABLE: guardrail_results
-- ============================================================================

CREATE TABLE guardrail_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    guardrail_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    tripwire_triggered BOOLEAN NOT NULL,
    output_info JSONB,
    checked_value TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_guardrail_results_run FOREIGN KEY (run_id) 
        REFERENCES runs(id) ON DELETE CASCADE,
    CONSTRAINT fk_guardrail_results_guardrail FOREIGN KEY (guardrail_id) 
        REFERENCES guardrails(id) ON DELETE RESTRICT,
    CONSTRAINT fk_guardrail_results_agent FOREIGN KEY (agent_id) 
        REFERENCES agents(id) ON DELETE RESTRICT
);

CREATE INDEX idx_guardrail_results_run_id ON guardrail_results(run_id);
CREATE INDEX idx_guardrail_results_guardrail_id ON guardrail_results(guardrail_id);
CREATE INDEX idx_guardrail_results_tripwire ON guardrail_results(tripwire_triggered);
CREATE INDEX idx_guardrail_results_agent ON guardrail_results(agent_id, tripwire_triggered);

-- ============================================================================
-- TABLE: tool_guardrails
-- ============================================================================

CREATE TABLE tool_guardrails (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tool_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    type tool_guardrail_type_enum NOT NULL,
    guardrail_function_ref VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB,
    
    CONSTRAINT fk_tool_guardrails_tool FOREIGN KEY (tool_id) 
        REFERENCES tools(id) ON DELETE CASCADE,
    CONSTRAINT uq_tool_guardrails UNIQUE (tool_id, name, type)
);

CREATE INDEX idx_tool_guardrails_tool_id ON tool_guardrails(tool_id);
CREATE INDEX idx_tool_guardrails_type ON tool_guardrails(type);

-- ============================================================================
-- TABLE: tool_calls
-- ============================================================================

CREATE TABLE tool_calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    run_item_id UUID NOT NULL,
    tool_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    call_id VARCHAR(255),
    parameters JSONB NOT NULL,
    output TEXT,
    output_type VARCHAR(50),
    status tool_call_status_enum NOT NULL DEFAULT 'pending',
    needs_approval BOOLEAN NOT NULL DEFAULT false,
    approved_at TIMESTAMP,
    executed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_tool_calls_run FOREIGN KEY (run_id) 
        REFERENCES runs(id) ON DELETE CASCADE,
    CONSTRAINT fk_tool_calls_run_item FOREIGN KEY (run_item_id) 
        REFERENCES run_items(id) ON DELETE CASCADE,
    CONSTRAINT fk_tool_calls_tool FOREIGN KEY (tool_id) 
        REFERENCES tools(id) ON DELETE RESTRICT,
    CONSTRAINT fk_tool_calls_agent FOREIGN KEY (agent_id) 
        REFERENCES agents(id) ON DELETE RESTRICT
);

CREATE INDEX idx_tool_calls_run_id ON tool_calls(run_id);
CREATE INDEX idx_tool_calls_tool_id ON tool_calls(tool_id);
CREATE INDEX idx_tool_calls_call_id ON tool_calls(call_id);
CREATE INDEX idx_tool_calls_status ON tool_calls(status);
CREATE INDEX idx_tool_calls_pending ON tool_calls(status, created_at) 
    WHERE status = 'pending';
CREATE INDEX idx_tool_calls_gin ON tool_calls USING GIN (parameters);

-- Unique constraint for call_id when present
CREATE UNIQUE INDEX idx_tool_calls_call_id_unique ON tool_calls(call_id) 
    WHERE call_id IS NOT NULL;

-- ============================================================================
-- TABLE: tool_guardrail_results
-- ============================================================================

CREATE TABLE tool_guardrail_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tool_call_id UUID NOT NULL,
    tool_guardrail_id UUID NOT NULL,
    tripwire_triggered BOOLEAN NOT NULL,
    output_info JSONB,
    checked_value TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_tool_guardrail_results_tool_call FOREIGN KEY (tool_call_id) 
        REFERENCES tool_calls(id) ON DELETE CASCADE,
    CONSTRAINT fk_tool_guardrail_results_guardrail FOREIGN KEY (tool_guardrail_id) 
        REFERENCES tool_guardrails(id) ON DELETE RESTRICT
);

CREATE INDEX idx_tool_guardrail_results_tool_call ON tool_guardrail_results(tool_call_id);
CREATE INDEX idx_tool_guardrail_results_guardrail ON tool_guardrail_results(tool_guardrail_id);
CREATE INDEX idx_tool_guardrail_results_tripwire ON tool_guardrail_results(tripwire_triggered);

-- ============================================================================
-- TABLE: interruptions
-- ============================================================================

CREATE TABLE interruptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    run_item_id UUID NOT NULL,
    interruption_type interruption_type_enum NOT NULL,
    interruption_data JSONB,
    resolved BOOLEAN NOT NULL DEFAULT false,
    resolved_at TIMESTAMP,
    resolution_action VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_interruptions_run FOREIGN KEY (run_id) 
        REFERENCES runs(id) ON DELETE CASCADE,
    CONSTRAINT fk_interruptions_run_item FOREIGN KEY (run_item_id) 
        REFERENCES run_items(id) ON DELETE CASCADE,
    CONSTRAINT chk_resolution_action CHECK (
        resolution_action IS NULL OR 
        resolution_action IN ('approved', 'rejected', 'cancelled')
    )
);

CREATE INDEX idx_interruptions_run_id ON interruptions(run_id);
CREATE INDEX idx_interruptions_resolved ON interruptions(resolved);
CREATE INDEX idx_interruptions_unresolved ON interruptions(run_id, resolved) 
    WHERE resolved = false;

-- ============================================================================
-- TABLE: mcp_servers
-- ============================================================================

CREATE TABLE mcp_servers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    server_type VARCHAR(50) NOT NULL,
    connection_config JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT uq_mcp_servers_name UNIQUE (name),
    CONSTRAINT chk_mcp_server_type CHECK (server_type IN ('stdio', 'sse', 'http'))
);

CREATE INDEX idx_mcp_servers_name ON mcp_servers(name);

-- ============================================================================
-- TABLE: agent_mcp_servers
-- ============================================================================

CREATE TABLE agent_mcp_servers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL,
    mcp_server_id UUID NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_agent_mcp_servers_agent FOREIGN KEY (agent_id) 
        REFERENCES agents(id) ON DELETE CASCADE,
    CONSTRAINT fk_agent_mcp_servers_server FOREIGN KEY (mcp_server_id) 
        REFERENCES mcp_servers(id) ON DELETE CASCADE,
    CONSTRAINT uq_agent_mcp_servers UNIQUE (agent_id, mcp_server_id)
);

-- ============================================================================
-- TABLE: usage_metrics
-- ============================================================================

CREATE TABLE usage_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    model_response_id UUID,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost DECIMAL(15,4),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_usage_metrics_run FOREIGN KEY (run_id) 
        REFERENCES runs(id) ON DELETE CASCADE,
    CONSTRAINT fk_usage_metrics_model_response FOREIGN KEY (model_response_id) 
        REFERENCES model_responses(id) ON DELETE SET NULL,
    CONSTRAINT chk_usage_input_tokens CHECK (input_tokens >= 0),
    CONSTRAINT chk_usage_output_tokens CHECK (output_tokens >= 0),
    CONSTRAINT chk_usage_total_tokens CHECK (total_tokens >= 0),
    CONSTRAINT chk_usage_cached_tokens CHECK (cached_tokens >= 0),
    CONSTRAINT chk_usage_reasoning_tokens CHECK (reasoning_tokens >= 0),
    CONSTRAINT chk_usage_cost CHECK (estimated_cost IS NULL OR estimated_cost >= 0),
    CONSTRAINT chk_usage_consistency CHECK (total_tokens = input_tokens + output_tokens)
);

CREATE INDEX idx_usage_metrics_run_id ON usage_metrics(run_id);
CREATE INDEX idx_usage_metrics_model_response_id ON usage_metrics(model_response_id);
CREATE INDEX idx_usage_metrics_cost ON usage_metrics(estimated_cost);

-- ============================================================================
-- TABLE: request_usage_entries
-- ============================================================================

CREATE TABLE request_usage_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    model_response_id UUID,
    request_number INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_request_usage_run FOREIGN KEY (run_id) 
        REFERENCES runs(id) ON DELETE CASCADE,
    CONSTRAINT fk_request_usage_model_response FOREIGN KEY (model_response_id) 
        REFERENCES model_responses(id) ON DELETE SET NULL,
    CONSTRAINT chk_request_input_tokens CHECK (input_tokens >= 0),
    CONSTRAINT chk_request_output_tokens CHECK (output_tokens >= 0),
    CONSTRAINT chk_request_total_tokens CHECK (total_tokens >= 0),
    CONSTRAINT chk_request_number CHECK (request_number > 0),
    CONSTRAINT chk_request_consistency CHECK (total_tokens = input_tokens + output_tokens)
);

CREATE INDEX idx_request_usage_run_id ON request_usage_entries(run_id);
CREATE INDEX idx_request_usage_response_id ON request_usage_entries(model_response_id);
CREATE INDEX idx_request_usage_sequence ON request_usage_entries(run_id, request_number);

-- ============================================================================
-- TABLE: run_states
-- ============================================================================

CREATE TABLE run_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    state_type run_state_type_enum NOT NULL,
    schema_version VARCHAR(20) NOT NULL,
    state_data JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    resumed_at TIMESTAMP,
    
    CONSTRAINT fk_run_states_run FOREIGN KEY (run_id) 
        REFERENCES runs(id) ON DELETE CASCADE,
    CONSTRAINT chk_schema_version CHECK (schema_version ~ '^[0-9]+\.[0-9]+$')
);

CREATE INDEX idx_run_states_run_id ON run_states(run_id);
CREATE INDEX idx_run_states_type ON run_states(state_type);
CREATE INDEX idx_run_states_created_at ON run_states(created_at);
CREATE INDEX idx_run_states_current_agent ON run_states((state_data->'currentAgent'->>'name'));
CREATE INDEX idx_run_states_current_turn ON run_states((state_data->>'currentTurn'));
CREATE INDEX idx_run_states_gin ON run_states USING GIN (state_data);

-- Unique partial index for active state (only one active state per run)
CREATE UNIQUE INDEX idx_run_states_active_unique ON run_states(run_id) 
    WHERE is_active = true;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE agents IS 'Stores agent definitions and configurations';
COMMENT ON TABLE model_settings IS 'Stores model-specific configuration parameters';
COMMENT ON TABLE runs IS 'Stores agent execution instances';
COMMENT ON TABLE sessions IS 'Stores conversation history sessions';
COMMENT ON TABLE session_items IS 'Stores individual items in a session conversation history';
COMMENT ON TABLE run_items IS 'Stores individual items generated during a run (messages, tool calls, etc.)';
COMMENT ON TABLE model_responses IS 'Stores raw LLM responses';
COMMENT ON TABLE tools IS 'Stores tool definitions';
COMMENT ON TABLE agent_tools IS 'Junction table for agent-tool relationships';
COMMENT ON TABLE handoff_configs IS 'Stores handoff-specific configurations';
COMMENT ON TABLE agent_handoffs IS 'Junction table for agent handoff relationships';
COMMENT ON TABLE guardrails IS 'Stores guardrail definitions';
COMMENT ON TABLE agent_guardrails IS 'Junction table for agent-guardrail relationships';
COMMENT ON TABLE guardrail_results IS 'Stores guardrail execution results';
COMMENT ON TABLE tool_guardrails IS 'Stores tool-specific guardrail definitions';
COMMENT ON TABLE tool_guardrail_results IS 'Stores tool guardrail execution results';
COMMENT ON TABLE tool_calls IS 'Stores individual tool call executions';
COMMENT ON TABLE interruptions IS 'Stores run interruptions (e.g., tool approval requests)';
COMMENT ON TABLE mcp_servers IS 'Stores MCP (Model Context Protocol) server configurations';
COMMENT ON TABLE agent_mcp_servers IS 'Junction table for agent-MCP server relationships';
COMMENT ON TABLE usage_metrics IS 'Stores aggregated token usage and cost metrics per run';
COMMENT ON TABLE request_usage_entries IS 'Stores per-request token usage breakdown';
COMMENT ON TABLE run_states IS 'Stores serialized RunState snapshots for resuming interrupted runs';

