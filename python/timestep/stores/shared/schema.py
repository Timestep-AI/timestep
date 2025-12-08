"""Schema initialization for Timestep database."""

async def initialize_schema(db) -> None:
    """
    Initialize the database schema for all tables.
    This creates the necessary tables and indexes if they don't exist.
    
    Args:
        db: DatabaseConnection instance with unified interface (execute, fetchval, fetchrow)
    """
    
    # Create run_state_type_enum if it doesn't exist
    await db.execute("""
        DO $$ BEGIN
            CREATE TYPE run_state_type_enum AS ENUM ('interrupted', 'checkpoint', 'final');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    
    # Create run_states table
    # Use TEXT for run_id to support both UUID and session/conversation IDs
    await db.execute("""
        CREATE TABLE IF NOT EXISTS run_states (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id TEXT NOT NULL,
            state_type run_state_type_enum NOT NULL,
            schema_version VARCHAR(20) NOT NULL,
            state_data JSONB NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT true,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            resumed_at TIMESTAMP,
            CONSTRAINT chk_schema_version CHECK (schema_version ~ '^[0-9]+\\.[0-9]+$')
        )
    """)
    
    # Create indexes (one per query)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_run_states_run_id ON run_states(run_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_run_states_type ON run_states(state_type)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_run_states_created_at ON run_states(created_at)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_run_states_gin ON run_states USING GIN (state_data)")
    
    # Create unique partial index for active state
    await db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_run_states_active_unique 
        ON run_states(run_id) 
        WHERE is_active = true
    """)
    
    # Create agents table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL UNIQUE,
            instructions TEXT,
            instructions_fn TEXT,
            prompt JSONB,
            prompt_fn TEXT,
            handoff_description TEXT,
            handoff_output_type_warning_enabled BOOLEAN,
            model TEXT,
            model_settings JSONB,
            output_type JSONB,
            tool_use_behavior JSONB,
            reset_tool_choice BOOLEAN DEFAULT true,
            mcp_config JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Create tools table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS tools (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            parameters JSONB,
            strict BOOLEAN,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Create agent_tools junction table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS agent_tools (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            tool_id UUID NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
            order_index INTEGER NOT NULL,
            UNIQUE(agent_id, tool_id, order_index)
        )
    """)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_tools_agent_id ON agent_tools(agent_id)")
    
    # Create guardrails table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS guardrails (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            type TEXT NOT NULL CHECK (type IN ('input', 'output')),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Create agent_guardrails junction table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS agent_guardrails (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            guardrail_id UUID NOT NULL REFERENCES guardrails(id) ON DELETE CASCADE,
            type TEXT NOT NULL CHECK (type IN ('input', 'output')),
            order_index INTEGER NOT NULL,
            UNIQUE(agent_id, guardrail_id, type, order_index)
        )
    """)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_guardrails_agent_id ON agent_guardrails(agent_id)")
    
    # Create agent_handoffs table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS agent_handoffs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            handoff_agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
            tool_name TEXT NOT NULL,
            tool_description TEXT,
            input_json_schema JSONB DEFAULT '{}',
            strict_json_schema BOOLEAN DEFAULT true,
            order_index INTEGER NOT NULL,
            metadata JSONB,
            UNIQUE(agent_id, order_index)
        )
    """)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_handoffs_agent_id ON agent_handoffs(agent_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_handoffs_handoff_agent_id ON agent_handoffs(handoff_agent_id)")
    
    # Create mcp_servers table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS mcp_servers (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            server_type TEXT NOT NULL,
            config_data JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Create agent_mcp_servers junction table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS agent_mcp_servers (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            mcp_server_id UUID NOT NULL REFERENCES mcp_servers(id) ON DELETE CASCADE,
            order_index INTEGER NOT NULL,
            UNIQUE(agent_id, mcp_server_id, order_index)
        )
    """)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_mcp_servers_agent_id ON agent_mcp_servers(agent_id)")
    
    # Create sessions table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id TEXT NOT NULL UNIQUE,
            session_type TEXT NOT NULL,
            config_data JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id)")

