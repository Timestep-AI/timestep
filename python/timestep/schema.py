"""Schema initialization for Timestep database."""

async def initialize_schema(db) -> None:
    """
    Initialize the database schema for run_states table.
    This creates the necessary tables and indexes if they don't exist.
    """
    # Check if run_states table exists
    if hasattr(db, 'fetchval'):
        # asyncpg
        table_exists = await db.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'run_states'
            )
        """)
    elif hasattr(db, 'fetchrow'):
        # PGLite - use fetchrow method
        row = await db.fetchrow("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'run_states'
            ) as exists
        """)
        table_exists = row.get('exists', False) if row else False
    else:
        # Fallback: try query method
        result = await db.query("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'run_states'
            ) as exists
        """)
        rows = result.get('rows', []) if isinstance(result, dict) else []
        table_exists = rows[0].get('exists', False) if rows else False
    
    if table_exists:
        # Table already exists
        return
    
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

