/** Schema initialization for Timestep database. */

export async function initializeSchema(db: any): Promise<void> {
  /**
   * Initialize the database schema for run_states table.
   * This creates the necessary tables and indexes if they don't exist.
   */
  
  // Check if run_states table exists
  const tableCheck = await db.query(`
    SELECT EXISTS (
      SELECT FROM information_schema.tables 
      WHERE table_schema = 'public' 
      AND table_name = 'run_states'
    );
  `);
  
  if (tableCheck.rows && tableCheck.rows[0]?.exists) {
    // Table already exists
    return;
  }
  
  // Create run_state_type_enum if it doesn't exist
  try {
    await db.query(`
      CREATE TYPE run_state_type_enum AS ENUM ('interrupted', 'checkpoint', 'final');
    `);
  } catch (e: any) {
    // Ignore if type already exists
    if (!e.message?.includes('already exists') && !e.message?.includes('duplicate')) {
      throw e;
    }
  }
  
  // Create run_states table
  // Use TEXT for run_id to support both UUID and session/conversation IDs
  await db.query(`
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
    );
  `);
  
  // Create indexes (one per query)
  await db.query(`CREATE INDEX IF NOT EXISTS idx_run_states_run_id ON run_states(run_id)`);
  await db.query(`CREATE INDEX IF NOT EXISTS idx_run_states_type ON run_states(state_type)`);
  await db.query(`CREATE INDEX IF NOT EXISTS idx_run_states_created_at ON run_states(created_at)`);
  await db.query(`CREATE INDEX IF NOT EXISTS idx_run_states_gin ON run_states USING GIN (state_data)`);
  
  // Create unique partial index for active state
  await db.query(`
    CREATE UNIQUE INDEX IF NOT EXISTS idx_run_states_active_unique 
    ON run_states(run_id) 
    WHERE is_active = true
  `);
}

