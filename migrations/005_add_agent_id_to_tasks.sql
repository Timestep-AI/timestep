-- Add agent_id column to tasks table
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS agent_id TEXT REFERENCES agents(id) ON DELETE CASCADE;

-- Create index on agent_id for efficient queries
CREATE INDEX IF NOT EXISTS idx_tasks_agent_id ON tasks(agent_id);

