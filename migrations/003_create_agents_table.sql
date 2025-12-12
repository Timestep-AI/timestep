-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    tools JSONB NOT NULL DEFAULT '[]'::jsonb,
    model TEXT NOT NULL DEFAULT 'gpt-4.1',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on created_at for sorting
CREATE INDEX IF NOT EXISTS idx_agents_created_at ON agents(created_at);

-- Trigger to auto-update updated_at on agents
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

