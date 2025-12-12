-- Add openai_messages column to tasks table
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS openai_messages JSONB;

-- Create index on openai_messages for potential queries
CREATE INDEX IF NOT EXISTS idx_tasks_openai_messages ON tasks USING GIN (openai_messages);

