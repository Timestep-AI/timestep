-- Add parent_context_id column to contexts table
ALTER TABLE contexts ADD COLUMN IF NOT EXISTS parent_context_id UUID REFERENCES contexts(id) ON DELETE SET NULL;

-- Create index on parent_context_id for efficient child queries
CREATE INDEX IF NOT EXISTS idx_contexts_parent_id ON contexts(parent_context_id);


