-- Seed default agents
-- Personal Assistant
INSERT INTO agents (id, name, description, tools, model)
VALUES (
    'personal-assistant',
    'Personal Assistant',
    'Personal assistant with web search and agent handoff capabilities',
    '["handoff", "web_search"]'::jsonb,
    'gpt-4.1'
)
ON CONFLICT (id) DO NOTHING;

-- Weather Assistant
INSERT INTO agents (id, name, description, tools, model)
VALUES (
    'weather-assistant',
    'Weather Assistant',
    'Specialized weather information agent',
    '["get_weather"]'::jsonb,
    'gpt-4.1'
)
ON CONFLICT (id) DO NOTHING;

