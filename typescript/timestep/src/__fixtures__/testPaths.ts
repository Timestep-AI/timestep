/**
 * Test fixtures for consistent test data
 */

import {fileURLToPath} from 'node:url';
import {dirname, join} from 'node:path';

// Get the directory of this fixtures file
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Point to the actual test fixtures directory
export const TEST_TIMESTEP_PATHS = {
	configDir: join(__dirname, 'test-config'),
	appConfig: join(__dirname, 'test-config', 'app.json'),
	agentsConfig: join(__dirname, 'test-config', 'agents.jsonl'),
	modelProviders: join(__dirname, 'test-config', 'modelProviders.jsonl'),
	mcpServers: join(__dirname, 'test-config', 'mcpServers.jsonl'),
	contexts: join(__dirname, 'test-config', 'contexts.jsonl'),
};

export const TEST_APP_CONFIG = {
	appPort: 8080,
};

export const TEST_VERSION_INFO = {
	version: '1.0.0',
	name: '@timestep-ai/timestep',
	description: 'Test description',
	timestamp: new Date().toISOString(),
};

export const TEST_USER_ID = 'test-user';

export const TEST_MCP_TOOLS = [];

// Test data fixtures
export const TEST_AGENTS = [
	{
		id: 'test-agent-1',
		name: 'Test Agent 1',
		instructions: 'A test agent for unit testing',
		tools: [],
		handoffs: [],
		modelProviderId: 'test-model-provider',
		createdAt: '2024-01-01T00:00:00.000Z',
		updatedAt: '2024-01-01T00:00:00.000Z',
	},
	{
		id: 'test-agent-2',
		name: 'Test Agent 2',
		instructions: 'Another test agent',
		tools: [],
		handoffs: [],
		modelProviderId: 'test-model-provider',
		createdAt: '2024-01-01T00:00:00.000Z',
		updatedAt: '2024-01-01T00:00:00.000Z',
	},
];

export const TEST_MODEL_PROVIDERS = [
	{
		id: 'test-model-provider',
		provider: 'openai',
		baseUrl: 'https://api.openai.com/v1',
		modelsUrl: 'https://api.openai.com/v1/models',
		apiKey: 'test-api-key',
	},
	{
		id: 'test-ollama-provider',
		provider: 'ollama',
		baseUrl: 'http://localhost:11434',
		modelsUrl: 'http://localhost:11434/api/tags',
		apiKey: null,
	},
];

export const TEST_MCP_SERVERS = [
	{
		id: 'test-mcp-server-1',
		name: 'Test MCP Server 1',
		description: 'A test MCP server',
		serverUrl: 'http://localhost:3001',
		enabled: true,
		authToken: null,
	},
	{
		id: 'test-mcp-server-2',
		name: 'Test MCP Server 2',
		description: 'Another test MCP server',
		serverUrl: 'http://localhost:3002',
		enabled: false,
		authToken: 'test-token',
	},
];

export const TEST_CONTEXTS = [
	{
		id: 'test-context-1',
		taskHistories: {
			'task-1': [
				{
					role: 'user',
					content: 'Hello, how are you?',
					timestamp: '2024-01-01T00:00:00.000Z',
				},
			],
		},
		taskStates: {
			'task-1': 'completed',
		},
		tasks: {
			'task-1': {
				id: 'task-1',
				status: 'completed',
				createdAt: '2024-01-01T00:00:00.000Z',
				updatedAt: '2024-01-01T00:00:00.000Z',
			},
		},
	},
	{
		id: 'test-context-2',
		taskHistories: {},
		taskStates: {},
		tasks: {},
	},
];
