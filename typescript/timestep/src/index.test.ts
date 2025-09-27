import {describe, it, expect, vi} from 'vitest';
import {
	TEST_TIMESTEP_PATHS,
	TEST_APP_CONFIG,
	TEST_VERSION_INFO,
	TEST_USER_ID,
	TEST_MCP_TOOLS,
} from './__fixtures__/testPaths.js';

// Mock utils to prevent MCP connections during tests
vi.mock('./utils.js', () => ({
	getTimestepPaths: vi.fn(() => TEST_TIMESTEP_PATHS),
	getVersion: vi.fn(() => Promise.resolve(TEST_VERSION_INFO)),
	encryptSecret: vi.fn(),
	decryptSecret: vi.fn(),
	isEncryptedSecret: vi.fn(),
	maskSecret: vi.fn(),
	getCurrentUserId: vi.fn(() => TEST_USER_ID),
	listAllMcpTools: vi.fn(() => Promise.resolve(TEST_MCP_TOOLS)),
	createMcpClient: vi.fn(),
	loadAppConfig: vi.fn(() => TEST_APP_CONFIG),
}));

describe('index.ts exports', () => {
	it('should export core functions', async () => {
		const utils = await import('./utils.js');
		const modelsApi = await import('./api/modelsApi.js');
		const contextsApi = await import('./api/contextsApi.js');
		const tracesApi = await import('./api/tracesApi.js');
		const toolsApi = await import('./api/toolsApi.js');

		expect(utils.getVersion).toBeDefined();
		expect(modelsApi.listModels).toBeDefined();
		expect(contextsApi.listContexts).toBeDefined();
		expect(tracesApi.listTraces).toBeDefined();
		expect(toolsApi.listTools).toBeDefined();
	});

	it('should export from index.ts', async () => {
		const indexModule = await import('./index.js');

		// Test that key exports are available
		expect(indexModule.getTimestepPaths).toBeDefined();
		expect(indexModule.getVersion).toBeDefined();
		expect(indexModule.listModels).toBeDefined();
		expect(indexModule.listContexts).toBeDefined();
		expect(indexModule.listAgents).toBeDefined();
		expect(indexModule.listMcpServers).toBeDefined();
		expect(indexModule.listModelProviders).toBeDefined();
		expect(indexModule.Context).toBeDefined();
		expect(indexModule.TimestepAIAgentExecutor).toBeDefined();
		expect(indexModule.ContextAwareRequestHandler).toBeDefined();
	});

	it('should export agent functions', async () => {
		const agentsApi = await import('./api/agentsApi.js');

		expect(agentsApi.listAgents).toBeDefined();
		expect(agentsApi.getAgent).toBeDefined();
		expect(agentsApi.isAgentAvailable).toBeDefined();
		expect(agentsApi.handleAgentRequest).toBeDefined();
		expect(agentsApi.handleListAgents).toBeDefined();
		expect(agentsApi.createAgentRequestHandler).toBeDefined();
	});

	it('should export MCP server functions', async () => {
		const mcpServersApi = await import('./api/mcpServersApi.js');

		expect(mcpServersApi.listMcpServers).toBeDefined();
		expect(mcpServersApi.getMcpServer).toBeDefined();
		expect(mcpServersApi.handleMcpServerRequest).toBeDefined();
	});

	it('should export model provider functions', async () => {
		const modelProvidersApi = await import('./api/modelProvidersApi.js');

		expect(modelProvidersApi.listModelProviders).toBeDefined();
		expect(modelProvidersApi.getModelProvider).toBeDefined();
	});

	it('should export core classes and types', async () => {
		const context = await import('./types/context.js');
		const repositoryContainer = await import(
			'./services/backing/repositoryContainer.js'
		);
		const agentExecutor = await import('./core/agentExecutor.js');
		const contextAwareRequestHandler = await import(
			'./api/contextAwareRequestHandler.js'
		);

		expect(context.Context).toBeDefined();
		expect(repositoryContainer.DefaultRepositoryContainer).toBeDefined();
		expect(agentExecutor.TimestepAIAgentExecutor).toBeDefined();
		expect(contextAwareRequestHandler.ContextAwareRequestHandler).toBeDefined();
	});

	it('should export configuration functions', async () => {
		const defaultMcpServers = await import('./config/defaultMcpServers.js');
		const defaultAgents = await import('./config/defaultAgents.js');
		const defaultModelProviders = await import(
			'./config/defaultModelProviders.js'
		);

		expect(defaultMcpServers.getDefaultMcpServers).toBeDefined();
		expect(defaultMcpServers.getBuiltinMcpServer).toBeDefined();
		expect(defaultAgents.getDefaultAgents).toBeDefined();
		expect(defaultModelProviders.getDefaultModelProviders).toBeDefined();
	});
});
