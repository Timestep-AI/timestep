import {describe, it, expect, vi, beforeEach} from 'vitest';

// Mock dependencies first
vi.mock('../services/backing/repositoryContainer.js', () => ({
	RepositoryContainer: vi.fn().mockImplementation(() => ({
		mcpServers: {
			list: vi.fn(),
			load: vi.fn(),
			save: vi.fn(),
			delete: vi.fn(),
		},
	})),
	DefaultRepositoryContainer: vi.fn().mockImplementation(() => ({
		mcpServers: {
			list: vi.fn(),
			load: vi.fn(),
			save: vi.fn(),
			delete: vi.fn(),
		},
	})),
}));

// Mock McpServerService
const mockMcpServerServiceInstance = {
	listMcpServers: vi.fn().mockResolvedValue([]),
	getMcpServer: vi.fn().mockResolvedValue(null),
	saveMcpServer: vi.fn().mockResolvedValue(undefined),
	deleteMcpServer: vi.fn().mockResolvedValue(undefined),
	handleMcpServerRequest: vi.fn().mockResolvedValue({}),
	callMcpTool: vi.fn().mockResolvedValue({}),
};

vi.mock('../services/mcpServerService.js', () => ({
	McpServerService: vi
		.fn()
		.mockImplementation(() => mockMcpServerServiceInstance),
}));

// Mock utils to prevent MCP connections
vi.mock('../utils.js', () => ({
	getTimestepPaths: vi.fn(() => ({
		agentsConfig: '/test/path/agents.jsonl',
		contextsConfig: '/test/path/contexts.jsonl',
		mcpServersConfig: '/test/path/mcpServers.jsonl',
		modelProvidersConfig: '/test/path/modelProviders.jsonl',
	})),
	listAllMcpTools: vi.fn(() => Promise.resolve([])),
	createMcpClient: vi.fn(),
}));

// Now import the module under test
import {
	listMcpServers,
	getMcpServer,
	saveMcpServer,
	deleteMcpServer,
	handleMcpServerRequest,
	callMcpTool,
	type McpServer,
} from './mcpServersApi.js';

describe('mcpServersApi', () => {
	beforeEach(() => {
		// Reset mock implementations to default values
		mockMcpServerServiceInstance.listMcpServers.mockResolvedValue([]);
		mockMcpServerServiceInstance.getMcpServer.mockResolvedValue(null);
		mockMcpServerServiceInstance.saveMcpServer.mockResolvedValue(undefined);
		mockMcpServerServiceInstance.deleteMcpServer.mockResolvedValue(undefined);
		mockMcpServerServiceInstance.handleMcpServerRequest.mockResolvedValue({});
		mockMcpServerServiceInstance.callMcpTool.mockResolvedValue({});
	});

	describe('listMcpServers', () => {
		it('should be a function', () => {
			expect(typeof listMcpServers).toBe('function');
		});

		it('should return a promise', () => {
			const result = listMcpServers();
			expect(result).toBeInstanceOf(Promise);
		});

		it('should execute listMcpServers with real data', async () => {
			const mockMcpServers: McpServer[] = [
				{
					id: 'server-1',
					name: 'Test Server 1',
					description: 'Test description 1',
					serverUrl: 'https://api.test1.com',
					enabled: true,
					authToken: 'token-1',
				},
				{
					id: 'server-2',
					name: 'Test Server 2',
					description: 'Test description 2',
					serverUrl: 'https://api.test2.com',
					enabled: false,
				},
			];

			mockMcpServerServiceInstance.listMcpServers.mockResolvedValue(
				mockMcpServers,
			);

			const result = await listMcpServers();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
			expect(result.data).toHaveLength(2);
			expect(result.data[0].id).toBe('server-1');
			expect(result.data[1].id).toBe('server-2');
		});

		it('should handle empty server list', async () => {
			mockMcpServerServiceInstance.listMcpServers.mockResolvedValue([]);

			const result = await listMcpServers();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(result.data).toHaveLength(0);
		});

		it('should handle service errors gracefully', async () => {
			mockMcpServerServiceInstance.listMcpServers.mockRejectedValue(
				new Error('Service error'),
			);

			await expect(listMcpServers()).rejects.toThrow(
				'Failed to list MCP servers: Error: Service error',
			);
		});

		it('should handle multiple concurrent calls', async () => {
			const mockMcpServers: McpServer[] = [
				{
					id: 'server-1',
					name: 'Test Server 1',
					description: 'Test description 1',
					serverUrl: 'https://api.test1.com',
					enabled: true,
				},
			];

			mockMcpServerServiceInstance.listMcpServers.mockResolvedValue(
				mockMcpServers,
			);

			const promises = [listMcpServers(), listMcpServers(), listMcpServers()];

			const results = await Promise.all(promises);

			expect(results).toHaveLength(3);
			results.forEach(result => {
				expect(result.object).toBe('list');
				expect(result.data).toHaveLength(1);
				expect(result.data[0].id).toBe('server-1');
			});
		});
	});

	describe('getMcpServer', () => {
		it('should be a function', () => {
			expect(typeof getMcpServer).toBe('function');
		});

		it('should return a promise', () => {
			const result = getMcpServer('test-id');
			expect(result).toBeInstanceOf(Promise);
		});

		it('should execute getMcpServer with valid ID', async () => {
			const mockServer: McpServer = {
				id: 'server-1',
				name: 'Test Server',
				description: 'Test description',
				serverUrl: 'https://api.test.com',
				enabled: true,
				authToken: 'test-token',
			};

			mockMcpServerServiceInstance.getMcpServer.mockResolvedValue(mockServer);

			const result = await getMcpServer('server-1');

			expect(result).toBeDefined();
			expect(result).toEqual(mockServer);
			expect(result?.id).toBe('server-1');
			expect(result?.name).toBe('Test Server');
		});

		it('should return null when server not found', async () => {
			mockMcpServerServiceInstance.getMcpServer.mockResolvedValue(null);

			const result = await getMcpServer('non-existent-server');

			expect(result).toBeNull();
		});

		it('should handle service errors gracefully', async () => {
			mockMcpServerServiceInstance.getMcpServer.mockRejectedValue(
				new Error('Service error'),
			);

			await expect(getMcpServer('server-1')).rejects.toThrow(
				'Failed to get MCP server: Error: Service error',
			);
		});

		it('should handle different server IDs', async () => {
			const mockServer: McpServer = {
				id: 'different-server',
				name: 'Different Server',
				description: 'Different description',
				serverUrl: 'https://api.different.com',
				enabled: false,
			};

			mockMcpServerServiceInstance.getMcpServer.mockResolvedValue(mockServer);

			const result = await getMcpServer('different-server');

			expect(result).toBeDefined();
			expect(result?.id).toBe('different-server');
			expect(result?.name).toBe('Different Server');
		});
	});

	describe('saveMcpServer', () => {
		it('should be a function', () => {
			expect(typeof saveMcpServer).toBe('function');
		});

		it('should execute saveMcpServer successfully', async () => {
			const mockServer: McpServer = {
				id: 'server-1',
				name: 'Test Server',
				description: 'Test description',
				serverUrl: 'https://api.test.com',
				enabled: true,
				authToken: 'test-token',
			};

			mockMcpServerServiceInstance.saveMcpServer.mockResolvedValue(undefined);

			await expect(saveMcpServer(mockServer)).resolves.not.toThrow();
		});

		it('should handle service errors gracefully', async () => {
			const mockServer: McpServer = {
				id: 'server-1',
				name: 'Test Server',
				description: 'Test description',
				serverUrl: 'https://api.test.com',
				enabled: true,
			};

			mockMcpServerServiceInstance.saveMcpServer.mockRejectedValue(
				new Error('Save error'),
			);

			await expect(saveMcpServer(mockServer)).rejects.toThrow(
				'Failed to save MCP server: Error: Save error',
			);
		});

		it('should handle multiple save operations', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'server-1',
					name: 'Server 1',
					description: 'Description 1',
					serverUrl: 'https://api.test1.com',
					enabled: true,
				},
				{
					id: 'server-2',
					name: 'Server 2',
					description: 'Description 2',
					serverUrl: 'https://api.test2.com',
					enabled: false,
				},
			];

			mockMcpServerServiceInstance.saveMcpServer.mockResolvedValue(undefined);

			const promises = mockServers.map(server => saveMcpServer(server));
			await expect(Promise.all(promises)).resolves.not.toThrow();
		});
	});

	describe('deleteMcpServer', () => {
		it('should be a function', () => {
			expect(typeof deleteMcpServer).toBe('function');
		});

		it('should execute deleteMcpServer successfully', async () => {
			mockMcpServerServiceInstance.deleteMcpServer.mockResolvedValue(undefined);

			await expect(deleteMcpServer('server-1')).resolves.not.toThrow();
		});

		it('should handle service errors gracefully', async () => {
			mockMcpServerServiceInstance.deleteMcpServer.mockRejectedValue(
				new Error('Delete error'),
			);

			await expect(deleteMcpServer('server-1')).rejects.toThrow(
				'Failed to delete MCP server: Error: Delete error',
			);
		});

		it('should handle different server IDs', async () => {
			mockMcpServerServiceInstance.deleteMcpServer.mockResolvedValue(undefined);

			await expect(deleteMcpServer('different-server')).resolves.not.toThrow();
		});
	});

	describe('handleMcpServerRequest', () => {
		it('should be a function', () => {
			expect(typeof handleMcpServerRequest).toBe('function');
		});

		it('should execute handleMcpServerRequest successfully', async () => {
			const mockRequest = {
				jsonrpc: '2.0',
				method: 'tools/list',
				params: {},
				id: 'test-request',
			};
			const mockResponse = {
				jsonrpc: '2.0',
				result: {tools: []},
				id: 'test-request',
			};

			mockMcpServerServiceInstance.handleMcpServerRequest.mockResolvedValue(
				mockResponse,
			);

			const result = await handleMcpServerRequest('server-1', mockRequest);

			expect(result).toBeDefined();
			expect(result).toEqual(mockResponse);
		});

		it('should handle service errors gracefully', async () => {
			const mockRequest = {
				jsonrpc: '2.0',
				method: 'tools/list',
				params: {},
				id: 'test-request',
			};

			mockMcpServerServiceInstance.handleMcpServerRequest.mockRejectedValue(
				new Error('Request error'),
			);

			await expect(
				handleMcpServerRequest('server-1', mockRequest),
			).rejects.toThrow(
				'Failed to handle MCP server request: Error: Request error',
			);
		});

		it('should handle different request types', async () => {
			const mockRequest = {
				jsonrpc: '2.0',
				method: 'tools/call',
				params: {name: 'test-tool', arguments: {}},
				id: 'test-call',
			};
			const mockResponse = {
				jsonrpc: '2.0',
				result: {success: true},
				id: 'test-call',
			};

			mockMcpServerServiceInstance.handleMcpServerRequest.mockResolvedValue(
				mockResponse,
			);

			const result = await handleMcpServerRequest('server-1', mockRequest);

			expect(result).toBeDefined();
			expect(result).toEqual(mockResponse);
		});
	});

	describe('callMcpTool', () => {
		it('should be a function', () => {
			expect(typeof callMcpTool).toBe('function');
		});

		it('should execute callMcpTool with basic parameters', async () => {
			const mockResponse = {
				jsonrpc: '2.0',
				result: {success: true},
				id: 'tools-call',
			};

			mockMcpServerServiceInstance.handleMcpServerRequest.mockResolvedValue(
				mockResponse,
			);

			const result = await callMcpTool('server-1', 'test-tool', {
				arg1: 'value1',
			});

			expect(result).toBeDefined();
			expect(result).toEqual(mockResponse);
		});

		it('should execute callMcpTool with custom ID', async () => {
			const mockResponse = {
				jsonrpc: '2.0',
				result: {success: true},
				id: 'custom-id',
			};

			mockMcpServerServiceInstance.handleMcpServerRequest.mockResolvedValue(
				mockResponse,
			);

			const result = await callMcpTool(
				'server-1',
				'test-tool',
				{arg1: 'value1'},
				'custom-id',
			);

			expect(result).toBeDefined();
			expect(result).toEqual(mockResponse);
		});

		it('should handle service errors gracefully', async () => {
			mockMcpServerServiceInstance.handleMcpServerRequest.mockRejectedValue(
				new Error('Tool call error'),
			);

			await expect(callMcpTool('server-1', 'test-tool', {})).rejects.toThrow(
				'Failed to call MCP tool: Error: Tool call error',
			);
		});

		it('should handle empty arguments', async () => {
			const mockResponse = {
				jsonrpc: '2.0',
				result: {success: true},
				id: 'tools-call',
			};

			mockMcpServerServiceInstance.handleMcpServerRequest.mockResolvedValue(
				mockResponse,
			);

			const result = await callMcpTool('server-1', 'test-tool');

			expect(result).toBeDefined();
			expect(result).toEqual(mockResponse);
		});

		it('should handle complex arguments', async () => {
			const complexArgs = {
				param1: 'value1',
				param2: {nested: 'value'},
				param3: [1, 2, 3],
				param4: true,
			};
			const mockResponse = {
				jsonrpc: '2.0',
				result: {success: true},
				id: 'tools-call',
			};

			mockMcpServerServiceInstance.handleMcpServerRequest.mockResolvedValue(
				mockResponse,
			);

			const result = await callMcpTool('server-1', 'complex-tool', complexArgs);

			expect(result).toBeDefined();
			expect(result).toEqual(mockResponse);
		});
	});

	describe('Integration Tests', () => {
		it('should handle full workflow with multiple operations', async () => {
			const mockServer: McpServer = {
				id: 'workflow-server',
				name: 'Workflow Server',
				description: 'Workflow description',
				serverUrl: 'https://api.workflow.com',
				enabled: true,
				authToken: 'workflow-token',
			};

			mockMcpServerServiceInstance.listMcpServers.mockResolvedValue([
				mockServer,
			]);
			mockMcpServerServiceInstance.getMcpServer.mockResolvedValue(mockServer);
			mockMcpServerServiceInstance.saveMcpServer.mockResolvedValue(undefined);
			mockMcpServerServiceInstance.deleteMcpServer.mockResolvedValue(undefined);
			mockMcpServerServiceInstance.handleMcpServerRequest.mockResolvedValue({
				success: true,
			});

			// Test list
			const listResult = await listMcpServers();
			expect(listResult.data).toHaveLength(1);
			expect(listResult.data[0].id).toBe('workflow-server');

			// Test get
			const getResult = await getMcpServer('workflow-server');
			expect(getResult).toEqual(mockServer);

			// Test save
			await expect(saveMcpServer(mockServer)).resolves.not.toThrow();

			// Test tool call
			const toolResult = await callMcpTool('workflow-server', 'test-tool', {});
			expect(toolResult.success).toBe(true);

			// Test delete
			await expect(deleteMcpServer('workflow-server')).resolves.not.toThrow();
		});

		it('should handle error scenarios gracefully', async () => {
			mockMcpServerServiceInstance.listMcpServers.mockRejectedValue(
				new Error('List error'),
			);
			mockMcpServerServiceInstance.getMcpServer.mockRejectedValue(
				new Error('Get error'),
			);
			mockMcpServerServiceInstance.saveMcpServer.mockRejectedValue(
				new Error('Save error'),
			);
			mockMcpServerServiceInstance.deleteMcpServer.mockRejectedValue(
				new Error('Delete error'),
			);
			mockMcpServerServiceInstance.handleMcpServerRequest.mockRejectedValue(
				new Error('Request error'),
			);

			await expect(listMcpServers()).rejects.toThrow(
				'Failed to list MCP servers',
			);
			await expect(getMcpServer('test')).rejects.toThrow(
				'Failed to get MCP server',
			);
			await expect(saveMcpServer({} as McpServer)).rejects.toThrow(
				'Failed to save MCP server',
			);
			await expect(deleteMcpServer('test')).rejects.toThrow(
				'Failed to delete MCP server',
			);
			await expect(handleMcpServerRequest('test', {})).rejects.toThrow(
				'Failed to handle MCP server request',
			);
		});
	});

	describe('Performance Tests', () => {
		it('should handle multiple rapid listMcpServers calls', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'server-1',
					name: 'Server 1',
					description: 'Description 1',
					serverUrl: 'https://api.test1.com',
					enabled: true,
				},
			];

			mockMcpServerServiceInstance.listMcpServers.mockResolvedValue(
				mockServers,
			);

			const start = Date.now();
			const promises = Array.from({length: 20}, () => listMcpServers());
			const results = await Promise.all(promises);
			const end = Date.now();

			expect(results).toHaveLength(20);
			expect(end - start).toBeLessThan(5000); // Should complete within 5 seconds

			results.forEach(result => {
				expect(result.object).toBe('list');
				expect(result.data).toHaveLength(1);
				expect(result.data[0].id).toBe('server-1');
			});
		});

		it('should handle concurrent tool calls efficiently', async () => {
			const mockResponse = {
				jsonrpc: '2.0',
				result: {success: true},
				id: 'tools-call',
			};

			mockMcpServerServiceInstance.handleMcpServerRequest.mockResolvedValue(
				mockResponse,
			);

			const start = Date.now();
			const promises = Array.from({length: 10}, (_, i) =>
				callMcpTool('server-1', `tool-${i}`, {arg: `value-${i}`}),
			);
			const results = await Promise.all(promises);
			const end = Date.now();

			expect(results).toHaveLength(10);
			expect(end - start).toBeLessThan(5000); // Should complete within 5 seconds

			results.forEach(result => {
				expect(result).toEqual(mockResponse);
			});
		});
	});
});
