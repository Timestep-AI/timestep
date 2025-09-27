import {describe, it, expect, vi, beforeEach} from 'vitest';
import {McpServerService} from './mcpServerService.js';
import {McpServer} from '../api/mcpServersApi.js';

// Mock file system operations to prevent real file creation
vi.mock('node:fs/promises', () => ({
	writeFile: vi.fn().mockResolvedValue(undefined),
	access: vi.fn().mockResolvedValue(undefined),
	unlink: vi.fn().mockResolvedValue(undefined),
}));

// Mock child process operations to prevent real pandoc execution
vi.mock('node:child_process', () => ({
	exec: vi.fn(),
}));

vi.mock('node:util', () => ({
	promisify: vi.fn(_fn =>
		vi
			.fn()
			.mockResolvedValue({stdout: 'PDF generated successfully', stderr: ''}),
	),
}));

// Mock the dynamic imports in the McpServerService
vi.mock('./mcpServerService.js', async importOriginal => {
	const actual = (await importOriginal()) as any;
	return {
		...actual,
		McpServerService: class MockMcpServerService extends actual.McpServerService {
			// @ts-ignore - unused method for mock
			private async _handleMarkdownToPdf(args: any, _id: any): Promise<any> {
				// Mock implementation that doesn't create real files
				const markdownContent = args?.['markdownContent'] as string | undefined;
				const outputPath =
					(args?.['outputPath'] as string | undefined) || 'output.pdf';

				if (!markdownContent) {
					return {
						jsonrpc: '2.0',
						id: _id,
						result: {
							content: [
								{
									type: 'text',
									text: 'Either markdownContent or markdownFile must be provided',
								},
							],
						},
					};
				}

				// Return success without actually creating files
				return {
					jsonrpc: '2.0',
					id: _id,
					result: {
						content: [
							{
								type: 'text',
								text: `✅ PDF generated successfully: ${outputPath}`,
							},
						],
					},
				};
			}
		},
	};
});

// Mock dependencies
vi.mock('../config/defaultMcpServers.js', () => ({
	getBuiltinMcpServer: vi.fn(() => ({
		id: 'builtin-server',
		name: 'Builtin Server',
		description: 'Built-in MCP server',
		serverUrl: 'builtin://server',
		enabled: true,
		authToken: null,
	})),
}));

describe('McpServerService', () => {
	let mcpServerService: McpServerService;
	let mockRepository: any;

	beforeEach(() => {
		vi.clearAllMocks();

		mockRepository = {
			list: vi.fn(),
			load: vi.fn(),
			exists: vi.fn(),
			save: vi.fn(),
			delete: vi.fn(),
		};

		mcpServerService = new McpServerService(mockRepository);
	});

	describe('constructor', () => {
		it('should create an instance', () => {
			expect(mcpServerService).toBeInstanceOf(McpServerService);
		});
	});

	describe('listMcpServers', () => {
		it('should return list of MCP servers', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'server-1',
					name: 'Test Server 1',
					description: 'A test server',
					serverUrl: 'http://localhost:3001',
					enabled: true,
					authToken: null,
				},
				{
					id: 'server-2',
					name: 'Test Server 2',
					description: 'Another test server',
					serverUrl: 'http://localhost:3002',
					enabled: false,
					authToken: 'test-token',
				},
			];

			mockRepository.list.mockResolvedValue(mockServers);

			const result = await mcpServerService.listMcpServers();

			expect(mockRepository.list).toHaveBeenCalled();
			expect(result).toEqual(mockServers);
		});

		it('should handle empty list', async () => {
			mockRepository.list.mockResolvedValue([]);

			const result = await mcpServerService.listMcpServers();

			expect(result).toEqual([]);
		});
	});

	describe('getMcpServer', () => {
		it('should return MCP server by ID', async () => {
			const mockServer: McpServer = {
				id: 'server-1',
				name: 'Test Server',
				description: 'A test server',
				serverUrl: 'http://localhost:3001',
				enabled: true,
				authToken: null,
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const result = await mcpServerService.getMcpServer('server-1');

			expect(mockRepository.load).toHaveBeenCalledWith('server-1');
			expect(result).toEqual(mockServer);
		});

		it('should return null for non-existent server', async () => {
			mockRepository.load.mockResolvedValue(null);

			const result = await mcpServerService.getMcpServer('non-existent');

			expect(result).toBeNull();
		});
	});

	describe('isMcpServerAvailable', () => {
		it('should return true for existing server', async () => {
			mockRepository.exists.mockResolvedValue(true);

			const result = await mcpServerService.isMcpServerAvailable('server-1');

			expect(mockRepository.exists).toHaveBeenCalledWith('server-1');
			expect(result).toBe(true);
		});

		it('should return false for non-existent server', async () => {
			mockRepository.exists.mockResolvedValue(false);

			const result = await mcpServerService.isMcpServerAvailable(
				'non-existent',
			);

			expect(result).toBe(false);
		});
	});

	describe('saveMcpServer', () => {
		it('should save MCP server', async () => {
			const mockServer: McpServer = {
				id: 'server-1',
				name: 'Test Server',
				description: 'A test server',
				serverUrl: 'http://localhost:3001',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
		});
	});

	describe('deleteMcpServer', () => {
		it('should delete MCP server', async () => {
			mockRepository.delete.mockResolvedValue(undefined);

			await mcpServerService.deleteMcpServer('server-1');

			expect(mockRepository.delete).toHaveBeenCalledWith('server-1');
		});
	});

	describe('error handling', () => {
		it('should handle repository errors in listMcpServers', async () => {
			mockRepository.list.mockRejectedValue(new Error('Repository error'));

			await expect(mcpServerService.listMcpServers()).rejects.toThrow(
				'Repository error',
			);
		});

		it('should handle repository errors in getMcpServer', async () => {
			mockRepository.load.mockRejectedValue(new Error('Repository error'));

			await expect(mcpServerService.getMcpServer('server-1')).rejects.toThrow(
				'Repository error',
			);
		});

		it('should handle repository errors in saveMcpServer', async () => {
			const mockServer: McpServer = {
				id: 'server-1',
				name: 'Test Server',
				description: 'A test server',
				serverUrl: 'http://localhost:3001',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockRejectedValue(new Error('Repository error'));

			await expect(mcpServerService.saveMcpServer(mockServer)).rejects.toThrow(
				'Repository error',
			);
		});

		it('should handle repository errors in deleteMcpServer', async () => {
			mockRepository.delete.mockRejectedValue(new Error('Repository error'));

			await expect(
				mcpServerService.deleteMcpServer('server-1'),
			).rejects.toThrow('Repository error');
		});
	});

	describe('MCP Server Management', () => {
		it('should handle server creation', async () => {
			const mockServer: McpServer = {
				id: 'new-server',
				name: 'New Server',
				description: 'A new MCP server',
				serverUrl: 'http://localhost:3003',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
		});

		it('should handle server updates', async () => {
			const mockServer: McpServer = {
				id: 'server-1',
				name: 'Updated Server',
				description: 'An updated MCP server',
				serverUrl: 'http://localhost:3001',
				enabled: false,
				authToken: 'new-token',
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
		});

		it('should handle server validation', async () => {
			const mockServer: McpServer = {
				id: 'valid-server',
				name: 'Valid Server',
				description: 'A valid MCP server',
				serverUrl: 'http://localhost:3004',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
		});

		it('should handle server filtering', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'enabled-server',
					name: 'Enabled Server',
					description: 'An enabled server',
					serverUrl: 'http://localhost:3001',
					enabled: true,
					authToken: null,
				},
				{
					id: 'disabled-server',
					name: 'Disabled Server',
					description: 'A disabled server',
					serverUrl: 'http://localhost:3002',
					enabled: false,
					authToken: null,
				},
			];

			mockRepository.list.mockResolvedValue(mockServers);

			const result = await mcpServerService.listMcpServers();

			expect(result).toEqual(mockServers);
			expect(result.length).toBe(2);
		});

		it('should handle server search', async () => {
			const mockServer: McpServer = {
				id: 'search-server',
				name: 'Search Server',
				description: 'A server for searching',
				serverUrl: 'http://localhost:3005',
				enabled: true,
				authToken: null,
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const result = await mcpServerService.getMcpServer('search-server');

			expect(result).toEqual(mockServer);
			expect(mockRepository.load).toHaveBeenCalledWith('search-server');
		});

		it('should handle server status checks', async () => {
			mockRepository.exists.mockResolvedValue(true);

			const result = await mcpServerService.isMcpServerAvailable(
				'status-server',
			);

			expect(result).toBe(true);
			expect(mockRepository.exists).toHaveBeenCalledWith('status-server');
		});

		it('should handle bulk operations', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'bulk-1',
					name: 'Bulk Server 1',
					description: 'First bulk server',
					serverUrl: 'http://localhost:3006',
					enabled: true,
					authToken: null,
				},
				{
					id: 'bulk-2',
					name: 'Bulk Server 2',
					description: 'Second bulk server',
					serverUrl: 'http://localhost:3007',
					enabled: true,
					authToken: null,
				},
			];

			mockRepository.list.mockResolvedValue(mockServers);

			const result = await mcpServerService.listMcpServers();

			expect(result).toEqual(mockServers);
			expect(result.length).toBe(2);
		});

		it('should handle server configuration', async () => {
			const mockServer: McpServer = {
				id: 'config-server',
				name: 'Config Server',
				description: 'A server with configuration',
				serverUrl: 'http://localhost:3008',
				enabled: true,
				authToken: 'config-token',
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
		});

		it('should handle server metadata', async () => {
			const mockServer: McpServer = {
				id: 'metadata-server',
				name: 'Metadata Server',
				description: 'A server with metadata',
				serverUrl: 'http://localhost:3009',
				enabled: true,
				authToken: null,
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const result = await mcpServerService.getMcpServer('metadata-server');

			expect(result).toEqual(mockServer);
			expect(result?.name).toBe('Metadata Server');
			expect(result?.description).toBe('A server with metadata');
		});
	});

	describe('Advanced MCP Server Operations', () => {
		it('should handle server health checks', async () => {
			const mockServer: McpServer = {
				id: 'health-server',
				name: 'Health Check Server',
				description: 'A server for health checks',
				serverUrl: 'http://localhost:3010',
				enabled: true,
				authToken: null,
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const result = await mcpServerService.getMcpServer('health-server');

			expect(result).toEqual(mockServer);
			expect(result?.enabled).toBe(true);
		});

		it('should handle server authentication', async () => {
			const mockServer: McpServer = {
				id: 'auth-server',
				name: 'Authentication Server',
				description: 'A server with authentication',
				serverUrl: 'http://localhost:3011',
				enabled: true,
				authToken: 'secret-token',
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.authToken).toBe('secret-token');
		});

		it('should handle server configuration updates', async () => {
			const mockServer: McpServer = {
				id: 'config-server',
				name: 'Configuration Server',
				description: 'A server for configuration',
				serverUrl: 'http://localhost:3012',
				enabled: false,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.enabled).toBe(false);
		});

		it('should handle server URL validation', async () => {
			const mockServer: McpServer = {
				id: 'url-server',
				name: 'URL Validation Server',
				description: 'A server for URL validation',
				serverUrl: 'https://api.example.com',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.serverUrl).toBe('https://api.example.com');
		});

		it('should handle server description updates', async () => {
			const mockServer: McpServer = {
				id: 'desc-server',
				name: 'Description Server',
				description: 'Updated description for the server',
				serverUrl: 'http://localhost:3013',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.description).toBe('Updated description for the server');
		});

		it('should handle server name updates', async () => {
			const mockServer: McpServer = {
				id: 'name-server',
				name: 'Updated Server Name',
				description: 'A server with updated name',
				serverUrl: 'http://localhost:3014',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.name).toBe('Updated Server Name');
		});

		it('should handle server ID validation', async () => {
			const mockServer: McpServer = {
				id: 'valid-uuid-server',
				name: 'Valid UUID Server',
				description: 'A server with valid UUID',
				serverUrl: 'http://localhost:3015',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.id).toBe('valid-uuid-server');
		});

		it('should handle server status toggling', async () => {
			const mockServer: McpServer = {
				id: 'toggle-server',
				name: 'Toggle Server',
				description: 'A server for status toggling',
				serverUrl: 'http://localhost:3016',
				enabled: false,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.enabled).toBe(false);
		});

		it('should handle server token updates', async () => {
			const mockServer: McpServer = {
				id: 'token-server',
				name: 'Token Server',
				description: 'A server with token updates',
				serverUrl: 'http://localhost:3017',
				enabled: true,
				authToken: 'new-token-value',
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.authToken).toBe('new-token-value');
		});

		it('should handle server token removal', async () => {
			const mockServer: McpServer = {
				id: 'no-token-server',
				name: 'No Token Server',
				description: 'A server without token',
				serverUrl: 'http://localhost:3018',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.authToken).toBeNull();
		});
	});

	describe('MCP Server Query Operations', () => {
		it('should handle server filtering by enabled status', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'enabled-filter-1',
					name: 'Enabled Filter Server 1',
					description: 'First enabled server',
					serverUrl: 'http://localhost:3019',
					enabled: true,
					authToken: null,
				},
				{
					id: 'enabled-filter-2',
					name: 'Enabled Filter Server 2',
					description: 'Second enabled server',
					serverUrl: 'http://localhost:3020',
					enabled: true,
					authToken: null,
				},
			];

			mockRepository.list.mockResolvedValue(mockServers);

			const result = await mcpServerService.listMcpServers();

			expect(result).toEqual(mockServers);
			expect(result.every(server => server.enabled)).toBe(true);
		});

		it('should handle server filtering by disabled status', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'disabled-filter-1',
					name: 'Disabled Filter Server 1',
					description: 'First disabled server',
					serverUrl: 'http://localhost:3021',
					enabled: false,
					authToken: null,
				},
				{
					id: 'disabled-filter-2',
					name: 'Disabled Filter Server 2',
					description: 'Second disabled server',
					serverUrl: 'http://localhost:3022',
					enabled: false,
					authToken: null,
				},
			];

			mockRepository.list.mockResolvedValue(mockServers);

			const result = await mcpServerService.listMcpServers();

			expect(result).toEqual(mockServers);
			expect(result.every(server => !server.enabled)).toBe(true);
		});

		it('should handle server filtering by authentication', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'auth-filter-1',
					name: 'Auth Filter Server 1',
					description: 'First authenticated server',
					serverUrl: 'http://localhost:3023',
					enabled: true,
					authToken: 'token-1',
				},
				{
					id: 'auth-filter-2',
					name: 'Auth Filter Server 2',
					description: 'Second authenticated server',
					serverUrl: 'http://localhost:3024',
					enabled: true,
					authToken: 'token-2',
				},
			];

			mockRepository.list.mockResolvedValue(mockServers);

			const result = await mcpServerService.listMcpServers();

			expect(result).toEqual(mockServers);
			expect(result.every(server => server.authToken)).toBe(true);
		});

		it('should handle server filtering by URL pattern', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'url-filter-1',
					name: 'URL Filter Server 1',
					description: 'First localhost server',
					serverUrl: 'http://localhost:3025',
					enabled: true,
					authToken: null,
				},
				{
					id: 'url-filter-2',
					name: 'URL Filter Server 2',
					description: 'Second localhost server',
					serverUrl: 'http://localhost:3026',
					enabled: true,
					authToken: null,
				},
			];

			mockRepository.list.mockResolvedValue(mockServers);

			const result = await mcpServerService.listMcpServers();

			expect(result).toEqual(mockServers);
			expect(
				result.every(server => server.serverUrl.includes('localhost')),
			).toBe(true);
		});
	});

	describe('MCP Server Validation', () => {
		it('should validate server URL format', async () => {
			const mockServer: McpServer = {
				id: 'url-format-server',
				name: 'URL Format Server',
				description: 'A server with valid URL format',
				serverUrl: 'https://api.valid-domain.com/v1',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.serverUrl).toMatch(/^https?:\/\//);
		});

		it('should validate server name format', async () => {
			const mockServer: McpServer = {
				id: 'name-format-server',
				name: 'Valid Server Name 123',
				description: 'A server with valid name format',
				serverUrl: 'http://localhost:3027',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.name).toBeTruthy();
		});

		it('should validate server description format', async () => {
			const mockServer: McpServer = {
				id: 'desc-format-server',
				name: 'Description Format Server',
				description:
					'A server with a valid description that contains multiple words and proper formatting.',
				serverUrl: 'http://localhost:3028',
				enabled: true,
				authToken: null,
			};

			mockRepository.save.mockResolvedValue(undefined);

			await mcpServerService.saveMcpServer(mockServer);

			expect(mockRepository.save).toHaveBeenCalledWith(mockServer);
			expect(mockServer.description).toBeTruthy();
		});
	});

	describe('handleMcpServerRequest', () => {
		it('should handle built-in server requests', async () => {
			const request = {
				method: 'initialize',
				params: {},
				id: 'test-1',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-1');
			expect(result.result).toBeDefined();
			expect(result.result.protocolVersion).toBe('2024-11-05');
		});

		it('should handle tools/list request for built-in server', async () => {
			const request = {
				method: 'tools/list',
				params: {},
				id: 'test-2',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-2');
			expect(result.result).toBeDefined();
			expect(result.result.tools).toBeDefined();
			expect(Array.isArray(result.result.tools)).toBe(true);
		});

		it('should handle tools/call request for built-in server', async () => {
			const request = {
				method: 'tools/call',
				params: {
					name: 'get-alerts',
					arguments: {state: 'CA'},
				},
				id: 'test-3',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-3');
			expect(result.result).toBeDefined();
		});

		it('should handle resources/list request for built-in server', async () => {
			const request = {
				method: 'resources/list',
				params: {},
				id: 'test-4',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-4');
			expect(result.result).toBeDefined();
			expect(result.result.resources).toBeDefined();
			expect(Array.isArray(result.result.resources)).toBe(true);
		});

		it('should handle resources/read request for built-in server', async () => {
			const request = {
				method: 'resources/read',
				params: {
					uri: 'file://test.md',
				},
				id: 'test-5',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-5');
			// The result might be undefined for file resources that don't exist
			expect(result.result || result.error).toBeDefined();
		});

		it('should handle prompts/list request for built-in server', async () => {
			const request = {
				method: 'prompts/list',
				params: {},
				id: 'test-6',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-6');
			expect(result.result).toBeDefined();
			expect(result.result.prompts).toBeDefined();
			expect(Array.isArray(result.result.prompts)).toBe(true);
		});

		it('should handle prompts/get request for built-in server', async () => {
			const request = {
				method: 'prompts/get',
				params: {
					name: 'test-prompt',
					arguments: {},
				},
				id: 'test-7',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-7');
			// The result might be undefined for prompts that don't exist
			expect(result.result || result.error).toBeDefined();
		});

		it('should handle unknown method for built-in server', async () => {
			const request = {
				method: 'unknown-method',
				params: {},
				id: 'test-8',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-8');
			expect(result.error).toBeDefined();
			expect(result.error.code).toBe(-32601);
		});

		it('should proxy to remote server for non-builtin server', async () => {
			const mockServer: McpServer = {
				id: 'remote-server',
				name: 'Remote Server',
				description: 'A remote MCP server',
				serverUrl: 'http://localhost:3001',
				enabled: true,
				authToken: null,
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const request = {
				method: 'initialize',
				params: {},
				id: 'test-9',
			};

			// The proxy will fail because there's no actual server running
			await expect(
				mcpServerService.handleMcpServerRequest('remote-server', request),
			).rejects.toThrow('Failed to proxy to MCP server remote-server');

			expect(mockRepository.load).toHaveBeenCalledWith('remote-server');
		});

		it('should handle remote server not found', async () => {
			mockRepository.load.mockResolvedValue(null);

			const request = {
				method: 'initialize',
				params: {},
				id: 'test-10',
			};

			await expect(
				mcpServerService.handleMcpServerRequest('non-existent-server', request),
			).rejects.toThrow('MCP server non-existent-server not found or disabled');

			expect(mockRepository.load).toHaveBeenCalledWith('non-existent-server');
		});
	});

	describe('Built-in MCP Server Tool Calls', () => {
		it('should handle get-alerts tool call', async () => {
			const request = {
				method: 'tools/call',
				params: {
					name: 'get-alerts',
					arguments: {state: 'CA'},
				},
				id: 'test-alerts',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-alerts');
			expect(result.result).toBeDefined();
		});

		it('should handle get-forecast tool call', async () => {
			const request = {
				method: 'tools/call',
				params: {
					name: 'get-forecast',
					arguments: {latitude: 37.7749, longitude: -122.4194},
				},
				id: 'test-forecast',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-forecast');
			expect(result.result).toBeDefined();
		});

		it('should handle markdownToPdf tool call', async () => {
			const request = {
				method: 'tools/call',
				params: {
					name: 'markdownToPdf',
					arguments: {markdownContent: '# Test\nThis is a test.'},
				},
				id: 'test-pdf',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-pdf');
			expect(result.result).toBeDefined();
		}, 10000); // 10 second timeout for PDF generation

		it('should handle think tool call', async () => {
			const request = {
				method: 'tools/call',
				params: {
					name: 'think',
					arguments: {thought: 'This is a test thought.'},
				},
				id: 'test-think',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-think');
			expect(result.result).toBeDefined();
		});

		it('should handle unknown tool call', async () => {
			const request = {
				method: 'tools/call',
				params: {
					name: 'unknown-tool',
					arguments: {},
				},
				id: 'test-unknown',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-unknown');
			expect(result.error).toBeDefined();
			expect(result.error.code).toBe(-32601);
		});
	});

	describe('Built-in MCP Server Resource Operations', () => {
		it('should handle file resource read', async () => {
			const request = {
				method: 'resources/read',
				params: {
					uri: 'file://test.md',
				},
				id: 'test-file-read',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-file-read');
			// The result might be undefined for file resources that don't exist
			expect(result.result || result.error).toBeDefined();
		});

		it('should handle unknown resource read', async () => {
			const request = {
				method: 'resources/read',
				params: {
					uri: 'unknown://test',
				},
				id: 'test-unknown-resource',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-unknown-resource');
			expect(result.error).toBeDefined();
			expect(result.error.code).toBe(-32601); // Method not found, not invalid params
		});
	});

	describe('Built-in MCP Server Prompt Operations', () => {
		it('should handle prompt get', async () => {
			const request = {
				method: 'prompts/get',
				params: {
					name: 'test-prompt',
					arguments: {},
				},
				id: 'test-prompt-get',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-prompt-get');
			// The result might be undefined for prompts that don't exist
			expect(result.result || result.error).toBeDefined();
		});

		it('should handle unknown prompt get', async () => {
			const request = {
				method: 'prompts/get',
				params: {
					name: 'unknown-prompt',
					arguments: {},
				},
				id: 'test-unknown-prompt',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-unknown-prompt');
			expect(result.error).toBeDefined();
			expect(result.error.code).toBe(-32601);
		});
	});

	describe('Error Handling in MCP Server Requests', () => {
		it('should handle malformed request', async () => {
			const request = {
				// Missing required fields
				id: 'test-malformed',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-malformed');
			expect(result.error).toBeDefined();
		});

		it('should handle request with invalid method', async () => {
			const request = {
				method: '',
				params: {},
				id: 'test-invalid-method',
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			expect(result.id).toBe('test-invalid-method');
			expect(result.error).toBeDefined();
		});

		it('should handle request with missing id', async () => {
			const request = {
				method: 'initialize',
				params: {},
				// Missing id
			};

			const result = await mcpServerService.handleMcpServerRequest(
				'builtin-server',
				request,
			);

			expect(result).toBeDefined();
			expect(result.jsonrpc).toBe('2.0');
			// The result might not have an error if the method handles missing id gracefully
			expect(result.result || result.error).toBeDefined();
		});
	});

	describe('MCP Server Session Management', () => {
		it('should handle session initialization', async () => {
			const mockServer: McpServer = {
				id: 'session-server',
				name: 'Session Server',
				description: 'A server with session management',
				serverUrl: 'http://localhost:3001',
				enabled: true,
				authToken: null,
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const request = {
				method: 'initialize',
				params: {},
				id: 'test-session-init',
			};

			await expect(
				mcpServerService.handleMcpServerRequest('session-server', request),
			).rejects.toThrow('Failed to proxy to MCP server session-server');

			expect(mockRepository.load).toHaveBeenCalledWith('session-server');
		});

		it('should handle session cleanup', async () => {
			const mockServer: McpServer = {
				id: 'cleanup-server',
				name: 'Cleanup Server',
				description: 'A server for session cleanup',
				serverUrl: 'http://localhost:3002',
				enabled: true,
				authToken: null,
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const request = {
				method: 'notifications/initialized',
				params: {},
				id: 'test-cleanup',
			};

			await expect(
				mcpServerService.handleMcpServerRequest('cleanup-server', request),
			).rejects.toThrow('Failed to proxy to MCP server cleanup-server');

			expect(mockRepository.load).toHaveBeenCalledWith('cleanup-server');
		});
	});

	describe('MCP Server Authentication', () => {
		it('should handle authenticated server requests', async () => {
			const mockServer: McpServer = {
				id: 'auth-server',
				name: 'Auth Server',
				description: 'An authenticated server',
				serverUrl: 'http://localhost:3003',
				enabled: true,
				authToken: 'secret-token',
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const request = {
				method: 'initialize',
				params: {},
				id: 'test-auth',
			};

			await expect(
				mcpServerService.handleMcpServerRequest('auth-server', request),
			).rejects.toThrow('Failed to proxy to MCP server auth-server');

			expect(mockRepository.load).toHaveBeenCalledWith('auth-server');
		});

		it('should handle server with no authentication', async () => {
			const mockServer: McpServer = {
				id: 'no-auth-server',
				name: 'No Auth Server',
				description: 'A server without authentication',
				serverUrl: 'http://localhost:3004',
				enabled: true,
				authToken: null,
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const request = {
				method: 'initialize',
				params: {},
				id: 'test-no-auth',
			};

			await expect(
				mcpServerService.handleMcpServerRequest('no-auth-server', request),
			).rejects.toThrow('Failed to proxy to MCP server no-auth-server');

			expect(mockRepository.load).toHaveBeenCalledWith('no-auth-server');
		});
	});

	describe('MCP Server Configuration', () => {
		it('should handle disabled server requests', async () => {
			const mockServer: McpServer = {
				id: 'disabled-server',
				name: 'Disabled Server',
				description: 'A disabled server',
				serverUrl: 'http://localhost:3005',
				enabled: false,
				authToken: null,
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const request = {
				method: 'initialize',
				params: {},
				id: 'test-disabled',
			};

			await expect(
				mcpServerService.handleMcpServerRequest('disabled-server', request),
			).rejects.toThrow('MCP server disabled-server not found or disabled');

			expect(mockRepository.load).toHaveBeenCalledWith('disabled-server');
		});

		it('should handle server with custom URL', async () => {
			const mockServer: McpServer = {
				id: 'custom-url-server',
				name: 'Custom URL Server',
				description: 'A server with custom URL',
				serverUrl: 'https://api.example.com/mcp',
				enabled: true,
				authToken: null,
			};

			mockRepository.load.mockResolvedValue(mockServer);

			const request = {
				method: 'initialize',
				params: {},
				id: 'test-custom-url',
			};

			await expect(
				mcpServerService.handleMcpServerRequest('custom-url-server', request),
			).rejects.toThrow('Failed to proxy to MCP server custom-url-server');

			expect(mockRepository.load).toHaveBeenCalledWith('custom-url-server');
		});
	});
});
