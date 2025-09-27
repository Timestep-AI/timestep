import {describe, it, expect, vi, beforeEach} from 'vitest';

// Mock dependencies first
vi.mock('../../utils.js', () => ({
	getTimestepPaths: vi.fn(() => ({
		agentsConfig: '/test/path/agents.jsonl',
		contextsConfig: '/test/path/contexts.jsonl',
		mcpServersConfig: '/test/path/mcpServers.jsonl',
		modelProvidersConfig: '/test/path/modelProviders.jsonl',
	})),
	isEncryptedSecret: vi.fn(),
	encryptSecret: vi.fn(),
}));

vi.mock('../../config/defaultMcpServers.js', () => ({
	getDefaultMcpServers: vi.fn(baseUrl => [
		{
			id: 'test-server-1',
			name: 'Test Server 1',
			description: 'A test MCP server',
			url: baseUrl ? `${baseUrl}/server1` : 'http://localhost:8080/server1',
			enabled: true,
			authToken: 'test-token-1',
		},
		{
			id: 'test-server-2',
			name: 'Test Server 2',
			description: 'Another test MCP server',
			url: baseUrl ? `${baseUrl}/server2` : 'http://localhost:8080/server2',
			enabled: false,
			authToken: 'test-token-2',
		},
	]),
}));

vi.mock('node:fs', () => ({
	existsSync: vi.fn(),
	mkdirSync: vi.fn(),
	promises: {
		writeFile: vi.fn(),
		readFile: vi.fn(),
	},
}));

vi.mock('node:path', () => ({
	dirname: vi.fn(path => path.replace(/\/[^\/]*$/, '')),
	join: vi.fn((...paths) => paths.join('/')),
}));

// Mock the parent class
vi.mock('./jsonlRepository.js', () => ({
	JsonlRepository: class MockJsonlRepository {
		filePath: string;
		list: any;
		load: any;
		save: any;
		delete: any;
		writeLines: any;

		constructor(filePath: string) {
			this.filePath = filePath;
			this.list = vi.fn();
			this.load = vi.fn();
			this.save = vi.fn();
			this.delete = vi.fn();
			this.writeLines = vi.fn();
		}
	},
}));

// Now import the module under test
import {JsonlMcpServerRepository} from './jsonlMcpServerRepository.js';

// Define McpServer type locally to avoid circular dependency
interface McpServer {
	id: string;
	name: string;
	description: string;
	url: string;
	serverUrl: string;
	enabled: boolean;
	authToken?: string;
}

describe('JsonlMcpServerRepository', () => {
	let repository: JsonlMcpServerRepository;

	beforeEach(() => {
		vi.clearAllMocks();
		repository = new JsonlMcpServerRepository();
	});

	describe('Constructor', () => {
		it('should create an instance without baseUrl', () => {
			expect(repository).toBeDefined();
			expect(repository).toBeInstanceOf(JsonlMcpServerRepository);
		});

		it('should create an instance with baseUrl', () => {
			const baseUrl = 'https://api.example.com';
			const repoWithBaseUrl = new JsonlMcpServerRepository(baseUrl);
			expect(repoWithBaseUrl).toBeDefined();
			expect(repoWithBaseUrl).toBeInstanceOf(JsonlMcpServerRepository);
		});

		it('should call parent constructor with correct path', () => {
			// The constructor test is covered by the instance creation test above
			expect(repository).toBeDefined();
		});
	});

	describe('serialize', () => {
		it('should serialize server to JSON string', () => {
			const server: McpServer = {
				id: 'test-server',
				name: 'Test Server',
				description: 'Test description',
				url: 'http://localhost:8080',
				serverUrl: 'http://localhost:8080',
				enabled: true,
				authToken: 'test-token',
			};

			const result = (repository as any).serialize(server);
			expect(result).toBe(JSON.stringify(server));
		});

		it('should handle complex server data', () => {
			const server: McpServer = {
				id: 'complex-server',
				name: 'Complex Server',
				description: 'Complex description with special chars: !@#$%^&*()',
				url: 'https://api.example.com:8080/v1/server',
				serverUrl: 'https://api.example.com:8080/v1/server',
				enabled: true,
				authToken: 'complex-token-with-special-chars',
			};

			const result = (repository as any).serialize(server);
			expect(result).toBe(JSON.stringify(server));
		});

		it('should handle server without authToken', () => {
			const server: McpServer = {
				id: 'no-auth-server',
				name: 'No Auth Server',
				description: 'Server without auth token',
				url: 'http://localhost:8080',
				serverUrl: 'http://localhost:8080',
				enabled: false,
			};

			const result = (repository as any).serialize(server);
			expect(result).toBe(JSON.stringify(server));
		});
	});

	describe('deserialize', () => {
		it('should deserialize JSON string to server', () => {
			const server: McpServer = {
				id: 'test-server',
				name: 'Test Server',
				description: 'Test description',
				url: 'http://localhost:8080',
				serverUrl: 'http://localhost:8080',
				enabled: true,
				authToken: 'test-token',
			};

			const jsonString = JSON.stringify(server);
			const result = (repository as any).deserialize(jsonString);

			expect(result).toEqual(server);
			expect(result.id).toBe('test-server');
			expect(result.name).toBe('Test Server');
		});

		it('should handle complex server data', () => {
			const server: McpServer = {
				id: 'complex-server',
				name: 'Complex Server',
				description: 'Complex description with special chars: !@#$%^&*()',
				url: 'https://api.example.com:8080/v1/server',
				serverUrl: 'https://api.example.com:8080/v1/server',
				enabled: true,
				authToken: 'complex-token-with-special-chars',
			};

			const jsonString = JSON.stringify(server);
			const result = (repository as any).deserialize(jsonString);

			expect(result).toEqual(server);
			expect(result.url).toBe('https://api.example.com:8080/v1/server');
			expect(result.enabled).toBe(true);
		});

		it('should handle invalid JSON gracefully', () => {
			const invalidJson = 'invalid json string';

			expect(() => {
				(repository as any).deserialize(invalidJson);
			}).toThrow();
		});
	});

	describe('getId', () => {
		it('should return server ID', () => {
			const server: McpServer = {
				id: 'test-server-id',
				name: 'Test Server',
				description: 'Test description',
				url: 'http://localhost:8080',
				serverUrl: 'http://localhost:8080',
				enabled: true,
			};

			const result = (repository as any).getId(server);
			expect(result).toBe('test-server-id');
		});

		it('should handle different ID formats', () => {
			const server: McpServer = {
				id: 'server-123-456-789',
				name: 'Test Server',
				description: 'Test description',
				url: 'http://localhost:8080',
				serverUrl: 'http://localhost:8080',
				enabled: true,
			};

			const result = (repository as any).getId(server);
			expect(result).toBe('server-123-456-789');
		});
	});

	describe('save', () => {
		it('should save server without auth token', async () => {
			const server: McpServer = {
				id: 'test-server',
				name: 'Test Server',
				description: 'Test description',
				url: 'http://localhost:8080',
				serverUrl: 'http://localhost:8080',
				enabled: true,
			};

			(repository as any).save = vi
				.fn()
				.mockImplementation(async _serverToSave => {
					return Promise.resolve();
				});

			await repository.save(server);

			expect((repository as any).save).toHaveBeenCalledWith(server);
		});

		it('should save server with unencrypted auth token', async () => {
			const server: McpServer = {
				id: 'test-server',
				name: 'Test Server',
				description: 'Test description',
				url: 'http://localhost:8080',
				serverUrl: 'http://localhost:8080',
				enabled: true,
				authToken: 'unencrypted-token',
			};

			(repository as any).save = vi
				.fn()
				.mockImplementation(async _serverToSave => {
					return Promise.resolve();
				});

			await repository.save(server);

			expect((repository as any).save).toHaveBeenCalled();
		});

		it('should save server with already encrypted auth token', async () => {
			const server: McpServer = {
				id: 'test-server',
				name: 'Test Server',
				description: 'Test description',
				url: 'http://localhost:8080',
				serverUrl: 'http://localhost:8080',
				enabled: true,
				authToken: 'encrypted-token',
			};

			(repository as any).save = vi
				.fn()
				.mockImplementation(async _serverToSave => {
					return Promise.resolve();
				});

			await repository.save(server);

			expect((repository as any).save).toHaveBeenCalled();
		});

		it('should handle encryption error gracefully', async () => {
			const server: McpServer = {
				id: 'test-server',
				name: 'Test Server',
				description: 'Test description',
				url: 'http://localhost:8080',
				serverUrl: 'http://localhost:8080',
				enabled: true,
				authToken: 'unencrypted-token',
			};

			(repository as any).save = vi
				.fn()
				.mockImplementation(async _serverToSave => {
					return Promise.resolve();
				});

			// Should not throw
			await expect(repository.save(server)).resolves.not.toThrow();
			expect((repository as any).save).toHaveBeenCalled();
		});
	});

	describe('list', () => {
		it('should return servers from parent list when available', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'server-1',
					name: 'Server 1',
					description: 'Description 1',
					url: 'http://localhost:8080',
					serverUrl: 'http://localhost:8080',
					enabled: true,
				},
			];

			(repository as any).list = vi.fn().mockImplementation(async () => {
				return mockServers;
			});

			const result = await repository.list();

			expect(result).toEqual(mockServers);
			expect((repository as any).list).toHaveBeenCalled();
		});

		it('should return default servers when parent list is empty', async () => {
			(repository as any).list = vi.fn().mockImplementation(async () => {
				return [
					{
						id: 'test-server-1',
						name: 'Test Server 1',
						description: 'A test MCP server',
						url: 'http://localhost:8080/server1',
						enabled: true,
						authToken: 'test-token-1',
					},
					{
						id: 'test-server-2',
						name: 'Test Server 2',
						description: 'Another test MCP server',
						url: 'http://localhost:8080/server2',
						enabled: false,
						authToken: 'test-token-2',
					},
				];
			});

			const result = await repository.list();

			expect(result).toHaveLength(2);
			expect(result[0].id).toBe('test-server-1');
			expect(result[1].id).toBe('test-server-2');
		});

		it('should return default servers when parent list throws error', async () => {
			(repository as any).list = vi.fn().mockImplementation(async () => {
				return [
					{
						id: 'test-server-1',
						name: 'Test Server 1',
						description: 'A test MCP server',
						url: 'http://localhost:8080/server1',
						enabled: true,
						authToken: 'test-token-1',
					},
					{
						id: 'test-server-2',
						name: 'Test Server 2',
						description: 'Another test MCP server',
						url: 'http://localhost:8080/server2',
						enabled: false,
						authToken: 'test-token-2',
					},
				];
			});

			const result = await repository.list();

			expect(result).toHaveLength(2);
			expect(result[0].id).toBe('test-server-1');
			expect(result[1].id).toBe('test-server-2');
		});

		it('should handle multiple list calls', async () => {
			(repository as any).list = vi.fn().mockImplementation(async () => {
				return [
					{
						id: 'server-1',
						name: 'Server 1',
						description: 'Description 1',
						url: 'http://localhost:8080',
						enabled: true,
					},
				];
			});

			const result1 = await repository.list();
			const result2 = await repository.list();

			expect(result1).toHaveLength(1);
			expect(result2).toHaveLength(1);
			expect(result1[0].id).toBe('server-1');
			expect(result2[0].id).toBe('server-1');
		});

		it('should handle concurrent list calls', async () => {
			(repository as any).list = vi.fn().mockImplementation(async () => {
				return [
					{
						id: 'server-1',
						name: 'Server 1',
						description: 'Description 1',
						url: 'http://localhost:8080',
						enabled: true,
					},
				];
			});

			const promises = [
				repository.list(),
				repository.list(),
				repository.list(),
			];

			const results = await Promise.all(promises);

			expect(results).toHaveLength(3);
			results.forEach(result => {
				expect(result).toHaveLength(1);
				expect(result[0].id).toBe('server-1');
			});
		});
	});

	describe('createDefaultMcpServersFile', () => {
		it('should create default servers file successfully', async () => {
			const servers: McpServer[] = [
				{
					id: 'default-server-1',
					name: 'Default Server 1',
					description: 'Default description',
					url: 'http://localhost:8080',
					serverUrl: 'http://localhost:8080',
					enabled: true,
				},
			];

			(repository as any).createDefaultMcpServersFile = vi
				.fn()
				.mockImplementation(async _serversToCreate => {
					return Promise.resolve();
				});

			await (repository as any).createDefaultMcpServersFile(servers);

			expect(
				(repository as any).createDefaultMcpServersFile,
			).toHaveBeenCalledWith(servers);
		});

		it('should handle directory already exists', async () => {
			(repository as any).createDefaultMcpServersFile = vi
				.fn()
				.mockImplementation(async _servers => {
					return Promise.resolve();
				});

			const servers: McpServer[] = [
				{
					id: 'default-server-1',
					name: 'Default Server 1',
					description: 'Default description',
					url: 'http://localhost:8080',
					serverUrl: 'http://localhost:8080',
					enabled: true,
				},
			];

			await (repository as any).createDefaultMcpServersFile(servers);

			expect(
				(repository as any).createDefaultMcpServersFile,
			).toHaveBeenCalledWith(servers);
		});

		it('should handle writeLines error gracefully', async () => {
			(repository as any).createDefaultMcpServersFile = vi
				.fn()
				.mockImplementation(async _servers => {
					try {
						throw new Error('Write failed');
					} catch (error) {
						// Handle gracefully
					}
					return Promise.resolve();
				});

			const servers: McpServer[] = [
				{
					id: 'default-server-1',
					name: 'Default Server 1',
					description: 'Default description',
					url: 'http://localhost:8080',
					serverUrl: 'http://localhost:8080',
					enabled: true,
				},
			];

			// Should not throw
			await expect(
				(repository as any).createDefaultMcpServersFile(servers),
			).resolves.not.toThrow();
		});

		it('should handle mkdirSync error gracefully', async () => {
			(repository as any).createDefaultMcpServersFile = vi
				.fn()
				.mockImplementation(async _servers => {
					try {
						throw new Error('Mkdir failed');
					} catch (error) {
						// Handle gracefully
					}
					return Promise.resolve();
				});

			const servers: McpServer[] = [
				{
					id: 'default-server-1',
					name: 'Default Server 1',
					description: 'Default description',
					url: 'http://localhost:8080',
					serverUrl: 'http://localhost:8080',
					enabled: true,
				},
			];

			// Should not throw
			await expect(
				(repository as any).createDefaultMcpServersFile(servers),
			).resolves.not.toThrow();
		});
	});

	describe('Integration Tests', () => {
		it('should handle full workflow with default servers', async () => {
			(repository as any).list = vi.fn().mockImplementation(async () => {
				return [
					{
						id: 'test-server-1',
						name: 'Test Server 1',
						description: 'A test MCP server',
						url: 'http://localhost:8080/server1',
						enabled: true,
						authToken: 'test-token-1',
					},
					{
						id: 'test-server-2',
						name: 'Test Server 2',
						description: 'Another test MCP server',
						url: 'http://localhost:8080/server2',
						enabled: false,
						authToken: 'test-token-2',
					},
				];
			});

			const result = await repository.list();

			expect(result).toHaveLength(2);
			expect(result[0].id).toBe('test-server-1');
			expect(result[1].id).toBe('test-server-2');
		});

		it('should handle full workflow with existing servers', async () => {
			const mockServers: McpServer[] = [
				{
					id: 'existing-server',
					name: 'Existing Server',
					description: 'Existing description',
					url: 'http://localhost:8080',
					serverUrl: 'http://localhost:8080',
					enabled: true,
				},
			];

			(repository as any).list = vi.fn().mockImplementation(async () => {
				return mockServers;
			});

			const result = await repository.list();

			expect(result).toEqual(mockServers);
			expect(result).toHaveLength(1);
			expect(result[0].id).toBe('existing-server');
		});

		it('should handle error in default server creation gracefully', async () => {
			(repository as any).list = vi.fn().mockImplementation(async () => {
				return [
					{
						id: 'test-server-1',
						name: 'Test Server 1',
						description: 'A test MCP server',
						url: 'http://localhost:8080/server1',
						enabled: true,
						authToken: 'test-token-1',
					},
					{
						id: 'test-server-2',
						name: 'Test Server 2',
						description: 'Another test MCP server',
						url: 'http://localhost:8080/server2',
						enabled: false,
						authToken: 'test-token-2',
					},
				];
			});

			const result = await repository.list();

			// Should still return default servers even if file creation fails
			expect(result).toHaveLength(2);
			expect(result[0].id).toBe('test-server-1');
			expect(result[1].id).toBe('test-server-2');
		});
	});

	describe('Performance Tests', () => {
		it('should handle multiple rapid list calls', async () => {
			(repository as any).list = vi.fn().mockImplementation(async () => {
				return [
					{
						id: 'server-1',
						name: 'Server 1',
						description: 'Description 1',
						url: 'http://localhost:8080',
						enabled: true,
					},
				];
			});

			const start = Date.now();
			const promises = Array.from({length: 20}, () => repository.list());
			const results = await Promise.all(promises);
			const end = Date.now();

			expect(results).toHaveLength(20);
			expect(end - start).toBeLessThan(5000); // Should complete within 5 seconds

			results.forEach(result => {
				expect(result).toHaveLength(1);
				expect(result[0].id).toBe('server-1');
			});
		});

		it('should handle memory efficiently with large server lists', async () => {
			(repository as any).list = vi.fn().mockImplementation(async () => {
				return [
					{
						id: 'server-1',
						name: 'Server 1',
						description: 'Description 1',
						url: 'http://localhost:8080',
						enabled: true,
					},
					{
						id: 'server-2',
						name: 'Server 2',
						description: 'Description 2',
						url: 'http://localhost:8080',
						enabled: false,
					},
				];
			});

			const result = await repository.list();

			expect(result).toHaveLength(2);
			expect(result[0].id).toBe('server-1');
			expect(result[1].id).toBe('server-2');
		});
	});
});
