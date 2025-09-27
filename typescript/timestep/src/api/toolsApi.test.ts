import {describe, it, expect, vi, beforeEach} from 'vitest';
import {
	TEST_TIMESTEP_PATHS,
	TEST_MCP_TOOLS,
} from '../__fixtures__/testPaths.js';

// Mock utils to prevent MCP connections
vi.mock('../utils.js', async importOriginal => {
	const actual = (await importOriginal()) as any;
	return {
		...actual,
		getTimestepPaths: vi.fn(() => TEST_TIMESTEP_PATHS),
		listAllMcpTools: vi.fn(() => Promise.resolve(TEST_MCP_TOOLS)),
		createMcpClient: vi.fn(),
	};
});

// Mock mcpServersApi
vi.mock('./mcpServersApi.js', () => ({
	callMcpTool: vi.fn(),
}));

// Mock repository container
vi.mock('../services/backing/repositoryContainer.js', () => ({
	DefaultRepositoryContainer: vi.fn().mockImplementation(() => ({
		mcpServerRepository: {
			list: vi.fn(() => Promise.resolve([])),
		},
	})),
}));

import {listTools, callToolById} from './toolsApi.js';

describe('toolsApi', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	describe('listTools', () => {
		it('should be a function', () => {
			expect(typeof listTools).toBe('function');
		});

		it('should return a promise', () => {
			const result = listTools();
			expect(result).toBeInstanceOf(Promise);
		});

		it('should execute listTools with real data', async () => {
			const result = await listTools();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
			expect(result.data.length).toBeGreaterThanOrEqual(0);
		});

		it('should handle multiple listTools operations', async () => {
			const result1 = await listTools();
			const result2 = await listTools();

			expect(result1.object).toBe('list');
			expect(result2.object).toBe('list');
			expect(Array.isArray(result1.data)).toBe(true);
			expect(Array.isArray(result2.data)).toBe(true);
		});

		it('should handle concurrent listTools operations', async () => {
			const promises = [listTools(), listTools(), listTools()];

			const results = await Promise.all(promises);

			expect(results).toHaveLength(3);
			results.forEach(result => {
				expect(result.object).toBe('list');
				expect(Array.isArray(result.data)).toBe(true);
			});
		});

		it('should handle error scenarios gracefully', async () => {
			const {listAllMcpTools} = await import('../utils.js');
			vi.mocked(listAllMcpTools).mockRejectedValueOnce(
				new Error('MCP connection failed'),
			);

			await expect(listTools()).rejects.toThrow(
				'Failed to list tools: Error: MCP connection failed',
			);
		});

		it('should handle rapid successive calls', async () => {
			const promises = Array.from({length: 10}, () => listTools());
			const results = await Promise.all(promises);

			expect(results).toHaveLength(10);
			results.forEach(result => {
				expect(result.object).toBe('list');
				expect(Array.isArray(result.data)).toBe(true);
			});
		});

		it('should handle tool data structure validation', async () => {
			const result = await listTools();

			expect(result).toHaveProperty('object', 'list');
			expect(result).toHaveProperty('data');
			expect(Array.isArray(result.data)).toBe(true);

			if (result.data.length > 0) {
				const tool = result.data[0];
				expect(tool).toHaveProperty('id');
				expect(tool).toHaveProperty('name');
				expect(tool).toHaveProperty('description');
				expect(tool).toHaveProperty('serverId');
				expect(tool).toHaveProperty('serverName');
				expect(tool).toHaveProperty('inputSchema');
				expect(tool).toHaveProperty('category');
				expect(tool).toHaveProperty('status');
				expect(tool.status).toBe('available');
			}
		});
	});

	describe('callToolById', () => {
		it('should be a function', () => {
			expect(typeof callToolById).toBe('function');
		});

		it('should return a promise', () => {
			const result = callToolById('server1.tool1');
			expect(result).toBeInstanceOf(Promise);
		});

		it('should execute callToolById with valid toolId', async () => {
			const {callMcpTool} = await import('./mcpServersApi.js');
			vi.mocked(callMcpTool).mockResolvedValueOnce({result: 'success'});

			const result = await callToolById('server1.tool1', {arg1: 'value1'});

			expect(callMcpTool).toHaveBeenCalledWith(
				'server1',
				'tool1',
				{arg1: 'value1'},
				'tools-call',
				expect.any(Object),
			);
			expect(result).toEqual({result: 'success'});
		});

		it('should handle different toolId formats', async () => {
			const {callMcpTool} = await import('./mcpServersApi.js');
			vi.mocked(callMcpTool).mockResolvedValue({result: 'success'});

			await callToolById('server123.tool456', {test: 'data'});
			expect(callMcpTool).toHaveBeenCalledWith(
				'server123',
				'tool456',
				{test: 'data'},
				'tools-call',
				expect.any(Object),
			);

			await callToolById('my-server.my-tool', {param: 'value'});
			expect(callMcpTool).toHaveBeenCalledWith(
				'my-server',
				'my-tool',
				{param: 'value'},
				'tools-call',
				expect.any(Object),
			);
		});

		it('should handle custom id parameter', async () => {
			const {callMcpTool} = await import('./mcpServersApi.js');
			vi.mocked(callMcpTool).mockResolvedValue({result: 'success'});

			await callToolById('server1.tool1', {arg: 'value'}, 'custom-id');

			expect(callMcpTool).toHaveBeenCalledWith(
				'server1',
				'tool1',
				{arg: 'value'},
				'custom-id',
				expect.any(Object),
			);
		});

		it('should handle empty args parameter', async () => {
			const {callMcpTool} = await import('./mcpServersApi.js');
			vi.mocked(callMcpTool).mockResolvedValue({result: 'success'});

			await callToolById('server1.tool1');

			expect(callMcpTool).toHaveBeenCalledWith(
				'server1',
				'tool1',
				{},
				'tools-call',
				expect.any(Object),
			);
		});

		it('should throw error for invalid toolId format', async () => {
			await expect(callToolById('invalid-tool-id')).rejects.toThrow(
				'Invalid toolId format. Expected {serverId}.{toolName}',
			);
		});

		it('should throw error for toolId without dot', async () => {
			await expect(callToolById('justtoolname')).rejects.toThrow(
				'Invalid toolId format. Expected {serverId}.{toolName}',
			);
		});

		it('should throw error for empty toolId', async () => {
			await expect(callToolById('')).rejects.toThrow(
				'Invalid toolId format. Expected {serverId}.{toolName}',
			);
		});

		it('should handle multiple callToolById operations', async () => {
			const {callMcpTool} = await import('./mcpServersApi.js');
			vi.mocked(callMcpTool).mockResolvedValue({result: 'success'});

			const promises = [
				callToolById('server1.tool1', {arg1: 'value1'}),
				callToolById('server2.tool2', {arg2: 'value2'}),
				callToolById('server3.tool3', {arg3: 'value3'}),
			];

			const results = await Promise.all(promises);

			expect(results).toHaveLength(3);
			results.forEach(result => {
				expect(result).toEqual({result: 'success'});
			});
			expect(callMcpTool).toHaveBeenCalledTimes(3);
		});

		it('should handle concurrent callToolById operations', async () => {
			const {callMcpTool} = await import('./mcpServersApi.js');
			vi.mocked(callMcpTool).mockResolvedValue({result: 'success'});

			const promises = Array.from({length: 5}, (_, i) =>
				callToolById(`server${i}.tool${i}`, {index: i}),
			);

			const results = await Promise.all(promises);

			expect(results).toHaveLength(5);
			expect(callMcpTool).toHaveBeenCalledTimes(5);
		});

		it('should handle callMcpTool errors', async () => {
			const {callMcpTool} = await import('./mcpServersApi.js');
			vi.mocked(callMcpTool).mockRejectedValueOnce(
				new Error('MCP server error'),
			);

			await expect(
				callToolById('server1.tool1', {arg: 'value'}),
			).rejects.toThrow('MCP server error');
		});

		it('should handle complex args parameter', async () => {
			const {callMcpTool} = await import('./mcpServersApi.js');
			vi.mocked(callMcpTool).mockResolvedValue({result: 'success'});

			const complexArgs = {
				string: 'test',
				number: 123,
				boolean: true,
				array: [1, 2, 3],
				object: {nested: 'value'},
				null: null,
			};

			await callToolById('server1.tool1', complexArgs);

			expect(callMcpTool).toHaveBeenCalledWith(
				'server1',
				'tool1',
				complexArgs,
				'tools-call',
				expect.any(Object),
			);
		});
	});

	describe('Performance Tests', () => {
		it('should handle large number of listTools operations', async () => {
			const start = Date.now();
			const promises = Array.from({length: 20}, () => listTools());
			const results = await Promise.all(promises);
			const end = Date.now();

			expect(results).toHaveLength(20);
			expect(end - start).toBeLessThan(5000); // Should complete within 5 seconds

			results.forEach(result => {
				expect(result.object).toBe('list');
				expect(Array.isArray(result.data)).toBe(true);
			});
		});

		it('should handle memory efficiently with multiple operations', async () => {
			const results = [];
			for (let i = 0; i < 10; i++) {
				const result = await listTools();
				results.push(result);
			}

			expect(results).toHaveLength(10);
			results.forEach(result => {
				expect(result.object).toBe('list');
				expect(Array.isArray(result.data)).toBe(true);
			});
		});

		it('should handle concurrent callToolById operations efficiently', async () => {
			const {callMcpTool} = await import('./mcpServersApi.js');
			vi.mocked(callMcpTool).mockResolvedValue({result: 'success'});

			const start = Date.now();
			const promises = Array.from({length: 15}, (_, i) =>
				callToolById(`server${i}.tool${i}`, {index: i}),
			);
			const results = await Promise.all(promises);
			const end = Date.now();

			expect(results).toHaveLength(15);
			expect(end - start).toBeLessThan(5000); // Should complete within 5 seconds
			expect(callMcpTool).toHaveBeenCalledTimes(15);
		});
	});
});
