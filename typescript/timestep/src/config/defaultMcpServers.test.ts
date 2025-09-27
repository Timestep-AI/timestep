import {describe, it, expect} from 'vitest';
import {
	getDefaultMcpServers,
	getBuiltinMcpServer,
} from './defaultMcpServers.js';

describe('defaultMcpServers', () => {
	describe('getDefaultMcpServers', () => {
		it('should return array of default MCP servers', () => {
			const servers = getDefaultMcpServers();

			expect(Array.isArray(servers)).toBe(true);
			expect(servers.length).toBeGreaterThan(0);
		});

		it('should return servers with required properties', () => {
			const servers = getDefaultMcpServers();
			const firstServer = servers[0];

			expect(firstServer).toHaveProperty('id');
			expect(firstServer).toHaveProperty('name');
			expect(firstServer).toHaveProperty('description');
			expect(firstServer).toHaveProperty('serverUrl');
			expect(firstServer).toHaveProperty('enabled');
		});

		it('should have valid UUIDs for server IDs', () => {
			const servers = getDefaultMcpServers();

			servers.forEach(server => {
				expect(server.id).toMatch(
					/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
				);
			});
		});

		it('should have non-empty names and descriptions', () => {
			const servers = getDefaultMcpServers();

			servers.forEach(server => {
				expect(server.name).toBeTruthy();
				expect(server.name.length).toBeGreaterThan(0);
				expect(server.description).toBeTruthy();
				expect(server.description.length).toBeGreaterThan(0);
			});
		});

		it('should have valid server URLs', () => {
			const servers = getDefaultMcpServers();

			servers.forEach(server => {
				expect(server.serverUrl).toBeTruthy();
				expect(server.serverUrl.length).toBeGreaterThan(0);
				expect(server.serverUrl).toMatch(/^https?:\/\//);
			});
		});

		it('should use custom baseUrl when provided', () => {
			const customBaseUrl = 'https://custom.example.com';
			const servers = getDefaultMcpServers(customBaseUrl);
			const builtinServer = servers.find(
				s => s.id === '00000000-0000-0000-0000-000000000000',
			);

			expect(builtinServer).toBeDefined();
			expect(builtinServer!.serverUrl).toBe(
				`${customBaseUrl}/mcp_servers/00000000-0000-0000-0000-000000000000`,
			);
		});

		it('should use default localhost URL when no baseUrl provided', () => {
			const servers = getDefaultMcpServers();
			const builtinServer = servers.find(
				s => s.id === '00000000-0000-0000-0000-000000000000',
			);

			expect(builtinServer).toBeDefined();
			expect(builtinServer!.serverUrl).toBe(
				'http://localhost:8080/mcp_servers/00000000-0000-0000-0000-000000000000',
			);
		});
	});

	describe('getBuiltinMcpServer', () => {
		it('should return the builtin MCP server', () => {
			const server = getBuiltinMcpServer();

			expect(server).toBeDefined();
			expect(server.id).toBe('00000000-0000-0000-0000-000000000000');
			expect(server.name).toBe('Built-in MCP Server');
			expect(server.description).toContain('Built-in MCP server');
			expect(server.enabled).toBe(true);
		});

		it('should use custom baseUrl when provided', () => {
			const customBaseUrl = 'https://custom.example.com';
			const server = getBuiltinMcpServer(customBaseUrl);

			expect(server.serverUrl).toBe(
				`${customBaseUrl}/mcp_servers/00000000-0000-0000-0000-000000000000`,
			);
		});

		it('should use default localhost URL when no baseUrl provided', () => {
			const server = getBuiltinMcpServer();

			expect(server.serverUrl).toBe(
				'http://localhost:8080/mcp_servers/00000000-0000-0000-0000-000000000000',
			);
		});

		it('should have all required properties', () => {
			const server = getBuiltinMcpServer();

			expect(server).toHaveProperty('id');
			expect(server).toHaveProperty('name');
			expect(server).toHaveProperty('description');
			expect(server).toHaveProperty('serverUrl');
			expect(server).toHaveProperty('enabled');

			expect(typeof server.id).toBe('string');
			expect(typeof server.name).toBe('string');
			expect(typeof server.description).toBe('string');
			expect(typeof server.serverUrl).toBe('string');
			expect(typeof server.enabled).toBe('boolean');
		});
	});
});
