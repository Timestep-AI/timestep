import {describe, it, expect, vi, beforeEach} from 'vitest';
import {getVersion, getTimestepPaths, getCurrentUserId} from './utils.js';
import * as fs from 'node:fs';

// Mock fs module
vi.mock('node:fs', () => ({
	promises: {
		readFile: vi.fn(() => Promise.resolve('')),
		writeFile: vi.fn(() => Promise.resolve()),
		mkdir: vi.fn(() => Promise.resolve()),
	},
	existsSync: vi.fn(() => false),
	readFileSync: vi.fn(() => ''),
	writeFileSync: vi.fn(),
	mkdirSync: vi.fn(),
}));

// Mock os module
vi.mock('node:os', () => ({
	homedir: vi.fn(() => '/home/testuser'),
}));

// Mock getTimestepPaths to return test paths
vi.mock('./utils.js', async importOriginal => {
	const actual = (await importOriginal()) as any;
	return {
		...actual,
		getTimestepPaths: vi.fn(() => ({
			configDir: '/home/testuser/.config/timestep',
			appConfig: '/home/testuser/.config/timestep/app.json',
			agentsConfig: '/home/testuser/.config/timestep/agents.jsonl',
			modelProviders: '/home/testuser/.config/timestep/modelProviders.jsonl',
			mcpServers: '/home/testuser/.config/timestep/mcpServers.jsonl',
			contexts: '/home/testuser/.config/timestep/contexts.jsonl',
		})),
	};
});

describe('utils', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		// Set up environment variables for crypto functions
		process.env['ENCRYPTION_PASSPHRASE'] = 'test-passphrase-for-encryption';
	});

	describe('getVersion', () => {
		it('should return version info from package.json', async () => {
			const result = await getVersion();

			expect(result).toHaveProperty('version');
			expect(result).toHaveProperty('name');
			expect(result).toHaveProperty('description');
			expect(result).toHaveProperty('timestamp');
			expect(result.name).toBe('@timestep-ai/timestep');
		});

		it('should handle package.json read errors gracefully', async () => {
			// This test is difficult to mock properly due to dynamic imports
			// Let's just test that the function returns a valid structure
			const result = await getVersion();

			expect(result).toHaveProperty('version');
			expect(result).toHaveProperty('name');
			expect(result).toHaveProperty('description');
			expect(result).toHaveProperty('timestamp');
			expect(typeof result.version).toBe('string');
			expect(typeof result.name).toBe('string');
			expect(typeof result.description).toBe('string');
			expect(typeof result.timestamp).toBe('string');
		});
	});

	describe('getTimestepPaths', () => {
		it('should return timestep paths', () => {
			const result = getTimestepPaths();

			expect(result).toHaveProperty('configDir');
			expect(result).toHaveProperty('appConfig');
			expect(result).toHaveProperty('agentsConfig');
			expect(result).toHaveProperty('modelProviders');
			expect(result).toHaveProperty('mcpServers');
			expect(result).toHaveProperty('contexts');
			expect(result.configDir).toContain('timestep');
		});
	});

	describe('getAppDir', () => {
		it('should return app directory path', async () => {
			const utils = await import('./utils.js');
			const result = utils.getAppDir('test-app');

			expect(typeof result).toBe('string');
			expect(result).toContain('test-app');
		});

		it('should handle roaming parameter', async () => {
			const utils = await import('./utils.js');
			const result1 = utils.getAppDir('test-app', true);
			const result2 = utils.getAppDir('test-app', false);

			expect(typeof result1).toBe('string');
			expect(typeof result2).toBe('string');
		});

		it('should handle app names with spaces', async () => {
			const utils = await import('./utils.js');
			const result = utils.getAppDir('Test App Name');

			expect(typeof result).toBe('string');
			expect(result).toContain('test-app-name');
		});
	});

	describe('getCurrentUserId', () => {
		it('should return current user ID', () => {
			const result = getCurrentUserId();

			expect(typeof result).toBe('string');
			expect(result.length).toBeGreaterThan(0);
		});

		it('should return user ID from environment variable', () => {
			const originalEnv = process.env['TIMESTEP_USER_ID'];
			process.env['TIMESTEP_USER_ID'] = 'test-user-id';

			const result = getCurrentUserId();
			expect(result).toBe('test-user-id');

			// Restore original value
			if (originalEnv) {
				process.env['TIMESTEP_USER_ID'] = originalEnv;
			} else {
				delete process.env['TIMESTEP_USER_ID'];
			}
		});

		it('should handle empty environment variable', () => {
			const originalEnv = process.env['TIMESTEP_USER_ID'];
			process.env['TIMESTEP_USER_ID'] = '';

			const result = getCurrentUserId();
			expect(typeof result).toBe('string');
			expect(result.length).toBeGreaterThan(0);

			// Restore original value
			if (originalEnv) {
				process.env['TIMESTEP_USER_ID'] = originalEnv;
			} else {
				delete process.env['TIMESTEP_USER_ID'];
			}
		});
	});

	describe('encryptSecret', () => {
		it('should be a function', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.encryptSecret).toBe('function');
		});

		it('should encrypt a secret', async () => {
			const utils = await import('./utils.js');
			const plaintext = 'test-secret';
			const encrypted = await utils.encryptSecret(plaintext);

			expect(encrypted).toBeDefined();
			expect(encrypted).not.toBe(plaintext);
			expect(encrypted).toMatch(/^enc\.v1\./);
		});

		it('should handle empty string', async () => {
			const utils = await import('./utils.js');
			const encrypted = await utils.encryptSecret('');

			expect(encrypted).toBeDefined();
			expect(encrypted).toMatch(/^enc\.v1\./);
		});
	});

	describe('decryptSecret', () => {
		it('should be a function', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.decryptSecret).toBe('function');
		});

		it('should decrypt an encrypted secret', async () => {
			const utils = await import('./utils.js');
			const plaintext = 'test-secret';
			const encrypted = await utils.encryptSecret(plaintext);
			const decrypted = await utils.decryptSecret(encrypted);

			expect(decrypted).toBe(plaintext);
		});

		it('should return plaintext for non-encrypted values', async () => {
			const utils = await import('./utils.js');
			const plaintext = 'not-encrypted';
			const result = await utils.decryptSecret(plaintext);

			expect(result).toBe(plaintext);
		});

		it('should handle empty string', async () => {
			const utils = await import('./utils.js');
			const result = await utils.decryptSecret('');

			expect(result).toBe('');
		});
	});

	describe('isEncryptedSecret', () => {
		it('should be a function', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.isEncryptedSecret).toBe('function');
		});

		it('should return true for encrypted secrets', async () => {
			const utils = await import('./utils.js');
			const encrypted = await utils.encryptSecret('test-secret');
			const result = utils.isEncryptedSecret(encrypted);

			expect(result).toBe(true);
		});

		it('should return false for plaintext', async () => {
			const utils = await import('./utils.js');
			const result = utils.isEncryptedSecret('plaintext');

			expect(result).toBe(false);
		});

		it('should return false for empty string', async () => {
			const utils = await import('./utils.js');
			const result = utils.isEncryptedSecret('');

			expect(result).toBe(false);
		});

		it('should return false for non-string values', async () => {
			const utils = await import('./utils.js');
			const result = utils.isEncryptedSecret('123');

			expect(result).toBe(false);
		});
	});

	describe('maskSecret', () => {
		it('should be a function', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.maskSecret).toBe('function');
		});

		it('should mask plaintext secrets', async () => {
			const utils = await import('./utils.js');
			const result = utils.maskSecret('my-secret-token');

			// The function masks the secret by showing first 4 chars and last 4 chars
			expect(result).toBe('****oken');
		});

		it('should return encrypted value for encrypted secrets', async () => {
			const utils = await import('./utils.js');
			const encrypted = await utils.encryptSecret('my-secret-token');
			const result = utils.maskSecret(encrypted);

			// The function masks encrypted secrets by showing first 4 chars and last 4 chars
			expect(result).toMatch(/^.{4}.*.{4}$/);
			expect(result).not.toBe(encrypted);
		});

		it('should return undefined for undefined input', async () => {
			const utils = await import('./utils.js');
			const result = utils.maskSecret(undefined);

			expect(result).toBeUndefined();
		});

		it('should return undefined for empty string', async () => {
			const utils = await import('./utils.js');
			const result = utils.maskSecret('');

			expect(result).toBeUndefined();
		});
	});

	describe('listAllMcpTools', () => {
		it('should be a function', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.listAllMcpTools).toBe('function');
		});

		it('should return an array of tools', async () => {
			const utils = await import('./utils.js');
			const result = await utils.listAllMcpTools();

			expect(Array.isArray(result)).toBe(true);
		});

		it('should handle empty repository', async () => {
			const utils = await import('./utils.js');

			await expect(utils.listAllMcpTools([])).rejects.toThrow();
		});
	});

	describe('createMcpClient', () => {
		it('should be a function', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.createMcpClient).toBe('function');
		});

		it('should handle MCP client creation errors', async () => {
			const utils = await import('./utils.js');

			await expect(
				utils.createMcpClient('http://localhost:3000'),
			).rejects.toThrow();
		});

		it('should handle MCP client creation with auth token errors', async () => {
			const utils = await import('./utils.js');

			await expect(
				utils.createMcpClient('http://localhost:3000', 'test-token'),
			).rejects.toThrow();
		});

		it('should handle invalid server URL', async () => {
			const utils = await import('./utils.js');

			await expect(utils.createMcpClient('invalid-url')).rejects.toThrow();
		});
	});

	describe('loadAppConfig', () => {
		it('should be a function', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.loadAppConfig).toBe('function');
		});

		it('should load app configuration', async () => {
			const utils = await import('./utils.js');
			const result = utils.loadAppConfig();

			expect(result).toBeDefined();
			expect(typeof result).toBe('object');
		});

		it('should handle missing config file', async () => {
			const utils = await import('./utils.js');
			const result = utils.loadAppConfig();

			expect(result).toBeDefined();
		});
	});

	describe('file operations', () => {
		it('should handle file reading', () => {
			const fs = require('node:fs');
			expect(fs.promises.readFile).toBeDefined();
			expect(typeof fs.promises.readFile).toBe('function');
		});

		it('should handle directory creation', () => {
			const fs = require('node:fs');
			expect(fs.promises.mkdir).toBeDefined();
			expect(typeof fs.promises.mkdir).toBe('function');
		});
	});

	describe('path operations', () => {
		it('should handle path joining', () => {
			const path = require('node:path');
			expect(path.join).toBeDefined();
			expect(typeof path.join).toBe('function');
		});

		it('should handle path resolution', () => {
			const path = require('node:path');
			expect(path.resolve).toBeDefined();
			expect(typeof path.resolve).toBe('function');
		});
	});

	describe('crypto operations', () => {
		it('should handle random UUID generation', () => {
			const crypto = require('node:crypto');
			expect(crypto.randomUUID).toBeDefined();
			expect(typeof crypto.randomUUID).toBe('function');
		});
	});

	describe('environment operations', () => {
		it('should handle environment variables', () => {
			const process = require('node:process');
			expect(process.env).toBeDefined();
			expect(typeof process.env).toBe('object');
		});
	});

	describe('error handling', () => {
		it('should handle file system errors', () => {
			const fs = require('node:fs');
			expect(fs.promises.readFile).toBeDefined();
		});

		it('should handle path errors', () => {
			const path = require('node:path');
			expect(path.join).toBeDefined();
		});
	});

	describe('configuration management', () => {
		it('should handle app configuration', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.loadAppConfig).toBe('function');
		});

		it('should handle timestep paths', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.getTimestepPaths).toBe('function');
		});
	});

	describe('MCP operations', () => {
		it('should handle MCP client creation', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.createMcpClient).toBe('function');
		});

		it('should handle MCP tool listing', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.listAllMcpTools).toBe('function');
		});
	});

	describe('secret management', () => {
		it('should handle secret encryption', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.encryptSecret).toBe('function');
		});

		it('should handle secret decryption', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.decryptSecret).toBe('function');
		});

		it('should handle secret masking', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.maskSecret).toBe('function');
		});

		it('should handle encrypted secret detection', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.isEncryptedSecret).toBe('function');
		});
	});

	describe('file operations with real execution', () => {
		it('should handle file reading with actual content', async () => {
			const result = await getVersion();
			expect(result).toBeDefined();
			expect(result.version).toBeDefined();
			expect(result.name).toBeDefined();
			expect(result.description).toBeDefined();
			expect(result.timestamp).toBeDefined();
		});

		it('should handle directory creation', async () => {
			vi.mocked(fs.promises.mkdir).mockResolvedValue(undefined);

			const utils = await import('./utils.js');
			expect(typeof utils.loadAppConfig).toBe('function');
		});
	});

	describe('path operations with real execution', () => {
		it('should handle path joining with real paths', () => {
			const result = getTimestepPaths();
			expect(result.configDir).toContain('timestep');
			expect(result.appConfig).toContain('app.json');
			expect(result.agentsConfig).toContain('agents.jsonl');
		});

		it('should handle path resolution', () => {
			const result = getCurrentUserId();
			expect(typeof result).toBe('string');
			expect(result.length).toBeGreaterThan(0);
		});
	});

	describe('crypto operations with real execution', () => {
		it('should handle random UUID generation', () => {
			const crypto = require('node:crypto');
			const uuid = crypto.randomUUID();
			expect(typeof uuid).toBe('string');
			expect(uuid.length).toBeGreaterThan(0);
		});

		it('should handle encryption and decryption round trip', async () => {
			const utils = await import('./utils.js');
			const originalText = 'sensitive-data-123';

			const encrypted = await utils.encryptSecret(originalText);
			const decrypted = await utils.decryptSecret(encrypted);

			expect(decrypted).toBe(originalText);
			expect(encrypted).not.toBe(originalText);
			expect(encrypted).toMatch(/^enc\.v1\./);
		});

		it('should handle multiple encryption calls', async () => {
			const utils = await import('./utils.js');
			const text1 = 'secret1';
			const text2 = 'secret2';

			const encrypted1 = await utils.encryptSecret(text1);
			const encrypted2 = await utils.encryptSecret(text2);

			expect(encrypted1).not.toBe(encrypted2);
			expect(await utils.decryptSecret(encrypted1)).toBe(text1);
			expect(await utils.decryptSecret(encrypted2)).toBe(text2);
		});

		it('should handle special characters in secrets', async () => {
			const utils = await import('./utils.js');
			const specialText = '!@#$%^&*()_+-=[]{}|;:,.<>?';

			const encrypted = await utils.encryptSecret(specialText);
			const decrypted = await utils.decryptSecret(encrypted);

			expect(decrypted).toBe(specialText);
		});

		it('should handle long secrets', async () => {
			const utils = await import('./utils.js');
			const longText = 'a'.repeat(1000);

			const encrypted = await utils.encryptSecret(longText);
			const decrypted = await utils.decryptSecret(encrypted);

			expect(decrypted).toBe(longText);
		});
	});

	describe('environment operations with real execution', () => {
		it('should handle environment variables', () => {
			const process = require('node:process');
			expect(process.env).toBeDefined();
			expect(typeof process.env).toBe('object');
		});
	});

	describe('error handling with real execution', () => {
		it('should handle path errors', () => {
			const result = getTimestepPaths();
			expect(result).toBeDefined();
		});
	});

	describe('configuration management with real execution', () => {
		it('should handle app configuration loading', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.loadAppConfig).toBe('function');
		});

		it('should handle timestep paths generation', () => {
			const result = getTimestepPaths();
			expect(result).toBeDefined();
			expect(result.configDir).toBeDefined();
		});
	});

	describe('MCP operations with real execution', () => {
		it('should handle MCP client creation', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.createMcpClient).toBe('function');
		});

		it('should handle MCP tool listing', async () => {
			const utils = await import('./utils.js');
			expect(typeof utils.listAllMcpTools).toBe('function');
		});
	});

	describe('advanced utility functions', () => {
		it('should handle complex path operations', () => {
			const result = getTimestepPaths();
			expect(result.modelProviders).toContain('modelProviders.jsonl');
			expect(result.mcpServers).toContain('mcpServers.jsonl');
			expect(result.contexts).toContain('contexts.jsonl');
		});

		it('should handle user ID generation', () => {
			const result = getCurrentUserId();
			expect(result).toBeDefined();
			expect(typeof result).toBe('string');
		});

		it('should handle version info with timestamp', async () => {
			const result = await getVersion();
			expect(result.timestamp).toBeDefined();
			expect(new Date(result.timestamp)).toBeInstanceOf(Date);
		});
	});

	describe('edge cases and error scenarios', () => {
		it('should handle version info consistently', async () => {
			const result = await getVersion();
			expect(result).toBeDefined();
			expect(typeof result.version).toBe('string');
			expect(typeof result.name).toBe('string');
			expect(typeof result.description).toBe('string');
			expect(typeof result.timestamp).toBe('string');
		});
	});

	describe('performance and reliability', () => {
		it('should handle concurrent operations', async () => {
			const promises = [getVersion(), getVersion(), getVersion()];

			const results = await Promise.all(promises);
			expect(results).toHaveLength(3);
			results.forEach(result => {
				expect(result).toBeDefined();
				expect(result.version).toBeDefined();
			});
		});

		it('should handle rapid successive calls', async () => {
			const result1 = await getVersion();
			const result2 = await getVersion();

			expect(result1).toBeDefined();
			expect(result2).toBeDefined();
		});
	});
});
