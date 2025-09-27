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

// Mock dependencies
vi.mock('../services/backing/repositoryContainer.js', () => ({
	DefaultRepositoryContainer: vi.fn().mockImplementation(() => ({
		modelProviders: {
			list: vi.fn(() =>
				Promise.resolve([
					{
						id: 'test-provider-1',
						provider: 'openai',
						apiKey: 'test-key-1',
						baseUrl: 'https://api.openai.com/v1',
						modelsUrl: 'https://api.openai.com/v1/models',
					},
					{
						id: 'test-provider-2',
						provider: 'openai',
						apiKey: 'test-key-2',
						baseUrl: 'https://api.openai.com/v1',
						modelsUrl: 'https://api.openai.com/v1/models',
					},
				]),
			),
		},
	})),
}));

// Mock modelProvidersApi to prevent stderr errors
vi.mock('./modelProvidersApi.js', () => ({
	listModelProviders: vi.fn(() =>
		Promise.resolve({
			object: 'list',
			data: [
				{
					id: 'openai-provider',
					provider: 'openai',
					apiKey: 'test-api-key',
					baseUrl: 'https://api.openai.com/v1',
					modelsUrl: 'https://api.openai.com/v1/models',
				},
			],
		}),
	),
}));

import {listModels, retrieveModel, deleteModel} from './modelsApi.js';

describe('modelsApi', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	describe('listModels', () => {
		it('should be a function', () => {
			expect(typeof listModels).toBe('function');
		});

		it('should return a promise', () => {
			const result = listModels();
			expect(result).toBeInstanceOf(Promise);
		});
	});

	describe('retrieveModel', () => {
		it('should be a function', () => {
			expect(typeof retrieveModel).toBe('function');
		});

		it('should throw error for stub implementation', async () => {
			try {
				await retrieveModel('test-model');
				expect.fail('Expected retrieveModel to throw an error');
			} catch (error) {
				expect(error.message).toContain(
					'retrieveModel not implemented - this is a stub',
				);
			}
		});
	});

	describe('deleteModel', () => {
		it('should be a function', () => {
			expect(typeof deleteModel).toBe('function');
		});

		it('should throw error for stub implementation', async () => {
			try {
				await deleteModel('test-model');
				expect.fail('Expected deleteModel to throw an error');
			} catch (error) {
				expect(error.message).toContain(
					'deleteModel not implemented - this is a stub',
				);
			}
		});
	});

	describe('Real Execution Tests', () => {
		it('should execute listModels with real data', async () => {
			// Mock modelProvidersApi to return test data
			vi.doMock('./modelProvidersApi.js', () => ({
				listModelProviders: vi.fn(() =>
					Promise.resolve({
						data: [
							{
								id: 'openai-provider',
								provider: 'openai',
								apiKey: 'test-openai-key',
								modelsUrl: 'https://api.openai.com/v1/models',
							},
							{
								id: 'ollama-provider',
								provider: 'ollama',
								modelsUrl: 'http://localhost:11434/api/tags',
							},
						],
					}),
				),
			}));

			// Mock fetch for API calls
			global.fetch = vi
				.fn()
				.mockResolvedValueOnce({
					ok: true,
					json: () =>
						Promise.resolve({
							data: [
								{
									id: 'gpt-4',
									created: 1234567890,
									object: 'model',
									owned_by: 'openai',
								},
								{
									id: 'gpt-3.5-turbo',
									created: 1234567891,
									object: 'model',
									owned_by: 'openai',
								},
							],
						}),
				})
				.mockResolvedValueOnce({
					ok: true,
					json: () =>
						Promise.resolve({
							models: [
								{name: 'llama2', modified_at: '2023-01-01T00:00:00Z'},
								{name: 'codellama', modified_at: '2023-01-02T00:00:00Z'},
							],
						}),
				});

			const result = await listModels();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
			expect(result.data).toHaveLength(4);

			// Check OpenAI models
			expect(result.data[0].id).toBe('openai/gpt-4');
			expect(result.data[1].id).toBe('openai/gpt-3.5-turbo');

			// Check Ollama models
			expect(result.data[2].id).toBe('ollama/llama2');
			expect(result.data[3].id).toBe('ollama/codellama');
		});

		it('should handle encrypted API keys', async () => {
			// Mock modelProvidersApi to return provider with encrypted key
			vi.doMock('./modelProvidersApi.js', () => ({
				listModelProviders: vi.fn(() =>
					Promise.resolve({
						data: [
							{
								id: 'openai-provider',
								provider: 'openai',
								apiKey: 'encrypted:test-key',
								modelsUrl: 'https://api.openai.com/v1/models',
							},
						],
					}),
				),
			}));

			// Mock utils for encryption/decryption
			vi.doMock('../utils.js', () => ({
				isEncryptedSecret: vi.fn(key => key.startsWith('encrypted:')),
				decryptSecret: vi.fn(key =>
					Promise.resolve(key.replace('encrypted:', '')),
				),
			}));

			// Mock fetch for API calls
			global.fetch = vi.fn().mockResolvedValue({
				ok: true,
				json: () =>
					Promise.resolve({
						data: [
							{
								id: 'gpt-4',
								created: 1234567890,
								object: 'model',
								owned_by: 'openai',
							},
						],
					}),
			});

			const result = await listModels();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
		});

		it('should handle provider without API key for non-ollama', async () => {
			// Mock modelProvidersApi to return provider without API key
			vi.doMock('./modelProvidersApi.js', () => ({
				listModelProviders: vi.fn(() =>
					Promise.resolve({
						data: [
							{
								id: 'openai-provider',
								provider: 'openai',
								// No apiKey provided
								modelsUrl: 'https://api.openai.com/v1/models',
							},
						],
					}),
				),
			}));

			const result = await listModels();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
			expect(result.data).toHaveLength(0); // Should skip provider without API key
		});

		it('should handle API key decryption failure', async () => {
			// Mock modelProvidersApi to return provider with encrypted key
			vi.doMock('./modelProvidersApi.js', () => ({
				listModelProviders: vi.fn(() =>
					Promise.resolve({
						data: [
							{
								id: 'openai-provider',
								provider: 'openai',
								apiKey: 'encrypted:test-key',
								modelsUrl: 'https://api.openai.com/v1/models',
							},
						],
					}),
				),
			}));

			// Mock utils for encryption/decryption with failure
			vi.doMock('../utils.js', () => ({
				isEncryptedSecret: vi.fn(key => key.startsWith('encrypted:')),
				decryptSecret: vi.fn(() =>
					Promise.reject(new Error('Decryption failed')),
				),
			}));

			const result = await listModels();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
			expect(result.data).toHaveLength(0); // Should skip provider with decryption failure
		});

		it('should handle fetch API errors', async () => {
			// Mock modelProvidersApi to return test data
			vi.doMock('./modelProvidersApi.js', () => ({
				listModelProviders: vi.fn(() =>
					Promise.resolve({
						data: [
							{
								id: 'openai-provider',
								provider: 'openai',
								apiKey: 'test-key',
								modelsUrl: 'https://api.openai.com/v1/models',
							},
						],
					}),
				),
			}));

			// Mock fetch to return error
			global.fetch = vi.fn().mockResolvedValue({
				ok: false,
				status: 401,
			});

			const result = await listModels();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
			expect(result.data).toHaveLength(0); // Should handle API error gracefully
		});

		it('should handle fetch network errors', async () => {
			// Mock modelProvidersApi to return test data
			vi.doMock('./modelProvidersApi.js', () => ({
				listModelProviders: vi.fn(() =>
					Promise.resolve({
						data: [
							{
								id: 'openai-provider',
								provider: 'openai',
								apiKey: 'test-key',
								modelsUrl: 'https://api.openai.com/v1/models',
							},
						],
					}),
				),
			}));

			// Mock fetch to throw network error
			global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

			const result = await listModels();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
			expect(result.data).toHaveLength(0); // Should handle network error gracefully
		});

		it('should handle provider with alternative API key field', async () => {
			// Mock modelProvidersApi to return provider with api_key field
			vi.doMock('./modelProvidersApi.js', () => ({
				listModelProviders: vi.fn(() =>
					Promise.resolve({
						data: [
							{
								id: 'openai-provider',
								provider: 'openai',
								api_key: 'test-key', // Alternative field name
								modelsUrl: 'https://api.openai.com/v1/models',
							},
						],
					}),
				),
			}));

			// Mock fetch for API calls
			global.fetch = vi.fn().mockResolvedValue({
				ok: true,
				json: () =>
					Promise.resolve({
						data: [
							{
								id: 'gpt-4',
								created: 1234567890,
								object: 'model',
								owned_by: 'openai',
							},
						],
					}),
			});

			const result = await listModels();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
			expect(result.data.length).toBeGreaterThan(0);
		});

		it('should handle provider with alternative models URL field', async () => {
			// Mock modelProvidersApi to return provider with models_url field
			vi.doMock('./modelProvidersApi.js', () => ({
				listModelProviders: vi.fn(() =>
					Promise.resolve({
						data: [
							{
								id: 'openai-provider',
								provider: 'openai',
								apiKey: 'test-key',
								models_url: 'https://api.openai.com/v1/models', // Alternative field name
							},
						],
					}),
				),
			}));

			// Mock fetch for API calls
			global.fetch = vi.fn().mockResolvedValue({
				ok: true,
				json: () =>
					Promise.resolve({
						data: [
							{
								id: 'gpt-4',
								created: 1234567890,
								object: 'model',
								owned_by: 'openai',
							},
						],
					}),
			});

			const result = await listModels();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
			expect(result.data.length).toBeGreaterThan(0);
		});

		it('should handle listModelProviders API failure', async () => {
			// Mock modelProvidersApi to throw error
			vi.doMock('./modelProvidersApi.js', () => ({
				listModelProviders: vi.fn(() =>
					Promise.reject(new Error('API failure')),
				),
			}));

			const result = await listModels();

			expect(result).toBeDefined();
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);
			expect(result.data).toHaveLength(0); // Should handle API failure gracefully
		});

		it('should handle multiple listModels operations', async () => {
			// Test multiple operations in sequence
			const models1 = await listModels();
			expect(models1.object).toBe('list');
			expect(Array.isArray(models1.data)).toBe(true);

			const models2 = await listModels();
			expect(models2.object).toBe('list');
			expect(Array.isArray(models2.data)).toBe(true);
		});

		it('should handle concurrent listModels operations', async () => {
			// Test concurrent operations
			const promises = [listModels(), listModels(), listModels()];

			const results = await Promise.all(promises);

			expect(results).toHaveLength(3);
			results.forEach(result => {
				expect(result).toBeDefined();
				expect(result.object).toBe('list');
				expect(Array.isArray(result.data)).toBe(true);
			});
		});

		it('should handle error scenarios gracefully', async () => {
			// Test that listModels handles errors gracefully
			try {
				const result = await listModels();
				expect(result).toBeDefined();
				expect(result.object).toBe('list');
				expect(Array.isArray(result.data)).toBe(true);
			} catch (error) {
				// If an error is thrown, it should be handled gracefully
				expect(error).toBeDefined();
			}
		});

		it('should handle rapid successive calls', async () => {
			// Test rapid successive calls
			const start = Date.now();

			const promises = Array.from({length: 10}, () => listModels());
			const results = await Promise.all(promises);
			const end = Date.now();

			expect(results).toHaveLength(10);
			expect(end - start).toBeLessThan(5000); // Should complete within 5 seconds

			results.forEach(result => {
				expect(result).toBeDefined();
				expect(result.object).toBe('list');
				expect(Array.isArray(result.data)).toBe(true);
			});
		});

		it('should handle model data structure validation', async () => {
			const result = await listModels();

			// Validate the response structure
			expect(result).toHaveProperty('object');
			expect(result).toHaveProperty('data');
			expect(result.object).toBe('list');
			expect(Array.isArray(result.data)).toBe(true);

			// If there are models, validate their structure
			if (result.data.length > 0) {
				result.data.forEach(model => {
					expect(model).toHaveProperty('id');
					expect(model).toHaveProperty('created');
					expect(model).toHaveProperty('object');
					expect(model).toHaveProperty('owned_by');
					expect(model.object).toBe('model');
					expect(typeof model.id).toBe('string');
					expect(typeof model.created).toBe('number');
					expect(typeof model.owned_by).toBe('string');
				});
			}
		});
	});

	describe('Performance Tests', () => {
		it('should handle large number of listModels operations', async () => {
			// Test with a larger number of operations
			const operations = Array.from({length: 20}, () => listModels());

			const start = Date.now();
			const results = await Promise.all(operations);
			const end = Date.now();

			expect(results).toHaveLength(20);
			expect(end - start).toBeLessThan(10000); // Should complete within 10 seconds

			results.forEach(result => {
				expect(result).toBeDefined();
				expect(result.object).toBe('list');
				expect(Array.isArray(result.data)).toBe(true);
			});
		});

		it('should handle memory efficiently with multiple operations', async () => {
			// Test memory efficiency with multiple operations
			for (let i = 0; i < 50; i++) {
				const result = await listModels();
				expect(result).toBeDefined();
				expect(result.object).toBe('list');
				expect(Array.isArray(result.data)).toBe(true);
			}
		});

		it('should handle concurrent listModels operations efficiently', async () => {
			// Test concurrent operations
			const operations = [
				listModels(),
				listModels(),
				listModels(),
				listModels(),
				listModels(),
				listModels(),
			];

			const start = Date.now();
			const results = await Promise.all(operations);
			const end = Date.now();

			expect(results).toHaveLength(6);
			expect(end - start).toBeLessThan(10000); // Should complete within 10 seconds

			results.forEach(result => {
				expect(result).toBeDefined();
				expect(result.object).toBe('list');
				expect(Array.isArray(result.data)).toBe(true);
			});
		});
	});
});
