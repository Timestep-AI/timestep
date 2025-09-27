import {describe, it, expect, vi, beforeEach} from 'vitest';
import {listModelProviders, getModelProvider} from './modelProvidersApi.js';

// Mock the ModelProviderService
const mockModelProviderService = {
	listModelProviders: vi.fn(),
	getModelProvider: vi.fn(),
};

vi.mock('../services/modelProviderService.js', () => ({
	ModelProviderService: vi
		.fn()
		.mockImplementation(() => mockModelProviderService),
}));

// Mock the repository container
vi.mock('../services/backing/repositoryContainer.js', () => ({
	DefaultRepositoryContainer: vi.fn().mockImplementation(() => ({
		modelProviderRepository: {
			list: vi.fn(),
			load: vi.fn(),
		},
	})),
}));

describe('modelProvidersApi', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	describe('listModelProviders', () => {
		it('should be a function', () => {
			expect(typeof listModelProviders).toBe('function');
		});

		it('should return a promise', () => {
			const result = listModelProviders();
			expect(result).toBeInstanceOf(Promise);
		});

		it('should handle successful model provider listing', async () => {
			mockModelProviderService.listModelProviders.mockResolvedValue([
				{id: 'provider-1', name: 'Test Provider 1', type: 'openai'},
				{id: 'provider-2', name: 'Test Provider 2', type: 'ollama'},
			]);

			const result = await listModelProviders();

			expect(result).toEqual({
				object: 'list',
				data: [
					{id: 'provider-1', name: 'Test Provider 1', type: 'openai'},
					{id: 'provider-2', name: 'Test Provider 2', type: 'ollama'},
				],
			});
		});

		it('should handle error in model provider listing', async () => {
			mockModelProviderService.listModelProviders.mockRejectedValue(
				new Error('Database error'),
			);

			await expect(listModelProviders()).rejects.toThrow(
				'Failed to list model providers: Error: Database error',
			);
		});
	});

	describe('getModelProvider', () => {
		it('should be a function', () => {
			expect(typeof getModelProvider).toBe('function');
		});

		it('should return a promise', () => {
			const result = getModelProvider('test-id');
			expect(result).toBeInstanceOf(Promise);
		});

		it('should handle successful model provider retrieval', async () => {
			mockModelProviderService.getModelProvider.mockResolvedValue({
				id: 'provider-1',
				name: 'Test Provider 1',
				type: 'openai',
			});

			const result = await getModelProvider('provider-1');

			expect(result).toEqual({
				id: 'provider-1',
				name: 'Test Provider 1',
				type: 'openai',
			});
		});

		it('should handle model provider not found', async () => {
			mockModelProviderService.getModelProvider.mockResolvedValue(null);

			const result = await getModelProvider('non-existent-provider');

			expect(result).toBeNull();
		});

		it('should handle error in model provider retrieval', async () => {
			mockModelProviderService.getModelProvider.mockRejectedValue(
				new Error('Database error'),
			);

			await expect(getModelProvider('provider-1')).rejects.toThrow(
				'Failed to get model provider: Error: Database error',
			);
		});
	});
});
