import {describe, it, expect, vi, beforeEach} from 'vitest';
import {ModelProviderService} from './modelProviderService.js';
import {ModelProvider} from '../api/modelProvidersApi.js';

describe('ModelProviderService', () => {
	let modelProviderService: ModelProviderService;
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

		modelProviderService = new ModelProviderService(mockRepository);
	});

	describe('listModelProviders', () => {
		it('should return list of model providers from repository', async () => {
			const mockProviders: ModelProvider[] = [
				{
					id: 'provider-1',
					provider: 'openai',
					baseUrl: 'https://api.openai.com',
					modelsUrl: 'https://api.openai.com/models',
				},
				{
					id: 'provider-2',
					provider: 'anthropic',
					baseUrl: 'https://api.anthropic.com',
					modelsUrl: 'https://api.anthropic.com/models',
				},
			];
			mockRepository.list.mockResolvedValue(mockProviders);

			const result = await modelProviderService.listModelProviders();

			expect(mockRepository.list).toHaveBeenCalled();
			expect(result).toEqual(mockProviders);
		});
	});

	describe('getModelProvider', () => {
		it('should return model provider from repository', async () => {
			const mockProvider: ModelProvider = {
				id: 'provider-1',
				provider: 'openai',
				baseUrl: 'https://api.openai.com',
				modelsUrl: 'https://api.openai.com/models',
			};
			mockRepository.load.mockResolvedValue(mockProvider);

			const result = await modelProviderService.getModelProvider('provider-1');

			expect(mockRepository.load).toHaveBeenCalledWith('provider-1');
			expect(result).toEqual(mockProvider);
		});

		it('should return null when provider not found', async () => {
			mockRepository.load.mockResolvedValue(null);

			const result = await modelProviderService.getModelProvider(
				'non-existent-provider',
			);

			expect(result).toBeNull();
		});
	});

	describe('isModelProviderAvailable', () => {
		it('should return true when provider exists', async () => {
			mockRepository.exists.mockResolvedValue(true);

			const result = await modelProviderService.isModelProviderAvailable(
				'provider-1',
			);

			expect(mockRepository.exists).toHaveBeenCalledWith('provider-1');
			expect(result).toBe(true);
		});

		it('should return false when provider does not exist', async () => {
			mockRepository.exists.mockResolvedValue(false);

			const result = await modelProviderService.isModelProviderAvailable(
				'non-existent-provider',
			);

			expect(result).toBe(false);
		});
	});

	describe('saveModelProvider', () => {
		it('should save model provider to repository', async () => {
			const mockProvider: ModelProvider = {
				id: 'provider-1',
				provider: 'openai',
				baseUrl: 'https://api.openai.com',
				modelsUrl: 'https://api.openai.com/models',
			};

			await modelProviderService.saveModelProvider(mockProvider);

			expect(mockRepository.save).toHaveBeenCalledWith(mockProvider);
		});
	});

	describe('deleteModelProvider', () => {
		it('should delete model provider from repository', async () => {
			await modelProviderService.deleteModelProvider('provider-1');

			expect(mockRepository.delete).toHaveBeenCalledWith('provider-1');
		});
	});
});
