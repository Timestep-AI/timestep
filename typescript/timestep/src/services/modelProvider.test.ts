import {describe, it, expect, vi, beforeEach} from 'vitest';
import {TimestepAIModelProvider} from './modelProvider.js';

// Mock dependencies
vi.mock('ollama', () => ({
	Ollama: vi.fn().mockImplementation(() => ({
		generate: vi.fn(),
		chat: vi.fn(),
	})),
}));

vi.mock('./backing/models.js', () => ({
	OllamaModel: vi.fn().mockImplementation(() => ({
		generate: vi.fn(),
		generateStream: vi.fn(),
	})),
}));

vi.mock('@openai/agents', () => ({
	ModelProvider: vi.fn(),
	OpenAIChatCompletionsModel: vi.fn(),
	OpenAIResponsesModel: vi.fn(),
}));

vi.mock('openai', () => ({
	default: vi.fn().mockImplementation(() => ({
		chat: {
			completions: {
				create: vi.fn(),
			},
		},
	})),
}));

vi.mock('../utils.js', () => ({
	getTimestepPaths: vi.fn(() => ({
		configDir: '/tmp/test-config',
		appConfig: '/tmp/test-config/app.json',
		agentsConfig: '/tmp/test-config/agents.jsonl',
		modelProviders: '/tmp/test-config/modelProviders.jsonl',
		mcpServers: '/tmp/test-config/mcpServers.jsonl',
		contexts: '/tmp/test-config/contexts.jsonl',
	})),
}));

vi.mock('./backing/repositoryContainer.js', () => ({
	DefaultRepositoryContainer: vi.fn().mockImplementation(() => ({
		modelProviders: {
			list: vi.fn(),
		},
	})),
}));

vi.mock('../api/modelProvidersApi.js', () => ({
	listModelProviders: vi.fn(() =>
		Promise.resolve({
			data: [
				{
					id: 'test-provider-1',
					provider: 'openai',
					base_url: 'https://api.openai.com/v1',
					models_url: 'https://api.openai.com/v1/models',
					api_key: 'test-openai-key',
				},
				{
					id: 'test-provider-2',
					provider: 'ollama',
					base_url: 'http://localhost:11434',
					models_url: 'http://localhost:11434/api/tags',
					api_key: null,
				},
			],
		}),
	),
}));

describe('modelProvider', () => {
	let modelProvider: TimestepAIModelProvider;
	let mockRepositoryContainer: any;

	beforeEach(() => {
		vi.clearAllMocks();

		mockRepositoryContainer = {
			modelProviders: {
				list: vi.fn(),
			},
		};

		modelProvider = new TimestepAIModelProvider(mockRepositoryContainer);
	});

	describe('TimestepAIModelProvider', () => {
		it('should be importable', () => {
			expect(TimestepAIModelProvider).toBeDefined();
		});

		it('should create an instance', () => {
			expect(modelProvider).toBeInstanceOf(TimestepAIModelProvider);
		});

		it('should create an instance with default repository container', () => {
			const defaultProvider = new TimestepAIModelProvider();
			expect(defaultProvider).toBeInstanceOf(TimestepAIModelProvider);
		});

		it('should have required methods', () => {
			expect(typeof modelProvider.getModel).toBe('function');
		});

		it('should handle getModel for OpenAI provider', async () => {
			// Test that the method exists and can be called
			expect(typeof modelProvider.getModel).toBe('function');

			// The actual implementation will be tested in integration tests
			// For now, just verify the method exists
		});

		it('should handle getModel for Ollama provider', async () => {
			// Test that the method exists and can be called
			expect(typeof modelProvider.getModel).toBe('function');

			// The actual implementation will be tested in integration tests
			// For now, just verify the method exists
		});

		it('should handle unknown provider', async () => {
			// Test that the method exists and can be called
			expect(typeof modelProvider.getModel).toBe('function');

			// The actual implementation will be tested in integration tests
			// For now, just verify the method exists
		});

		it('should handle missing provider configuration', async () => {
			// Mock empty model providers
			const {listModelProviders} = await import('../api/modelProvidersApi.js');
			vi.mocked(listModelProviders).mockResolvedValueOnce({
				object: 'list',
				data: [],
			});

			const newProvider = new TimestepAIModelProvider();

			await expect(newProvider.getModel('gpt-4')).rejects.toThrow();
		});

		it('should handle API errors gracefully', async () => {
			const {listModelProviders} = await import('../api/modelProvidersApi.js');
			vi.mocked(listModelProviders).mockRejectedValueOnce(
				new Error('API Error'),
			);

			const newProvider = new TimestepAIModelProvider();

			await expect(newProvider.getModel('gpt-4')).rejects.toThrow();
		});
	});
});
