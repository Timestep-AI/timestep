import {Ollama} from 'ollama';
import {OllamaModel} from './backing/models.js';
import {
	Model,
	ModelProvider,
	OpenAIChatCompletionsModel,
	OpenAIResponsesModel,
} from '@openai/agents';
import OpenAI from 'openai';
import {getTimestepPaths} from '../utils.js';
import {
	RepositoryContainer,
	DefaultRepositoryContainer,
} from './backing/repositoryContainer.js';

type ProviderConfig = {
	id?: string;
	provider?: string;
	name?: string;
	api_key?: string;
	base_url?: string;
	models_url?: string;
};

async function loadModelProviders(
	repositories?: RepositoryContainer,
): Promise<Record<string, ProviderConfig>> {
	try {
		const {listModelProviders} = await import('../api/modelProvidersApi.js');
		const response = await listModelProviders(repositories);
		const MODEL_PROVIDERS: Record<string, ProviderConfig> = {};

		for (const provider of response.data) {
			const key = provider.provider;
			if (!key) {
				console.warn(
					'Skipping model provider entry without a provider field:',
					provider,
				);
				continue;
			}
			MODEL_PROVIDERS[key] = provider;
		}

		return MODEL_PROVIDERS;
	} catch (error) {
		console.warn(
			`Failed to load model providers: ${error}. Using empty configuration.`,
		);
		return {};
	}
}

export class TimestepAIModelProvider implements ModelProvider {
	private modelProviders: Record<string, ProviderConfig> | null = null;
	private loadingPromise: Promise<Record<string, ProviderConfig>> | null = null;
	private repositories: RepositoryContainer;

	constructor(repositories?: RepositoryContainer) {
		this.repositories = repositories || new DefaultRepositoryContainer();
	}

	private async ensureModelProvidersLoaded(): Promise<
		Record<string, ProviderConfig>
	> {
		if (this.modelProviders !== null) {
			return this.modelProviders;
		}

		if (this.loadingPromise === null) {
			this.loadingPromise = loadModelProviders(this.repositories).then(
				providers => {
					this.modelProviders = providers;
					// console.log(
					// 	`Model providers loaded: ${JSON.stringify(this.modelProviders)}`,
					// );
					return providers;
				},
			);
		}

		return this.loadingPromise;
	}

	async getModel(_modelName?: string): Promise<Model> {
		const parts = _modelName?.split('/') || [];
		const modelProvider = parts.length > 1 ? parts[0] : 'openai';
		const modelId = parts.length > 1 ? parts.slice(1).join('/') : _modelName;

		if (!modelProvider) {
			throw new Error('Model provider not specified');
		}

		if (!modelId) {
			throw new Error('Model ID not specified');
		}

		// Load model providers
		const modelProviders = await this.ensureModelProvidersLoaded();

		// Check if any providers are configured
		if (Object.keys(modelProviders).length === 0) {
			const timestepPaths = getTimestepPaths();
			throw new Error(
				`No model providers configured. Please set up model providers configuration file at: ${timestepPaths.modelProviders}`,
			);
		}

		if (modelProvider === 'ollama') {
			const ollamaConfig = modelProviders[modelProvider];
			if (!ollamaConfig) {
				throw new Error(
					`Model provider configuration for '${modelProvider}' not found. Available providers: ${Object.keys(
						modelProviders,
					).join(', ')}`,
				);
			}
			if (!ollamaConfig.base_url) {
				throw new Error(
					`Missing 'base_url' for provider '${modelProvider}' in configuration`,
				);
			}
			const ollamaClient = new Ollama({
				host: ollamaConfig.base_url,
				headers: {Authorization: `Bearer ${ollamaConfig.api_key}`},
			});

			return new OllamaModel(ollamaClient, modelId);
		} else if (modelProvider === 'openai') {
			const openaiConfig = modelProviders[modelProvider];
			if (!openaiConfig) {
				throw new Error(
					`Model provider configuration for '${modelProvider}' not found. Available providers: ${Object.keys(
						modelProviders,
					).join(', ')}`,
				);
			}
			if (!openaiConfig.api_key) {
				throw new Error(
					`Missing 'api_key' for provider '${modelProvider}' in configuration`,
				);
			}
			const openaiClient = new OpenAI({
				apiKey: openaiConfig.api_key,
				baseURL: openaiConfig.base_url,
			}); // TODO: Use AsyncOpenAI?

			// Narrow via any to avoid duplicative OpenAI type private field conflict
			return new OpenAIResponsesModel(openaiClient as any, modelId);
		} else {
			const config = modelProviders[modelProvider];
			if (!config) {
				throw new Error(
					`Model provider configuration for '${modelProvider}' not found. Available providers: ${Object.keys(
						modelProviders,
					).join(', ')}`,
				);
			}
			if (!config.api_key) {
				throw new Error(
					`Missing 'api_key' for provider '${modelProvider}' in configuration`,
				);
			}
			const client = new OpenAI({
				apiKey: config.api_key,
				baseURL: config.base_url,
			}); // TODO: Use AsyncOpenAI?

			// Narrow via any to avoid duplicative OpenAI type private field conflict
			return new OpenAIChatCompletionsModel(client as any, modelId);
		}
	}
}
