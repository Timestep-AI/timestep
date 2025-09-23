import {Repository} from './repository.js';
import {ModelProvider} from '../../api/modelProvidersApi.js';
import {JsonlRepository} from './jsonlRepository.js';
import {getTimestepPaths} from '../../utils.js';
import {getDefaultModelProviders} from '../../config/defaultModelProviders.js';
import * as fs from 'node:fs';
import * as path from 'node:path';

// Get timestep configuration paths
const timestepPaths = getTimestepPaths();

/**
 * JSONL file-based implementation of ModelProviderRepository.
 * Stores model providers as JSON objects in a .jsonl file, one provider per line.
 */
export class JsonlModelProviderRepository
	extends JsonlRepository<ModelProvider, string>
	implements Repository<ModelProvider, string>
{
	constructor() {
		super(timestepPaths.modelProviders);
	}

	protected serialize(provider: ModelProvider): string {
		return JSON.stringify(provider);
	}

	protected deserialize(line: string): ModelProvider {
		return JSON.parse(line) as ModelProvider;
	}

	protected getId(provider: ModelProvider): string {
		return provider.id;
	}

	override async list(): Promise<ModelProvider[]> {
		try {
			const providers = await super.list();
			if (providers.length > 0) {
				// Mask secrets at rest if any plaintext values exist; migrate to encrypted on next save elsewhere
				// console.log(
				// 	`🤖 Loaded ${providers.length} model providers from ${this.filePath}`,
				// );
				return providers;
			}
		} catch (error) {
			console.warn(
				`Failed to read model providers configuration from '${this.filePath}': ${error}. Using default configuration.`,
			);
		}

		// If no providers found or error reading, try to create default configuration
		// In restricted environments (like Supabase Edge Functions), this will fail gracefully
		const defaultProviders = getDefaultModelProviders();
		try {
			await this.createDefaultModelProvidersFile();
			console.log(
				`🤖 Created default model providers configuration with ${defaultProviders.length} providers`,
			);
		} catch (error) {
			console.warn(
				`Unable to create default configuration file (restricted environment): ${error}`,
			);
			console.log(
				`🤖 Using in-memory default model providers configuration with ${defaultProviders.length} providers`,
			);
		}

		return defaultProviders;
	}

	override async save(provider: ModelProvider): Promise<void> {
		const toSave: ModelProvider = {...provider};
		if ((toSave as any).apiKey) {
			const {isEncryptedSecret, encryptSecret} = await import('../../utils.js');
			const value = (toSave as any).apiKey as string;
			if (!isEncryptedSecret(value)) {
				try {
					(toSave as any).apiKey = await encryptSecret(value);
				} catch (error) {
					console.warn('Failed to encrypt model provider apiKey:', error);
				}
			}
		}
		return super.save(toSave);
	}

	/**
	 * Create the model providers configuration file with default providers
	 */
	private async createDefaultModelProvidersFile(): Promise<void> {
		try {
			// Ensure the directory exists
			const dir = path.dirname(this.filePath);
			if (!fs.existsSync(dir)) {
				fs.mkdirSync(dir, {recursive: true});
			}

			// Write the default model providers as JSONL
			const defaultProviders = getDefaultModelProviders();
			const lines = defaultProviders.map(provider => this.serialize(provider));
			await this.writeLines(lines);
			console.log(
				`🤖 Created default model providers configuration at: ${this.filePath}`,
			);
		} catch (error) {
			console.warn(
				`Failed to create default model providers configuration at '${this.filePath}': ${error}`,
			);
		}
	}
}
