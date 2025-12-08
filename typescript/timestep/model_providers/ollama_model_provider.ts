import { Model, ModelProvider } from '@openai/agents-core';
import { OllamaModel } from '../models/ollama_model.ts';

/**
 * Options for OllamaModelProvider.
 */
export type OllamaModelProviderOptions = {
  apiKey?: string;
  baseURL?: string;
  ollamaClient?: any; // Will be dynamically imported
};

/**
 * The provider of Ollama's models
 */
export class OllamaModelProvider implements ModelProvider {
  #client?: any; // Will be dynamically imported
  #options: OllamaModelProviderOptions;
  #currentModelName?: string;

  constructor(options: OllamaModelProviderOptions = {}) {
    this.#options = options;
    if (this.#options.ollamaClient) {
      if (this.#options.apiKey) {
        throw new Error('Cannot provide both apiKey and ollamaClient');
      }
      if (this.#options.baseURL) {
        throw new Error('Cannot provide both baseURL and ollamaClient');
      }
      this.#client = this.#options.ollamaClient;
    }
  }

  /**
   * Lazy loads the Ollama client to not throw an error if you don't have an API key set but
   * never actually use the client.
   */
  async #getClient(): Promise<any> {
    if (!this.#client) {
      // Dynamically import Ollama only when needed
      const { Ollama } = await import('ollama');

      // Use Ollama Cloud URL if model name ends with "-cloud", otherwise use localhost
      const defaultHost = this.#currentModelName?.endsWith('-cloud')
        ? 'https://ollama.com'
        : 'http://localhost:11434';

      const config: any = {
        host: this.#options.baseURL || defaultHost,
      };

      if (this.#options.apiKey) {
        config.headers = {
          Authorization: `Bearer ${this.#options.apiKey}`,
        };
      }

      this.#client = new Ollama(config);
    }
    return this.#client;
  }

  async getModel(modelName: string): Promise<Model> {
    // Store the model name to determine the default host
    this.#currentModelName = modelName;
    const client = await this.#getClient();
    return new OllamaModel(modelName, client);
  }
}
