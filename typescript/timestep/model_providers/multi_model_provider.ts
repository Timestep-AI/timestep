import { Model, ModelProvider } from '@openai/agents-core';
import { OpenAIProvider } from '@openai/agents-openai';

export class MultiModelProviderMap {
  /** A map of model name prefixes to ModelProviders. */
  private _mapping: Map<string, ModelProvider> = new Map();

  hasPrefix(prefix: string): boolean {
    /** Returns True if the given prefix is in the mapping. */
    return this._mapping.has(prefix);
  }

  getMapping(): Map<string, ModelProvider> {
    /** Returns a copy of the current prefix -> ModelProvider mapping. */
    return new Map(this._mapping);
  }

  setMapping(mapping: Map<string, ModelProvider>): void {
    /** Overwrites the current mapping with a new one. */
    this._mapping = new Map(mapping);
  }

  getProvider(prefix: string): ModelProvider | undefined {
    /** Returns the ModelProvider for the given prefix.

        Args:
            prefix: The prefix of the model name e.g. "openai" or "my_prefix".
        */
    return this._mapping.get(prefix);
  }

  addProvider(prefix: string, provider: ModelProvider): void {
    /** Adds a new prefix -> ModelProvider mapping.

        Args:
            prefix: The prefix of the model name e.g. "openai" or "my_prefix".
            provider: The ModelProvider to use for the given prefix.
        */
    this._mapping.set(prefix, provider);
  }

  removeProvider(prefix: string): void {
    /** Removes the mapping for the given prefix.

        Args:
            prefix: The prefix of the model name e.g. "openai" or "my_prefix".
        */
    this._mapping.delete(prefix);
  }
}

export class MultiModelProvider implements ModelProvider {
  /** This ModelProvider maps to a Model based on the prefix of the model name. By default, the
    mapping is:
    - "openai/" prefix or no prefix -> OpenAIProvider. e.g. "openai/gpt-4.1", "gpt-4.1"
    - "ollama/" prefix -> OllamaModelProvider. e.g. "ollama/gpt-oss:20b" (local) or "ollama/gpt-oss:20b-cloud" (cloud)
    
    Note: The `-cloud` suffix in the model name (not the prefix) determines whether Ollama Cloud
    or local Ollama is used. Models without the `-cloud` suffix use local Ollama by default.

    You can override or customize this mapping.
    */

  private provider_map: MultiModelProviderMap | undefined;
  private openai_provider: OpenAIProvider;
  private _fallback_providers: Map<string, ModelProvider> = new Map();

  constructor(
    options: {
      provider_map?: MultiModelProviderMap;
      openai_api_key?: string;
      openai_base_url?: string;
      openai_client?: any; // AsyncOpenAI type
      openai_organization?: string;
      openai_project?: string;
      openai_use_responses?: boolean;
    } = {}
  ) {
    /** Create a new OpenAI provider.

        Args:
            provider_map: A MultiModelProviderMap that maps prefixes to ModelProviders. If not provided,
                we will use a default mapping. See the documentation for this class to see the
                default mapping.
            openai_api_key: The API key to use for the OpenAI provider. If not provided, we will use
                the default API key.
            openai_base_url: The base URL to use for the OpenAI provider. If not provided, we will
                use the default base URL.
            openai_client: An optional OpenAI client to use. If not provided, we will create a new
                OpenAI client using the api_key and base_url.
            openai_organization: The organization to use for the OpenAI provider.
            openai_project: The project to use for the OpenAI provider.
            openai_use_responses: Whether to use the OpenAI responses API.
        */
    this.provider_map = options.provider_map;
    this.openai_provider = new OpenAIProvider({
      apiKey: options.openai_api_key,
      baseURL: options.openai_base_url,
      openAIClient: options.openai_client,
      organization: options.openai_organization,
      project: options.openai_project,
      useResponses: options.openai_use_responses,
    });
  }

  private _getPrefixAndModelName(
    model_name: string | undefined
  ): [string | undefined, string | undefined] {
    if (model_name === undefined) {
      return [undefined, undefined];
    } else if (model_name.includes('/')) {
      const [prefix, ...rest] = model_name.split('/');
      return [prefix, rest.join('/')];
    } else {
      return [undefined, model_name];
    }
  }

  private async _createFallbackProvider(prefix: string): Promise<ModelProvider> {
    if (prefix === 'ollama') {
      // Import OllamaModelProvider only when needed
      const { OllamaModelProvider } = await import('./ollama_model_provider.ts');
      return new OllamaModelProvider();
    } else {
      throw new Error(`Unknown prefix: ${prefix}`);
    }
  }

  private async _getFallbackProvider(prefix: string | undefined): Promise<ModelProvider> {
    if (prefix === undefined || prefix === 'openai') {
      return this.openai_provider;
    } else if (this._fallback_providers.has(prefix)) {
      return this._fallback_providers.get(prefix)!;
    } else {
      const provider = await this._createFallbackProvider(prefix);
      this._fallback_providers.set(prefix, provider);
      return provider;
    }
  }

  async getModel(model_name: string | undefined): Promise<Model> {
    /** Returns a Model based on the model name. The model name can have a prefix, ending with
        a "/", which will be used to look up the ModelProvider. If there is no prefix, we will use
        the OpenAI provider.

        Args:
            model_name: The name of the model to get.

        Returns:
            A Model.
        */
    const [prefix, actualModelName] = this._getPrefixAndModelName(model_name);

    if (prefix && this.provider_map) {
      const provider = this.provider_map.getProvider(prefix);
      if (provider) {
        return provider.getModel(actualModelName);
      }
    }

    const fallbackProvider = await this._getFallbackProvider(prefix);
    return fallbackProvider.getModel(actualModelName);
  }
}

