from typing import Optional, Dict, Any
from ._vendored_imports import Model, ModelProvider, OpenAIProvider


class MultiModelProviderMap:
    """A map of model name prefixes to ModelProviders."""

    def __init__(self):
        self._mapping: Dict[str, ModelProvider] = {}

    def has_prefix(self, prefix: str) -> bool:
        """Returns True if the given prefix is in the mapping."""
        return prefix in self._mapping

    def get_mapping(self) -> Dict[str, ModelProvider]:
        """Returns a copy of the current prefix -> ModelProvider mapping."""
        return dict(self._mapping)

    def set_mapping(self, mapping: Dict[str, ModelProvider]) -> None:
        """Overwrites the current mapping with a new one."""
        self._mapping = dict(mapping)

    def get_provider(self, prefix: str) -> Optional[ModelProvider]:
        """Returns the ModelProvider for the given prefix.

        Args:
            prefix: The prefix of the model name e.g. "openai" or "my_prefix".
        """
        return self._mapping.get(prefix)

    def add_provider(self, prefix: str, provider: ModelProvider) -> None:
        """Adds a new prefix -> ModelProvider mapping.

        Args:
            prefix: The prefix of the model name e.g. "openai" or "my_prefix".
            provider: The ModelProvider to use for the given prefix.
        """
        self._mapping[prefix] = provider

    def remove_provider(self, prefix: str) -> None:
        """Removes the mapping for the given prefix.

        Args:
            prefix: The prefix of the model name e.g. "openai" or "my_prefix".
        """
        if prefix in self._mapping:
            del self._mapping[prefix]


class MultiModelProvider(ModelProvider):
    """This ModelProvider maps to a Model based on the prefix of the model name. By default, the
    mapping is:
    - "openai/" prefix or no prefix -> OpenAIProvider. e.g. "openai/gpt-4.1", "gpt-4.1"
    - "ollama/" prefix -> OllamaModelProvider. e.g. "ollama/gpt-oss:20b-cloud"

    You can override or customize this mapping.
    """

    def __init__(
        self,
        provider_map: Optional[MultiModelProviderMap] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_client: Optional[Any] = None,  # AsyncOpenAI type
        openai_organization: Optional[str] = None,
        openai_project: Optional[str] = None,
        openai_use_responses: Optional[bool] = None,
    ):
        """Create a new MultiModelProvider.

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
        """
        self.provider_map = provider_map
        self._fallback_providers: Dict[str, ModelProvider] = {}

        self.openai_provider = OpenAIProvider(
            api_key=openai_api_key,
            base_url=openai_base_url,
            openai_client=openai_client,
            organization=openai_organization,
            project=openai_project,
            use_responses=openai_use_responses,
        )

    def _get_prefix_and_model_name(
        self, model_name: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        if model_name is None:
            return None, None
        elif "/" in model_name:
            parts = model_name.split("/", 1)
            return parts[0], parts[1]
        else:
            return None, model_name

    def _create_fallback_provider(self, prefix: str) -> ModelProvider:
        if prefix == "ollama":
            # Import OllamaModelProvider only when needed
            from .ollama_model_provider import OllamaModelProvider

            return OllamaModelProvider()
        else:
            raise ValueError(f"Unknown prefix: {prefix}")

    def _get_fallback_provider(
        self, prefix: Optional[str]
    ) -> ModelProvider:
        if prefix is None or prefix == "openai":
            return self.openai_provider
        elif prefix in self._fallback_providers:
            return self._fallback_providers[prefix]
        else:
            provider = self._create_fallback_provider(prefix)
            self._fallback_providers[prefix] = provider
            return provider

    def get_model(self, model_name: Optional[str]) -> Model:
        """Returns a Model based on the model name. The model name can have a prefix, ending with
        a "/", which will be used to look up the ModelProvider. If there is no prefix, we will use
        the OpenAI provider.

        Args:
            model_name: The name of the model to get.

        Returns:
            A Model.
        """
        prefix, actual_model_name = self._get_prefix_and_model_name(model_name)

        if prefix and self.provider_map:
            provider = self.provider_map.get_provider(prefix)
            if provider:
                return provider.get_model(actual_model_name)

        fallback_provider = self._get_fallback_provider(prefix)
        return fallback_provider.get_model(actual_model_name)

