from typing import Optional, Any
from .._vendored_imports import Model, ModelProvider
from ..models.ollama_model import OllamaModel


class OllamaModelProvider(ModelProvider):
    """The provider of Ollama's models"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        ollama_client: Optional[Any] = None,  # Will be dynamically imported
    ):
        # Match TypeScript: constructor accepts options dict directly
        options = type('Options', (), {
            'api_key': api_key,
            'base_url': base_url,
            'ollama_client': ollama_client,
        })()

        if options.ollama_client:
            if options.api_key:
                raise ValueError("Cannot provide both apiKey and ollamaClient")
            if options.base_url:
                raise ValueError("Cannot provide both baseURL and ollamaClient")
            self._client = options.ollama_client
        else:
            self._client = None

        self._options = options
        self._current_model_name: Optional[str] = None

    def _get_client(self) -> Any:
        """Lazy loads the Ollama client to not throw an error if you don't have an API key set but
        never actually use the client.
        
        Note: This is synchronous because Python ModelProvider.get_model must be synchronous.
        """
        if self._client is None:
            # Dynamically import Ollama AsyncClient only when needed
            # We use AsyncClient because we call chat() with await
            try:
                from ollama import AsyncClient
            except ImportError:
                raise ImportError(
                    "ollama package is required. Install it with: pip install ollama"
                )

            # Use Ollama Cloud URL if model name ends with "-cloud", otherwise use localhost
            default_host = (
                "https://ollama.com"
                if self._current_model_name and self._current_model_name.endswith("-cloud")
                else "http://localhost:11434"
            )

            host = self._options.base_url or default_host
            
            # Python ollama.AsyncClient takes host as first positional argument
            # and headers can be passed via kwargs to httpx
            client_kwargs: dict[str, Any] = {}
            if self._options.api_key:
                client_kwargs["headers"] = {
                    "Authorization": f"Bearer {self._options.api_key}",
                }

            self._client = AsyncClient(host=host, **client_kwargs)

        return self._client

    def get_model(self, model_name: str) -> Model:
        """Get a model instance for the given model name.
        
        Note: This must be synchronous to match Python ModelProvider interface.
        """
        # Store the model name to determine the default host
        self._current_model_name = model_name
        client = self._get_client()
        return OllamaModel(model_name, client)

