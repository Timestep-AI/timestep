from __future__ import annotations

from typing import Optional
from agents import ModelProvider
from ollama_model import OllamaModel
from ollama import AsyncClient

class OllamaModelProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        ollama_client: AsyncClient | None = None,
    ) -> None:
        """Create a new Ollama provider.

        Args:
            api_key: The API key to use for the Ollama client. If not provided, we will use the
                default API key.
            base_url: The base URL to use for the Ollama client. If not provided, we will use the
                default base URL.
            ollama_client: An optional Ollama client to use. If not provided, we will create a new
                Ollama client using the api_key and base_url.
        """
        if ollama_client is not None:
            assert api_key is None and base_url is None, (
                "Don't provide api_key or base_url if you provide ollama_client"
            )
            self._client: AsyncClient | None = ollama_client
        else:
            self._client = None
            self._stored_api_key = api_key
            self._stored_base_url = base_url

    # We lazy load the client in case you never actually use OllamaModelProvider(). Otherwise
    # AsyncClient() raises an error if you don't have an API key set.
    def _get_client(self) -> AsyncClient:
        if self._client is None:
            # Use Ollama Cloud URL if model name ends with "-cloud", otherwise use localhost
            default_host = "https://ollama.com" if hasattr(self, '_current_model_name') and self._current_model_name.endswith('-cloud') else "http://localhost:11434"
            
            config = {
                'host': self._stored_base_url or default_host
            }
            
            if self._stored_api_key:
                config['headers'] = {'Authorization': f'Bearer {self._stored_api_key}'}
            
            self._client = AsyncClient(**config)
        return self._client

    def get_model(self, model_name: str) -> Model:
        # Store the model name to determine the default host
        self._current_model_name = model_name
        client = self._get_client()
        return OllamaModel(model=model_name, ollama_client=client)
