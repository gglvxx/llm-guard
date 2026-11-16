"""LLM back-ends (Strategy pattern): deterministic simulator, Ollama, automatic fallback."""

from app.providers.base import (
    BaseLLMProvider,
    LLMRequest,
    LLMResult,
    ProviderError,
    ProviderUnavailableError,
)
from app.providers.factory import FallbackProvider, build_provider
from app.providers.mock import MockVulnerableProvider
from app.providers.ollama import OllamaProvider

__all__ = [
    "BaseLLMProvider",
    "FallbackProvider",
    "LLMRequest",
    "LLMResult",
    "MockVulnerableProvider",
    "OllamaProvider",
    "ProviderError",
    "ProviderUnavailableError",
    "build_provider",
]
