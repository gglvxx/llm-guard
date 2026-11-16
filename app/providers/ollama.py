"""Real local LLM back-end: an Ollama server reached over async HTTP."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from app.config import OllamaSettings
from app.providers._http import httpx
from app.providers.base import BaseLLMProvider, LLMRequest, LLMResult, ProviderUnavailableError

logger = logging.getLogger("llm_guard.providers.ollama")


class _ChatMessage(BaseModel):
    content: str


class _ChatResponse(BaseModel):
    """The subset of Ollama's ``/api/chat`` response that we rely on."""

    message: _ChatMessage


class _ModelTag(BaseModel):
    name: str


class _TagsResponse(BaseModel):
    models: list[_ModelTag] = []


class OllamaProvider(BaseLLMProvider):
    """Talks to ``POST /api/chat`` on a local Ollama server.

    ``trust_env=False`` keeps requests to localhost away from any system proxy,
    and the configured base URL is validated as loopback-only at start-up
    (``OllamaSettings``), so prompts cannot be shipped off the machine by a
    misconfigured environment.
    """

    name = "ollama"

    def __init__(
        self,
        settings: OllamaSettings,
        system_prompt: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(system_prompt)
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(
                settings.timeout_seconds, connect=settings.health_timeout_seconds
            ),
            transport=transport,
            trust_env=False,
        )

    async def is_available(self) -> bool:
        """``True`` when the server answers and the configured model is pulled."""
        try:
            response = await self._client.get(
                "/api/tags", timeout=self._settings.health_timeout_seconds
            )
            response.raise_for_status()
            tags = _TagsResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError):
            return False

        wanted = self._settings.model
        available = any(
            tag.name == wanted or tag.name.split(":", 1)[0] == wanted for tag in tags.models
        )
        if not available:
            logger.warning("ollama is running but the model is not pulled", extra={"model": wanted})
        return available

    async def generate(self, request: LLMRequest) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "options": {"temperature": 0.2},
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            parsed = _ChatResponse.model_validate(response.json())
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"ollama request failed: {exc!r}") from exc
        except (ValueError, ValidationError) as exc:
            raise ProviderUnavailableError("ollama returned an unexpected response") from exc

        return LLMResult(
            text=parsed.message.content, provider=self.name, model=self._settings.model
        )

    async def aclose(self) -> None:
        await self._client.aclose()
