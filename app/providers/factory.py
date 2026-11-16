"""Provider selection and the automatic fallback decorator."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import replace

from app.config import ProviderSettings
from app.providers.base import BaseLLMProvider, LLMRequest, LLMResult, ProviderUnavailableError
from app.providers.mock import MockVulnerableProvider
from app.providers.ollama import OllamaProvider
from app.providers.persona import DEMO_SYSTEM_PROMPT

logger = logging.getLogger("llm_guard.providers")


class FallbackProvider(BaseLLMProvider):
    """Decorator that serves from ``fallback`` whenever ``primary`` is unavailable.

    The primary's health is cached for ``health_ttl_seconds`` (a tiny circuit
    breaker), so a stopped Ollama costs one short probe per TTL window instead
    of a connection timeout on every chat request.
    """

    name = "fallback"

    def __init__(
        self,
        primary: BaseLLMProvider,
        fallback: BaseLLMProvider,
        *,
        health_ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(primary.system_prompt)
        self._primary = primary
        self._fallback = fallback
        self._ttl = health_ttl_seconds
        self._clock = clock
        self._healthy: bool | None = None
        self._checked_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def fallback_active(self) -> bool:
        return self._healthy is False

    async def _primary_healthy(self) -> bool:
        async with self._lock:
            expired = self._clock() - self._checked_at >= self._ttl
            if self._healthy is None or expired:
                was_healthy = self._healthy
                self._healthy = await self._primary.is_available()
                self._checked_at = self._clock()
                if self._healthy is not was_healthy:
                    logger.info(
                        "primary provider health changed",
                        extra={"provider": self._primary.name, "healthy": self._healthy},
                    )
            return self._healthy

    def _mark_unhealthy(self) -> None:
        self._healthy = False
        self._checked_at = self._clock()

    def describe(self) -> str:
        return f"{self._primary.name} (fallback: {self._fallback.name})"

    async def is_available(self) -> bool:
        """Always available thanks to the fallback; refreshes the primary's cached health."""
        await self._primary_healthy()
        return True

    async def generate(self, request: LLMRequest) -> LLMResult:
        if await self._primary_healthy():
            try:
                return await self._primary.generate(request)
            except ProviderUnavailableError:
                logger.warning(
                    "primary provider failed; falling back",
                    extra={"provider": self._primary.name},
                    exc_info=True,
                )
                self._mark_unhealthy()
        result = await self._fallback.generate(request)
        return replace(result, fallback_used=True)

    async def aclose(self) -> None:
        await self._primary.aclose()
        await self._fallback.aclose()


def build_provider(settings: ProviderSettings) -> BaseLLMProvider:
    """Create the configured provider; Ollama is always wrapped with a mock fallback."""
    mock = MockVulnerableProvider(DEMO_SYSTEM_PROMPT, latency_ms=settings.mock.simulated_latency_ms)
    if settings.default == "mock":
        return mock
    return FallbackProvider(
        OllamaProvider(settings.ollama, DEMO_SYSTEM_PROMPT),
        mock,
        health_ttl_seconds=settings.ollama.health_ttl_seconds,
    )
