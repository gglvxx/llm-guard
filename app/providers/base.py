"""Strategy interface for LLM back-ends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class ProviderUnavailableError(ProviderError):
    """The back-end cannot be reached or returned an unusable response."""


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One completion request as prepared by the firewall."""

    #: Text sent as the user turn (wrapped in the defensive fence when ``hardened``).
    prompt: str
    #: ``True`` when the firewall flagged the prompt and fenced it as untrusted data.
    hardened: bool = False
    #: The fenced user text on its own (only set when ``hardened``).
    untrusted_body: str | None = None


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    provider: str
    model: str
    fallback_used: bool = False


class BaseLLMProvider(ABC):
    """Common contract for every LLM back-end (Strategy pattern).

    The rest of the application only depends on this interface, so swapping the
    deterministic simulator for a real local model is a configuration change.
    """

    name: ClassVar[str]

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        """The confidential instructions this back-end runs with."""
        return self._system_prompt

    @property
    def fallback_active(self) -> bool:
        """Whether responses are currently served by a fallback back-end."""
        return False

    def describe(self) -> str:
        """Human-readable identity for status pages."""
        return self.name

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResult:
        """Produce a completion or raise :class:`ProviderUnavailableError`."""

    async def is_available(self) -> bool:
        return True

    async def aclose(self) -> None:  # noqa: B027 - optional hook, intentionally empty
        """Release network resources (no-op by default)."""
