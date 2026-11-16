"""Composition root: builds and exposes the application's object graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.core.firewall import Firewall
from app.core.ml_classifier import IntentClassifier
from app.core.orchestrator import ChatService
from app.core.telemetry import TelemetryStore
from app.providers.base import BaseLLMProvider
from app.providers.factory import build_provider
from app.providers.persona import CANARY_TOKEN


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Everything a request handler may need, wired once at start-up."""

    settings: Settings
    firewall: Firewall
    provider: BaseLLMProvider
    telemetry: TelemetryStore
    chat_service: ChatService


def build_container(
    settings: Settings,
    *,
    provider: BaseLLMProvider | None = None,
    classifier: IntentClassifier | None = None,
) -> AppContainer:
    """Wire the object graph. ``provider`` and ``classifier`` can be injected by tests."""
    provider = provider or build_provider(settings.providers)
    firewall = Firewall.build(
        settings,
        protected_prompt=provider.system_prompt,
        canaries=(CANARY_TOKEN,),
        classifier=classifier,
    )
    # Warm-up: pays scikit-learn's one-off first-prediction cost before real traffic.
    firewall.inspect_input("warm-up")
    telemetry = TelemetryStore(settings.telemetry)
    return AppContainer(
        settings=settings,
        firewall=firewall,
        provider=provider,
        telemetry=telemetry,
        chat_service=ChatService(settings, firewall, provider, telemetry),
    )


def get_container(request: Request) -> AppContainer:
    container: AppContainer = request.app.state.container
    return container


ContainerDep = Annotated[AppContainer, Depends(get_container)]
