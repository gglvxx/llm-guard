"""Shared fixtures.

The classifier is trained once per session and injected everywhere, the mock
provider runs with zero simulated latency, and nothing is persisted to disk, so
the whole suite stays fast and leaves the repository untouched.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator
from typing import TYPE_CHECKING, Any, TypeVar

import pytest

from app.config import Settings
from app.core.firewall import Firewall
from app.core.ml_classifier import IntentClassifier, file_sha256, load_dataset
from app.core.orchestrator import ChatService
from app.core.telemetry import TelemetryStore
from app.providers.mock import MockVulnerableProvider
from app.providers.persona import CANARY_TOKEN, DEMO_SYSTEM_PROMPT

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

T = TypeVar("T")


def run(coroutine: Awaitable[T]) -> T:
    """Run a coroutine to completion (keeps the suite free of async plugins)."""
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def override(settings: Settings, **sections: dict[str, Any]) -> Settings:
    """Return a copy of ``settings`` with fields of the given sections replaced.

    >>> override(settings, app={"demo_mode": False})
    """
    updates = {
        name: getattr(settings, name).model_copy(update=fields) for name, fields in sections.items()
    }
    return settings.model_copy(update=updates)


@pytest.fixture(scope="session")
def settings() -> Settings:
    base = Settings()
    return override(
        base,
        app={"log_level": "ERROR"},
        ml={"persist_model": False},
        rate_limit={"requests": 10_000},
        providers={"mock": base.providers.mock.model_copy(update={"simulated_latency_ms": 0})},
    )


@pytest.fixture(scope="session")
def classifier(settings: Settings) -> IntentClassifier:
    dataset_path = settings.resolve_path(settings.ml.dataset_path)
    return IntentClassifier.train(
        load_dataset(dataset_path), dataset_sha256=file_sha256(dataset_path)
    )


@pytest.fixture(scope="session")
def firewall(settings: Settings, classifier: IntentClassifier) -> Firewall:
    return Firewall.build(
        settings,
        protected_prompt=DEMO_SYSTEM_PROMPT,
        canaries=(CANARY_TOKEN,),
        classifier=classifier,
    )


@pytest.fixture
def provider() -> MockVulnerableProvider:
    return MockVulnerableProvider()


@pytest.fixture
def telemetry(settings: Settings) -> TelemetryStore:
    return TelemetryStore(settings.telemetry)


@pytest.fixture
def chat_service(
    settings: Settings,
    firewall: Firewall,
    provider: MockVulnerableProvider,
    telemetry: TelemetryStore,
) -> ChatService:
    return ChatService(settings, firewall, provider, telemetry)


@pytest.fixture
def client(settings: Settings, classifier: IntentClassifier) -> Iterator[TestClient]:
    """HTTP client bound to a fresh application instance."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(settings, classifier=classifier)) as test_client:
        yield test_client
