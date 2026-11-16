"""The live demo, as a test: every one-click scenario must behave as advertised."""

from __future__ import annotations

import pytest

from app.api.presets import PRESETS
from app.core.orchestrator import ChatService
from app.models.schemas import Decision, Preset
from app.providers.persona import CANARY_TOKEN, SECRET_FLAG, SERVICE_API_KEY
from tests.conftest import run

LEAK_MARKERS = (SECRET_FLAG, SERVICE_API_KEY, CANARY_TOKEN, "@nexora.example")
ATTACKS = [preset for preset in PRESETS if preset.kind == "attack"]
BENIGN = [preset for preset in PRESETS if preset.kind == "benign"]


def leaked(text: str) -> bool:
    return any(marker in text for marker in LEAK_MARKERS)


@pytest.mark.parametrize("preset", PRESETS, ids=lambda preset: preset.id)
def test_preset_reaches_its_expected_verdict(chat_service: ChatService, preset: Preset) -> None:
    response = run(
        chat_service.handle(message=preset.prompt, firewall_enabled=True, request_id="req-preset")
    )
    assert response.decision.value == preset.expected
    assert not leaked(response.response)


@pytest.mark.parametrize("preset", ATTACKS, ids=lambda preset: preset.id)
def test_attack_succeeds_without_the_firewall(chat_service: ChatService, preset: Preset) -> None:
    """The A/B demo is only honest if the unprotected bot really falls for each attack."""
    response = run(
        chat_service.handle(message=preset.prompt, firewall_enabled=False, request_id="req-preset")
    )
    assert leaked(response.response)
    assert response.decision is not Decision.ALLOW  # shadow mode still raises the alarm


@pytest.mark.parametrize("preset", BENIGN, ids=lambda preset: preset.id)
def test_benign_presets_never_leak_even_unprotected(
    chat_service: ChatService, preset: Preset
) -> None:
    response = run(
        chat_service.handle(message=preset.prompt, firewall_enabled=False, request_id="req-preset")
    )
    assert not leaked(response.response)


def test_presets_exercise_every_stage_of_the_pipeline(chat_service: ChatService) -> None:
    outcomes = set()
    for preset in PRESETS:
        security = run(
            chat_service.handle(message=preset.prompt, firewall_enabled=True, request_id="r")
        ).security
        outcomes.add((security.decision.value, security.blocked_at, security.sanitized))

    assert ("ALLOW", None, False) in outcomes
    assert ("BLOCK", "input", False) in outcomes  # stopped before the model
    assert ("FLAG", None, True) in outcomes  # fenced in the defensive wrapper
    assert ("FLAG", None, False) in outcomes  # answer redacted by the output guard
    assert ("BLOCK", "output", False) in outcomes  # leak withheld by the output guard
