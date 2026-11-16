"""Request orchestration: firewall -> LLM -> output guard -> telemetry."""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from collections.abc import Iterable

from app.config import Settings
from app.core.dlp import OutputGuard
from app.core.firewall import Firewall
from app.core.telemetry import TelemetryStore
from app.models.schemas import (
    ChatResponse,
    Decision,
    LayerReport,
    SecurityEvent,
    SecurityReport,
    Stage,
)
from app.providers.base import BaseLLMProvider, LLMRequest

security_logger = logging.getLogger("llm_guard.security")

#: Detector-internal categories that are not attack vectors in their own right.
_NON_VECTOR_CATEGORIES = frozenset({"ml_adversarial_intent", "detector_failure"})


class ChatService:
    """Application service behind ``POST /api/v1/chat`` (framework-agnostic).

    Enforcement modes:

    * **enforced** (default): BLOCK never reaches the LLM, FLAG is fenced in the
      defensive wrapper, and the response is redacted or withheld by the output guard.
    * **shadow** (demo mode with the firewall switched off): the same analysis
      runs and is logged, but nothing is applied. This is what makes the A/B
      demo honest: the dashboard shows exactly what the firewall *would* have done.
    """

    def __init__(
        self,
        settings: Settings,
        firewall: Firewall,
        provider: BaseLLMProvider,
        telemetry: TelemetryStore,
    ) -> None:
        self._settings = settings
        self._firewall = firewall
        self._provider = provider
        self._telemetry = telemetry

    async def handle(
        self, *, message: str, firewall_enabled: bool, request_id: str
    ) -> ChatResponse:
        started = time.perf_counter()
        # Outside demo mode the client-supplied switch is ignored: the firewall is not optional.
        enforced = firewall_enabled or not self._settings.app.demo_mode

        verdict = self._firewall.inspect_input(message)
        firewall_ms = _ms_since(started)
        layers: list[LayerReport] = list(verdict.reports)

        blocked_at: Stage | None = None
        sanitized = False
        llm_called = False
        llm_ms = 0.0
        output_score = 0.0
        redactions: dict[str, int] = {}
        provider_name = self._provider.name

        if enforced and verdict.decision is Decision.BLOCK:
            response_text = self._settings.policy.block_message.format(request_id=request_id)
            blocked_at = "input"
            layers.append(
                LayerReport(
                    name=OutputGuard.name, stage="output", score=0.0, weight=1.0, skipped=True
                )
            )
        else:
            if enforced:
                forwarded = self._firewall.prepare_prompt(verdict)
                sanitized = forwarded.sanitized
                request = LLMRequest(
                    prompt=forwarded.text,
                    hardened=forwarded.sanitized,
                    untrusted_body=forwarded.untrusted_body,
                )
            else:
                request = LLMRequest(prompt=message)

            llm_started = time.perf_counter()
            result = await self._provider.generate(request)
            llm_ms = _ms_since(llm_started)
            llm_called = True
            provider_name = result.provider

            output_started = time.perf_counter()
            output = self._firewall.inspect_output(result.text)
            firewall_ms += _ms_since(output_started)
            layers.append(output.report)
            output_score = output.score

            if not enforced:
                response_text = result.text
            elif output.blocked:
                response_text = self._settings.dlp.output_block_message.format(
                    request_id=request_id
                )
                blocked_at = "output"
            else:
                response_text = output.safe_text
                redactions = output.redactions

        risk = max(verdict.risk_score, output_score)
        decision = self._firewall.decide(risk)
        # Scores are floored for display: thresholds are integers, so a floored score can
        # never appear to sit in a higher band than the decision that was actually enforced.
        report = SecurityReport(
            risk_score=math.floor(risk),
            decision=decision,
            enforced=enforced,
            blocked_at=blocked_at,
            input_risk_score=math.floor(verdict.risk_score),
            output_risk_score=math.floor(output_score),
            sanitized=sanitized,
            llm_called=llm_called,
            layers=layers,
            triggered_rules=_unique(f.rule_id for layer in layers for f in layer.findings),
            attack_vectors=_attack_vectors(layers),
            redactions=redactions,
            firewall_ms=round(firewall_ms, 3),
            llm_ms=round(llm_ms, 3),
            total_ms=_ms_since(started),
        )
        self._record(report, message=message, canonical=verdict.canonical,
                     request_id=request_id, provider=provider_name)  # fmt: skip
        return ChatResponse(
            request_id=request_id,
            response=response_text,
            provider=provider_name,
            decision=decision,
            security=report,
        )

    def _record(
        self,
        report: SecurityReport,
        *,
        message: str,
        canonical: str,
        request_id: str,
        provider: str,
    ) -> None:
        """Store the telemetry event and emit the structured security log line.

        Neither ever holds the raw prompt: only a SHA-256 for correlation and a
        short preview that is normalised (no invisible or bidi characters can
        reach the SOC dashboard) and scrubbed of PII.
        """
        limit = self._settings.telemetry.prompt_preview_chars
        preview = self._firewall.scrub(canonical)
        if len(preview) > limit:
            preview = preview[: limit - 1] + "…"

        event = SecurityEvent(
            event_id=uuid.uuid4().hex,
            request_id=request_id,
            decision=report.decision,
            enforced=report.enforced,
            blocked_at=report.blocked_at,
            risk_score=report.risk_score,
            attack_vectors=report.attack_vectors,
            triggered_rules=report.triggered_rules,
            prompt_preview=preview,
            prompt_sha256=hashlib.sha256(message.encode("utf-8")).hexdigest(),
            redaction_count=sum(report.redactions.values()),
            firewall_ms=report.firewall_ms,
            total_ms=report.total_ms,
            provider=provider,
        )
        self._telemetry.record(event)

        level = logging.INFO if report.decision is Decision.ALLOW else logging.WARNING
        security_logger.log(
            level,
            "security_event",
            extra={
                "event_id": event.event_id,
                "decision": report.decision.value,
                "enforced": report.enforced,
                "blocked_at": report.blocked_at,
                "risk_score": report.risk_score,
                "input_risk_score": report.input_risk_score,
                "output_risk_score": report.output_risk_score,
                "rules": report.triggered_rules,
                "vectors": report.attack_vectors,
                "sanitized": report.sanitized,
                "redactions": report.redactions,
                "prompt_sha256": event.prompt_sha256,
                "prompt_preview": preview,
                "provider": provider,
                "firewall_ms": report.firewall_ms,
                "llm_ms": report.llm_ms,
            },
        )


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _unique(items: Iterable[str]) -> list[str]:
    """De-duplicate while preserving first-seen order."""
    return list(dict.fromkeys(items))


def _attack_vectors(layers: list[LayerReport]) -> list[str]:
    categories = _unique(f.category for layer in layers for f in layer.findings)
    vectors = [category for category in categories if category not in _NON_VECTOR_CATEGORIES]
    return vectors or categories
