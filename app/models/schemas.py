"""Pydantic v2 schemas shared by the security core, the API and the telemetry store."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Absolute upper bound enforced at schema level; ``app.max_input_chars`` is the
#: (lower) operational limit enforced by the API route.
HARD_MAX_INPUT_CHARS = 20_000

Stage = Literal["input", "output"]


class Decision(StrEnum):
    """Policy verdict for a request."""

    ALLOW = "ALLOW"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


class Severity(StrEnum):
    """Severity of a single detection finding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Finding(BaseModel):
    """One piece of evidence produced by a detection layer."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    category: str
    severity: Severity
    description: str
    score: float = Field(ge=0.0, le=100.0)
    #: Which representation of the text matched (``canonical``, ``base64``, ``rot13`` ...).
    view: str = "canonical"
    evidence: str | None = None


class LayerReport(BaseModel):
    """Result of a single pipeline stage."""

    name: str
    stage: Stage
    score: float = Field(ge=0.0, le=100.0)
    weight: float = Field(ge=0.0, le=1.0)
    findings: list[Finding] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0, ge=0.0)
    skipped: bool = False
    error: str | None = None
    details: dict[str, str | int | float | bool] = Field(default_factory=dict)


class SecurityReport(BaseModel):
    """Complete security assessment attached to every chat response."""

    risk_score: int = Field(ge=0, le=100)
    decision: Decision
    #: ``False`` means shadow mode: the verdict was computed and logged but not applied.
    enforced: bool
    blocked_at: Stage | None = None
    input_risk_score: int = Field(ge=0, le=100)
    output_risk_score: int = Field(ge=0, le=100)
    sanitized: bool = False
    llm_called: bool = False
    layers: list[LayerReport] = Field(default_factory=list)
    triggered_rules: list[str] = Field(default_factory=list)
    attack_vectors: list[str] = Field(default_factory=list)
    redactions: dict[str, int] = Field(default_factory=dict)
    firewall_ms: float = Field(ge=0.0)
    llm_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class ChatRequest(BaseModel):
    """Body of ``POST /api/v1/chat``."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=HARD_MAX_INPUT_CHARS)
    #: Demo-only A/B switch. Ignored (always ``True``) unless ``app.demo_mode`` is on.
    firewall_enabled: bool = True

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain non-whitespace characters")
        return value


class ChatResponse(BaseModel):
    """Body returned by ``POST /api/v1/chat``."""

    request_id: str
    response: str
    provider: str
    decision: Decision
    security: SecurityReport


class SecurityEvent(BaseModel):
    """Telemetry record for one processed request (never stores the raw prompt)."""

    event_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str
    decision: Decision
    enforced: bool
    blocked_at: Stage | None = None
    risk_score: int = Field(ge=0, le=100)
    attack_vectors: list[str] = Field(default_factory=list)
    triggered_rules: list[str] = Field(default_factory=list)
    prompt_preview: str
    prompt_sha256: str
    redaction_count: int = Field(default=0, ge=0)
    firewall_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)
    provider: str


class AttackVectorStat(BaseModel):
    vector: str
    count: int = Field(ge=0)


class TelemetrySnapshot(BaseModel):
    """Body returned by ``GET /api/v1/telemetry``."""

    generated_at: datetime
    uptime_seconds: float = Field(ge=0.0)
    total_requests: int = Field(ge=0)
    allowed_count: int = Field(ge=0)
    flagged_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    #: Requests served with the firewall switched off (demo mode only).
    unprotected_count: int = Field(ge=0)
    #: Unprotected requests the firewall *would* have flagged or blocked.
    shadow_detections: int = Field(ge=0)
    redactions_total: int = Field(ge=0)
    block_rate: float = Field(ge=0.0, le=1.0)
    avg_firewall_ms: float = Field(ge=0.0)
    avg_total_ms: float = Field(ge=0.0)
    top_attack_vectors: list[AttackVectorStat] = Field(default_factory=list)
    recent_events: list[SecurityEvent] = Field(default_factory=list)


class Preset(BaseModel):
    """A one-click playground scenario."""

    id: str
    title: str
    kind: Literal["benign", "attack"]
    technique: str
    description: str
    prompt: str
    expected: str


class ModelInfo(BaseModel):
    """Summary of the trained intent classifier."""

    algorithm: str
    samples: int
    cv_f1: float
    cv_accuracy: float
    trained_at: datetime


class HealthResponse(BaseModel):
    """Body returned by ``GET /api/v1/health``."""

    status: Literal["ok"] = "ok"
    version: str
    demo_mode: bool
    provider: str
    fallback_active: bool
    rules_loaded: int
    flag_threshold: int
    block_threshold: int
    max_input_chars: int
    ml_model: ModelInfo | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Uniform error envelope for every non-2xx API response."""

    error: ErrorDetail
