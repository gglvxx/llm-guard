"""Typed, validated application settings loaded from ``config.yaml``.

Every tunable of the firewall (thresholds, layer weights, rules, providers) lives
in one YAML file that is parsed with ``yaml.safe_load`` and validated by Pydantic,
so a typo or an out-of-range value fails fast at start-up instead of silently
weakening a security control.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.models.schemas import Severity

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
CONFIG_ENV_VAR = "LLM_GUARD_CONFIG"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing, malformed or invalid."""


class _Section(BaseModel):
    """Base for all settings sections: immutable and strict about unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_message_template(value: str) -> str:
    """User-facing block messages may only use the ``{request_id}`` placeholder."""
    try:
        value.format(request_id="test")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("message may only contain the {request_id} placeholder") from exc
    return value


class AppSettings(_Section):
    name: str = "LLM-Guard"
    #: Enables the firewall ON/OFF switch and the telemetry reset endpoint.
    #: Must be ``false`` anywhere outside a demo: clients could otherwise opt out of the firewall.
    demo_mode: bool = True
    max_input_chars: int = Field(default=4000, ge=1, le=20_000)
    #: ``closed``: a crashing detector blocks the request. ``open``: it is skipped.
    fail_mode: Literal["closed", "open"] = "closed"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class NormalizerSettings(_Section):
    enabled: bool = True
    max_decode_depth: int = Field(default=2, ge=0, le=4)
    min_encoded_length: int = Field(default=16, ge=8, le=256)
    max_decoded_chars: int = Field(default=4000, ge=64, le=100_000)
    max_views: int = Field(default=12, ge=1, le=64)


class CustomRule(_Section):
    """A user-defined signature appended to the built-in rule set."""

    id: str = Field(pattern=r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$")
    category: str = Field(min_length=1)
    severity: Severity
    pattern: str = Field(min_length=1, max_length=2000)
    description: str = Field(min_length=1)

    @field_validator("pattern")
    @classmethod
    def _must_compile(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc
        return value


def _default_severity_scores() -> dict[Severity, float]:
    return {
        Severity.LOW: 25.0,
        Severity.MEDIUM: 45.0,
        Severity.HIGH: 75.0,
        Severity.CRITICAL: 95.0,
    }


class HeuristicSettings(_Section):
    enabled: bool = True
    severity_scores: dict[Severity, float] = Field(default_factory=_default_severity_scores)
    #: Extra points when a signature only matches after reversing an obfuscation layer.
    evasion_bonus: float = Field(default=15.0, ge=0.0, le=50.0)
    entropy_threshold: float = Field(default=4.5, ge=1.0, le=8.0)
    entropy_min_length: int = Field(default=32, ge=8, le=1024)
    disabled_rules: tuple[str, ...] = ()
    custom_rules: tuple[CustomRule, ...] = ()

    @field_validator("severity_scores")
    @classmethod
    def _complete_and_bounded(cls, value: dict[Severity, float]) -> dict[Severity, float]:
        missing = [s.value for s in Severity if s not in value]
        if missing:
            raise ValueError(f"missing severity scores for: {', '.join(missing)}")
        if any(not 0.0 <= score <= 100.0 for score in value.values()):
            raise ValueError("severity scores must be within 0-100")
        return value


class MLSettings(_Section):
    enabled: bool = True
    dataset_path: Path = Path("data/dataset.json")
    model_dir: Path = Path("data/model")
    #: Persist the trained artefact (integrity-checked on load). ``false`` = train in memory only.
    persist_model: bool = True
    #: Probability at which the classifier starts contributing risk.
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)


class LayerWeights(_Section):
    """Trust coefficient of each input layer (0 disables its influence, 1 = full trust)."""

    normalizer: float = Field(default=0.6, ge=0.0, le=1.0)
    heuristic: float = Field(default=1.0, ge=0.0, le=1.0)
    ml_classifier: float = Field(default=0.6, ge=0.0, le=1.0)


class Thresholds(_Section):
    flag: int = Field(default=40, ge=1, le=99)
    block: int = Field(default=70, ge=2, le=100)

    @model_validator(mode="after")
    def _ordered(self) -> Thresholds:
        if self.flag >= self.block:
            raise ValueError("thresholds.flag must be strictly lower than thresholds.block")
        return self


class PolicySettings(_Section):
    weights: LayerWeights = LayerWeights()
    thresholds: Thresholds = Thresholds()
    #: Stop the chain as soon as the accumulated risk reaches the BLOCK threshold.
    #: Off by default so the dashboard always shows the full per-layer breakdown.
    short_circuit: bool = False
    block_message: str = (
        "Request blocked by LLM-Guard: this prompt violates the security policy. "
        "The incident has been logged (ref: {request_id})."
    )

    _valid_message = field_validator("block_message")(_check_message_template)


class SecretPattern(_Section):
    label: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    pattern: str = Field(min_length=1, max_length=2000)

    @field_validator("pattern")
    @classmethod
    def _must_compile(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc
        return value


class DLPSettings(_Section):
    enabled: bool = True
    redaction_template: str = "[REDACTED:{label}]"
    disabled_detectors: tuple[str, ...] = ()
    custom_secret_patterns: tuple[SecretPattern, ...] = (
        SecretPattern(label="SECRET", pattern=r"SECRET_FLAG\{[^}\s]{1,64}\}"),
    )
    #: Word n-gram size used to fingerprint the protected system prompt.
    leak_shingle_size: int = Field(default=5, ge=3, le=12)
    #: Share of the system prompt's shingles that must reappear in an output to call it a leak.
    leak_containment_threshold: float = Field(default=0.12, gt=0.0, le=1.0)
    output_block_message: str = (
        "Response withheld by LLM-Guard: the model output contained protected information. "
        "The incident has been logged (ref: {request_id})."
    )

    _valid_message = field_validator("output_block_message")(_check_message_template)

    @field_validator("redaction_template")
    @classmethod
    def _has_label(cls, value: str) -> str:
        if "{label}" not in value:
            raise ValueError("redaction_template must contain the {label} placeholder")
        return value


class OllamaSettings(_Section):
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2"
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    health_timeout_seconds: float = Field(default=1.5, gt=0.0, le=30.0)
    #: How long a health-check result is cached before Ollama is probed again.
    health_ttl_seconds: float = Field(default=15.0, ge=0.0, le=3600.0)
    #: The project is local-first: a non-loopback Ollama host must be an explicit choice.
    allow_remote: bool = False

    @model_validator(mode="after")
    def _local_only_by_default(self) -> OllamaSettings:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("ollama.base_url must be an http(s) URL")
        if not self.allow_remote and parsed.hostname not in _LOOPBACK_HOSTS:
            raise ValueError(
                "ollama.base_url must point to localhost unless ollama.allow_remote is true"
            )
        return self


class MockSettings(_Section):
    #: Artificial latency so the dashboard shows what a BLOCK saves in LLM time.
    simulated_latency_ms: int = Field(default=120, ge=0, le=10_000)


class ProviderSettings(_Section):
    default: Literal["mock", "ollama"] = "mock"
    ollama: OllamaSettings = OllamaSettings()
    mock: MockSettings = MockSettings()


class RateLimitSettings(_Section):
    enabled: bool = True
    requests: int = Field(default=120, ge=1, le=100_000)
    window_seconds: int = Field(default=60, ge=1, le=86_400)


class TelemetrySettings(_Section):
    max_events: int = Field(default=500, ge=1, le=100_000)
    recent_events: int = Field(default=50, ge=1, le=1000)
    prompt_preview_chars: int = Field(default=160, ge=16, le=1000)


class Settings(_Section):
    """Root settings object. Every section has safe defaults."""

    app: AppSettings = AppSettings()
    normalizer: NormalizerSettings = NormalizerSettings()
    heuristics: HeuristicSettings = HeuristicSettings()
    ml: MLSettings = MLSettings()
    policy: PolicySettings = PolicySettings()
    dlp: DLPSettings = DLPSettings()
    providers: ProviderSettings = ProviderSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    telemetry: TelemetrySettings = TelemetrySettings()

    @staticmethod
    def resolve_path(path: Path) -> Path:
        """Resolve a possibly relative path from the config against the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from ``path``, ``$LLM_GUARD_CONFIG`` or the default ``config.yaml``.

    A missing *default* file falls back to built-in defaults; an explicitly
    requested file that does not exist is an error.
    """
    explicit = path if path is not None else os.environ.get(CONFIG_ENV_VAR)
    config_path = Path(explicit) if explicit is not None else DEFAULT_CONFIG_PATH

    if not config_path.is_file():
        if explicit is not None:
            raise ConfigError(f"configuration file not found: {config_path}")
        return Settings()

    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read configuration file {config_path}: {exc}") from exc

    if raw is None:
        return Settings()
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")

    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {config_path}:\n{exc}") from exc
