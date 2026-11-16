"""Pipeline abstractions: the shared context and the detection-stage contract.

The input firewall is a Chain of Responsibility: an ordered list of
:class:`DetectionStage` handlers that each receive the same
:class:`PipelineContext`, enrich it, and may halt the chain. ``run`` is a
Template Method that gives every stage identical timing, error containment and
fail-closed behaviour, so concrete stages only implement ``_analyze``.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from app.core.scoring import clamp, noisy_or
from app.models.schemas import Finding, LayerReport, Severity

logger = logging.getLogger("llm_guard.pipeline")

FailMode = Literal["closed", "open"]
Details = dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class TextView:
    """One representation of the analysed text.

    ``obfuscated`` marks views obtained by *reversing* an evasion technique
    (decoding Base64, folding homoglyphs ...): a signature that only matches
    there shows intent to hide and earns an evasion bonus. ``speculative``
    marks views produced without evidence that the technique was used (blanket
    ROT13); they are scanned by signatures but not by the ML classifier.
    """

    name: str
    text: str
    obfuscated: bool = False
    speculative: bool = False


@dataclass(slots=True)
class PipelineContext:
    """Mutable state handed from stage to stage."""

    raw_text: str
    canonical: str = ""
    views: list[TextView] = field(default_factory=list)
    #: Original substrings that were successfully decoded (skipped by the entropy check).
    decoded_sources: set[str] = field(default_factory=set)
    reports: list[LayerReport] = field(default_factory=list)
    halted: bool = False

    def __post_init__(self) -> None:
        if not self.canonical:
            self.canonical = self.raw_text
        if not self.views:
            self.views = [TextView(name="canonical", text=self.canonical)]

    def iter_views(self, *, include_speculative: bool = True) -> Iterator[TextView]:
        for view in self.views:
            if include_speculative or not view.speculative:
                yield view


@dataclass(slots=True)
class StageOutcome:
    """What a stage returns from ``_analyze``."""

    findings: list[Finding] = field(default_factory=list)
    #: Explicit score; when ``None`` the findings are combined with noisy-OR.
    score: float | None = None
    details: Details = field(default_factory=dict)


class DetectionStage(ABC):
    """A single handler in the input detection chain."""

    name: ClassVar[str]

    def __init__(self, *, weight: float, enabled: bool = True) -> None:
        self.weight = weight
        self.enabled = enabled

    def run(self, ctx: PipelineContext, *, fail_mode: FailMode = "closed") -> LayerReport:
        """Execute the stage with timing and failure containment (Template Method)."""
        if not self.enabled:
            return LayerReport(
                name=self.name, stage="input", score=0.0, weight=self.weight, skipped=True
            )

        started = time.perf_counter()
        try:
            outcome = self._analyze(ctx)
        except Exception:
            logger.exception("detection stage crashed", extra={"stage": self.name})
            return self._failure_report(started, fail_mode)

        score = (
            outcome.score
            if outcome.score is not None
            else noisy_or(finding.score for finding in outcome.findings)
        )
        return LayerReport(
            name=self.name,
            stage="input",
            score=clamp(score),
            weight=self.weight,
            findings=outcome.findings,
            duration_ms=_elapsed_ms(started),
            details=outcome.details,
        )

    def _failure_report(self, started: float, fail_mode: FailMode) -> LayerReport:
        """Fail closed (treat the crash as maximum risk) or open (skip the layer)."""
        if fail_mode == "open":
            return LayerReport(
                name=self.name,
                stage="input",
                score=0.0,
                weight=self.weight,
                duration_ms=_elapsed_ms(started),
                skipped=True,
                error="detector failed; skipped (fail-open)",
            )
        finding = Finding(
            rule_id="SYS-DETECTOR-ERROR",
            category="detector_failure",
            severity=Severity.CRITICAL,
            description=f"Layer '{self.name}' failed; request blocked by fail-closed policy",
            score=100.0,
        )
        return LayerReport(
            name=self.name,
            stage="input",
            score=100.0,
            # A broken detector must not be down-weighted into an ALLOW.
            weight=1.0,
            findings=[finding],
            duration_ms=_elapsed_ms(started),
            error="detector failed; blocked (fail-closed)",
        )

    @abstractmethod
    def _analyze(self, ctx: PipelineContext) -> StageOutcome:
        """Inspect (and optionally enrich) the context; return findings."""


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def snippet(text: str, start: int, end: int, *, radius: int = 24, limit: int = 96) -> str:
    """Return a single-line, bounded excerpt around ``text[start:end]`` for evidence."""
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    excerpt = " ".join(text[left:right].split())
    if len(excerpt) > limit:
        excerpt = excerpt[: limit - 1] + "…"
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) and not excerpt.endswith("…") else ""
    return f"{prefix}{excerpt}{suffix}"
