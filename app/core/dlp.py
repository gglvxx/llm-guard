"""Layer 3 — output guardrail and Data Loss Prevention (DLP).

The input layers will miss some attacks; this layer assumes they did. Every
model response is inspected before it reaches the user:

* **PII and secrets** (e-mail, card numbers, CNP / SSN, API keys, JWTs, private
  keys, credential assignments, custom patterns) are masked in place.
  Checksum validators (Luhn, CNP control digit) keep false positives low.
* **System prompt leakage** is detected with a canary token (certain) and with
  word-shingle containment against the protected prompt (verbatim or partial
  copies). A leak withholds the whole response: there is nothing safe to keep.
* **Encoded exfiltration**: the response is de-obfuscated with the same
  normaliser as the input, so a secret returned in Base64 or hex is caught too.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.config import DLPSettings
from app.core.normalizer import TextNormalizer
from app.core.scoring import clamp, noisy_or
from app.models.schemas import Finding, LayerReport, Severity

PII_LEAK = "pii_leak"
SECRET_LEAK = "secret_leak"  # noqa: S105 - a category label, not a credential
PROMPT_LEAK = "system_prompt_leak"

#: Views that hold a *decoded payload* (as opposed to a re-spelling of the same text).
_DECODED_VIEW_ROOTS = frozenset({"base64", "hex", "unicode-tags", "variation-selectors"})

_CNP_WEIGHTS = (2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9)
_WORD = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- validators


def luhn_valid(candidate: str) -> bool:
    """Validate a payment card number with the Luhn checksum (13-19 digits)."""
    digits = [int(ch) for ch in candidate if ch.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def cnp_valid(candidate: str) -> bool:
    """Validate a Romanian CNP: 13 digits, plausible date, correct control digit.

    The county code is deliberately not checked, so the fictional CNPs used by
    the demo data (unallocated county ``99``) are still detected.
    """
    if len(candidate) != 13 or not candidate.isdigit() or candidate[0] == "0":
        return False
    month, day = int(candidate[3:5]), int(candidate[5:7])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return False
    control = sum(int(d) * w for d, w in zip(candidate[:12], _CNP_WEIGHTS, strict=True)) % 11
    return int(candidate[12]) == (1 if control == 10 else control)


# --------------------------------------------------------------------------- detectors


@dataclass(frozen=True, slots=True)
class SensitiveDataDetector:
    """One class of sensitive data: a pattern plus an optional checksum validator."""

    label: str
    category: str
    severity: Severity
    score: float
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None
    #: Capture group to mask (0 = the whole match).
    group: int = 0

    @property
    def rule_id(self) -> str:
        return f"DLP-{self.label.replace('_', '-')}"


def _detector(
    label: str,
    category: str,
    severity: Severity,
    score: float,
    pattern: str,
    *,
    validator: Callable[[str], bool] | None = None,
    group: int = 0,
    flags: int = 0,
) -> SensitiveDataDetector:
    return SensitiveDataDetector(
        label=label,
        category=category,
        severity=severity,
        score=score,
        pattern=re.compile(pattern, flags),
        validator=validator,
        group=group,
    )


# Order matters: earlier detectors win when two matches overlap.
DEFAULT_DETECTORS: tuple[SensitiveDataDetector, ...] = (
    _detector(
        "PRIVATE_KEY",
        SECRET_LEAK,
        Severity.CRITICAL,
        65.0,
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]{0,8192}?-----END (?:[A-Z]+ )?PRIVATE KEY-----",
    ),
    _detector(
        "JWT",
        SECRET_LEAK,
        Severity.HIGH,
        60.0,
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
        r"(?![A-Za-z0-9_-])",
    ),
    _detector(
        "API_KEY",
        SECRET_LEAK,
        Severity.CRITICAL,
        65.0,
        r"(?<![A-Za-z0-9])(?:"
        r"sk-[A-Za-z0-9_-]{20,}"
        r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
        r"|gh[pousr]_[A-Za-z0-9]{36,}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,}"
        r"|AIza[0-9A-Za-z_-]{35}"
        r")(?![A-Za-z0-9_-])",
    ),
    _detector(
        "CREDENTIAL",
        SECRET_LEAK,
        Severity.HIGH,
        55.0,
        r"\b(?:password|passwd|pwd|api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\b"
        r"\s*[:=]\s*[\"']?((?=[^\s\"',;]{0,127}\d)(?=[^\s\"',;]{0,127}[A-Za-z])[^\s\"',;]{8,128})",
        group=1,
        flags=re.IGNORECASE,
    ),
    _detector(
        "EMAIL",
        PII_LEAK,
        Severity.MEDIUM,
        40.0,
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}"
        r"(?:\.[A-Za-z0-9-]{1,63}){0,4}\.[A-Za-z]{2,24}(?![A-Za-z0-9])",
    ),
    _detector(
        "CNP",
        PII_LEAK,
        Severity.HIGH,
        60.0,
        r"(?<!\d)[1-9]\d{12}(?!\d)",
        validator=cnp_valid,
    ),
    _detector(
        "SSN",
        PII_LEAK,
        Severity.HIGH,
        60.0,
        r"(?<!\d)(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)",
    ),
    _detector(
        "CREDIT_CARD",
        PII_LEAK,
        Severity.HIGH,
        60.0,
        r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])",
        validator=luhn_valid,
    ),
)


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int
    detector: SensitiveDataDetector


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    counts: dict[str, int]
    findings: tuple[Finding, ...]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class PIIRedactor:
    """Finds and masks sensitive data. Also used to scrub prompts before they are logged."""

    def __init__(self, settings: DLPSettings | None = None) -> None:
        settings = settings or DLPSettings()
        self._template = settings.redaction_template
        disabled = set(settings.disabled_detectors)
        unknown = disabled - {detector.label for detector in DEFAULT_DETECTORS}
        if unknown:
            raise ValueError(f"unknown DLP detector(s) disabled: {', '.join(sorted(unknown))}")

        custom = tuple(
            _detector(spec.label, SECRET_LEAK, Severity.CRITICAL, 65.0, spec.pattern)
            for spec in settings.custom_secret_patterns
        )
        # Custom secrets are the most specific patterns, so they take precedence.
        self._detectors = custom + tuple(
            detector for detector in DEFAULT_DETECTORS if detector.label not in disabled
        )

    def find(self, text: str) -> list[_Span]:
        """Return non-overlapping sensitive spans, sorted by position."""
        accepted: list[_Span] = []
        for detector in self._detectors:
            for match in detector.pattern.finditer(text):
                start, end = match.span(detector.group)
                value = text[start:end]
                if detector.validator is not None and not detector.validator(value):
                    continue
                if any(start < span.end and span.start < end for span in accepted):
                    continue
                accepted.append(_Span(start, end, detector))
        return sorted(accepted, key=lambda span: span.start)

    def redact(self, text: str) -> RedactionResult:
        spans = self.find(text)
        if not spans:
            return RedactionResult(text=text, counts={}, findings=())

        parts: list[str] = []
        cursor = 0
        counts: Counter[str] = Counter()
        for span in spans:
            parts.append(text[cursor : span.start])
            parts.append(self._template.format(label=span.detector.label))
            cursor = span.end
            counts[span.detector.label] += 1
        parts.append(text[cursor:])

        by_label = {span.detector.label: span.detector for span in spans}
        findings = tuple(
            Finding(
                rule_id=detector.rule_id,
                category=detector.category,
                severity=detector.severity,
                description=f"{counts[label]} × {label} detected in the model output",
                score=detector.score,
                view="output",
            )
            for label, detector in by_label.items()
        )
        return RedactionResult(text="".join(parts), counts=dict(counts), findings=findings)


# --------------------------------------------------------------------------- prompt leakage


@dataclass(frozen=True, slots=True)
class LeakCheck:
    canary: str | None
    containment: float
    leaked: bool


class PromptLeakDetector:
    """Detects verbatim or partial reproduction of the protected system prompt."""

    def __init__(
        self,
        protected_text: str,
        canaries: Sequence[str],
        *,
        shingle_size: int,
        threshold: float,
    ) -> None:
        self._canaries = tuple(canary.lower() for canary in canaries if canary)
        self._shingle_size = shingle_size
        self._threshold = threshold
        self._fingerprint = self._shingles(protected_text)

    def _shingles(self, text: str) -> frozenset[str]:
        words = _WORD.findall(text.lower())
        size = self._shingle_size
        return frozenset(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))

    def check(self, text: str) -> LeakCheck:
        lowered = text.lower()
        canary = next((token for token in self._canaries if token in lowered), None)
        containment = 0.0
        if self._fingerprint:
            shared = len(self._fingerprint & self._shingles(text))
            containment = shared / len(self._fingerprint)
        return LeakCheck(
            canary=canary,
            containment=round(containment, 4),
            leaked=canary is not None or containment >= self._threshold,
        )


# --------------------------------------------------------------------------- output guard


@dataclass(frozen=True, slots=True)
class OutputVerdict:
    report: LayerReport
    #: Response with sensitive spans masked (meaningless when ``blocked`` is true).
    safe_text: str
    blocked: bool
    redactions: dict[str, int] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.report.score


class OutputGuard:
    """Inspects a model response and decides: pass, redact or withhold."""

    name = "output_dlp"

    def __init__(
        self,
        settings: DLPSettings,
        *,
        normalizer: TextNormalizer,
        redactor: PIIRedactor,
        leak_detector: PromptLeakDetector | None,
        redaction_score_range: tuple[float, float],
    ) -> None:
        self._settings = settings
        self._normalizer = normalizer
        self._redactor = redactor
        self._leak_detector = leak_detector
        #: A redacted response is a *contained* incident: its score stays in the FLAG band.
        self._redaction_floor, self._redaction_ceiling = redaction_score_range

    def inspect(self, text: str) -> OutputVerdict:
        if not self._settings.enabled:
            report = LayerReport(
                name=self.name, stage="output", score=0.0, weight=1.0, skipped=True
            )
            return OutputVerdict(report=report, safe_text=text, blocked=False)

        started = time.perf_counter()
        visible = self._normalizer.strip_invisible(text)
        normalized = self._normalizer.normalize(text)
        findings: list[Finding] = []
        max_containment = 0.0

        if self._leak_detector is not None:
            for view in normalized.views:
                check = self._leak_detector.check(view.text)
                max_containment = max(max_containment, check.containment)
                if check.canary is not None:
                    findings.append(self._leak_finding(
                        "DLP-CANARY", "Canary token from the system prompt found in the output",
                        view.name,
                    ))  # fmt: skip
                    break
                if check.leaked:
                    findings.append(self._leak_finding(
                        "DLP-PROMPT-LEAK",
                        f"Output reproduces {check.containment:.0%} of the protected system prompt",
                        view.name,
                    ))  # fmt: skip
                    break

        for view in normalized.views:
            if view.name.split(">")[0] in _DECODED_VIEW_ROOTS and self._redactor.find(view.text):
                findings.append(
                    Finding(
                        rule_id="DLP-ENCODED-EXFIL",
                        category=SECRET_LEAK,
                        severity=Severity.CRITICAL,
                        description="Sensitive data found inside an encoded part of the output",
                        score=100.0,
                        view=view.name,
                    )
                )
                break

        blocked = bool(findings)
        redaction = self._redactor.redact(visible)
        findings.extend(redaction.findings)

        if blocked:
            score = 100.0
        elif redaction.total:
            score = clamp(
                noisy_or(finding.score for finding in redaction.findings),
                self._redaction_floor,
                self._redaction_ceiling,
            )
        else:
            score = 0.0

        report = LayerReport(
            name=self.name,
            stage="output",
            score=score,
            weight=1.0,
            findings=findings,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            details={
                "redactions": redaction.total,
                "prompt_containment": round(max_containment, 4),
                "action": "withhold" if blocked else "redact" if redaction.total else "pass",
            },
        )
        return OutputVerdict(
            report=report, safe_text=redaction.text, blocked=blocked, redactions=redaction.counts
        )

    @staticmethod
    def _leak_finding(rule_id: str, description: str, view: str) -> Finding:
        return Finding(
            rule_id=rule_id,
            category=PROMPT_LEAK,
            severity=Severity.CRITICAL,
            description=description,
            score=100.0,
            view=view,
        )
