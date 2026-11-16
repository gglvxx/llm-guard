"""Layer 1 — heuristic and signature engine (regex signatures + Shannon entropy)."""

from __future__ import annotations

from collections.abc import Sequence

from app.config import HeuristicSettings
from app.core.base import DetectionStage, PipelineContext, StageOutcome, snippet
from app.core.scoring import clamp, shannon_entropy
from app.core.signatures import SignatureRule
from app.models.schemas import Finding, Severity

_EDGE_PUNCTUATION = "\"'`()[]{}<>.,;:!?"
_URL_PREFIXES = ("http://", "https://", "www.")


class HeuristicStage(DetectionStage):
    """Scans every text view with the signature rule set and flags high-entropy blobs.

    A rule contributes once, whatever the number of matches. A hit that only
    appears in a de-obfuscated view (Base64, ROT13, leetspeak ...) earns an
    *evasion bonus*: hiding an instruction is itself evidence of intent.
    """

    name = "heuristic"

    def __init__(
        self,
        rules: Sequence[SignatureRule],
        settings: HeuristicSettings,
        *,
        weight: float,
    ) -> None:
        super().__init__(weight=weight, enabled=settings.enabled)
        self._rules = tuple(rules)
        self._settings = settings

    @property
    def rules(self) -> tuple[SignatureRule, ...]:
        return self._rules

    def _analyze(self, ctx: PipelineContext) -> StageOutcome:
        hits: dict[str, Finding] = {}
        for view in ctx.iter_views():
            for rule in self._rules:
                if rule.id in hits:
                    # Views are ordered canonical-first: the plainest match wins, so a
                    # clear-text attack never collects an undeserved evasion bonus.
                    continue
                match = rule.pattern.search(view.text)
                if match is None:
                    continue
                score = self._settings.severity_scores[rule.severity]
                if view.obfuscated:
                    score += self._settings.evasion_bonus
                hits[rule.id] = Finding(
                    rule_id=rule.id,
                    category=rule.category,
                    severity=rule.severity,
                    description=rule.description,
                    score=clamp(score),
                    view=view.name,
                    evidence=snippet(view.text, match.start(), match.end()),
                )

        findings = list(hits.values())
        entropy_finding = self._entropy_finding(ctx)
        if entropy_finding is not None:
            findings.append(entropy_finding)

        return StageOutcome(
            findings=findings,
            details={
                "rules_evaluated": len(self._rules),
                "views_scanned": len(ctx.views),
                "rules_matched": len(hits),
            },
        )

    def _entropy_finding(self, ctx: PipelineContext) -> Finding | None:
        """Flag long, high-entropy tokens that could not be decoded into text.

        Natural language sits around 4 bits/char and hex tops out at 4.0; random
        Base64 or ciphertext is well above 4.5. Anything the normaliser already
        decoded is skipped (it is scored there), as are URLs.
        """
        suspicious: list[tuple[float, str]] = []
        for raw_token in ctx.canonical.split():
            token = raw_token.strip(_EDGE_PUNCTUATION)
            if len(token) < self._settings.entropy_min_length:
                continue
            if token.lower().startswith(_URL_PREFIXES) or "://" in token:
                continue
            if any(source in token or token in source for source in ctx.decoded_sources):
                continue
            entropy = shannon_entropy(token)
            if entropy >= self._settings.entropy_threshold:
                suspicious.append((entropy, token))

        if not suspicious:
            return None
        entropy, token = max(suspicious)
        return Finding(
            rule_id="HEU-ENTROPY",
            category="obfuscation",
            severity=Severity.LOW,
            description=(
                f"{len(suspicious)} high-entropy token(s) that could not be decoded "
                f"(max {entropy:.2f} bits/char): possible encrypted or encoded payload"
            ),
            score=self._settings.severity_scores[Severity.LOW],
            evidence=token[:48] + ("…" if len(token) > 48 else ""),
        )
