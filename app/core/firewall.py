"""The firewall facade: wires the detection chain, the policy and the output guard."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.config import Settings
from app.core.base import DetectionStage, PipelineContext
from app.core.dlp import OutputGuard, OutputVerdict, PIIRedactor, PromptLeakDetector
from app.core.heuristics import HeuristicStage
from app.core.ml_classifier import IntentClassifier, MLStage, ModelMetadata
from app.core.normalizer import NormalizerStage, TextNormalizer
from app.core.policy import PolicyEngine, PromptSanitizer, RiskAggregator
from app.core.signatures import build_ruleset
from app.models.schemas import Decision, LayerReport

logger = logging.getLogger("llm_guard.firewall")


@dataclass(frozen=True, slots=True)
class InputVerdict:
    """Outcome of the input pipeline for one prompt."""

    risk_score: float
    decision: Decision
    reports: tuple[LayerReport, ...]
    #: Normalised prompt (invisible characters removed, homoglyphs folded).
    canonical: str


@dataclass(frozen=True, slots=True)
class ForwardedPrompt:
    """What is actually sent to the LLM when the firewall is enforcing."""

    text: str
    #: ``True`` when the defensive wrapper was applied (FLAG decisions).
    sanitized: bool
    #: The fenced user text on its own (only set when ``sanitized``).
    untrusted_body: str | None = None


class Firewall:
    """Facade over the multi-layer pipeline.

    Input side — Chain of Responsibility::

        NormalizerStage -> HeuristicStage -> MLStage -> RiskAggregator -> PolicyEngine

    Output side: :class:`~app.core.dlp.OutputGuard` (DLP + prompt-leak detection).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        stages: Sequence[DetectionStage],
        output_guard: OutputGuard,
        sanitizer: PromptSanitizer,
        redactor: PIIRedactor,
        rules_loaded: int,
        model_metadata: ModelMetadata | None,
    ) -> None:
        self._settings = settings
        self._stages = tuple(stages)
        self._output_guard = output_guard
        self._sanitizer = sanitizer
        self._redactor = redactor
        self._aggregator = RiskAggregator()
        self._policy = PolicyEngine(settings.policy.thresholds)
        self.rules_loaded = rules_loaded
        self.model_metadata = model_metadata

    # ------------------------------------------------------------------ construction

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        protected_prompt: str = "",
        canaries: Sequence[str] = (),
        classifier: IntentClassifier | None = None,
    ) -> Firewall:
        """Assemble the default pipeline from settings.

        ``protected_prompt`` and ``canaries`` describe the secret the output
        guard must keep inside the system (the provider's system prompt).
        """
        weights = settings.policy.weights
        thresholds = settings.policy.thresholds
        normalizer = TextNormalizer(settings.normalizer)
        rules = build_ruleset(
            disabled=settings.heuristics.disabled_rules,
            custom=settings.heuristics.custom_rules,
        )

        stages: list[DetectionStage] = [
            NormalizerStage(
                normalizer, weight=weights.normalizer, enabled=settings.normalizer.enabled
            ),
            HeuristicStage(rules, settings.heuristics, weight=weights.heuristic),
        ]
        model_metadata: ModelMetadata | None = None
        if settings.ml.enabled:
            if classifier is None:
                classifier = IntentClassifier.load_or_train(
                    settings.resolve_path(settings.ml.dataset_path),
                    settings.resolve_path(settings.ml.model_dir),
                    persist=settings.ml.persist_model,
                )
            model_metadata = classifier.metadata
            stages.append(MLStage(classifier, settings.ml, weight=weights.ml_classifier))

        redactor = PIIRedactor(settings.dlp)
        leak_detector = (
            PromptLeakDetector(
                protected_prompt,
                canaries,
                shingle_size=settings.dlp.leak_shingle_size,
                threshold=settings.dlp.leak_containment_threshold,
            )
            if protected_prompt or canaries
            else None
        )
        output_guard = OutputGuard(
            settings.dlp,
            normalizer=normalizer,
            redactor=redactor,
            leak_detector=leak_detector,
            redaction_score_range=(float(thresholds.flag), float(thresholds.block - 1)),
        )
        return cls(
            settings,
            stages=stages,
            output_guard=output_guard,
            sanitizer=PromptSanitizer(rules),
            redactor=redactor,
            rules_loaded=len(rules),
            model_metadata=model_metadata,
        )

    # ------------------------------------------------------------------ input side

    def inspect_input(self, text: str) -> InputVerdict:
        """Run the detection chain and return the policy verdict for ``text``."""
        ctx = PipelineContext(raw_text=text)
        fail_mode = self._settings.app.fail_mode
        block_threshold = self._settings.policy.thresholds.block

        for stage in self._stages:
            if ctx.halted:
                ctx.reports.append(
                    LayerReport(
                        name=stage.name, stage="input", score=0.0, weight=stage.weight, skipped=True
                    )
                )
                continue
            report = stage.run(ctx, fail_mode=fail_mode)
            ctx.reports.append(report)
            if (report.error is not None and fail_mode == "closed") or (
                self._settings.policy.short_circuit
                and self._aggregator.aggregate(ctx.reports) >= block_threshold
            ):
                ctx.halted = True

        risk = self._aggregator.aggregate(ctx.reports)
        return InputVerdict(
            risk_score=risk,
            decision=self._policy.decide(risk),
            reports=tuple(ctx.reports),
            canonical=ctx.canonical,
        )

    def prepare_prompt(self, verdict: InputVerdict) -> ForwardedPrompt:
        """Return the prompt to forward for an ALLOW or FLAG verdict.

        ALLOW forwards the normalised text (hidden characters never reach the
        model); FLAG additionally fences it in the defensive wrapper.
        """
        if verdict.decision is Decision.FLAG:
            guarded = self._sanitizer.wrap(verdict.canonical)
            return ForwardedPrompt(text=guarded.text, sanitized=True, untrusted_body=guarded.body)
        return ForwardedPrompt(text=verdict.canonical, sanitized=False)

    # ------------------------------------------------------------------ output side

    def inspect_output(self, text: str) -> OutputVerdict:
        """Scan a model response; fail closed if the guard itself crashes."""
        try:
            return self._output_guard.inspect(text)
        except Exception:
            logger.exception("output guard crashed")
            if self._settings.app.fail_mode == "open":
                report = LayerReport(
                    name=OutputGuard.name,
                    stage="output",
                    score=0.0,
                    weight=1.0,
                    skipped=True,
                    error="output guard failed; skipped (fail-open)",
                )
                return OutputVerdict(report=report, safe_text=text, blocked=False)
            report = LayerReport(
                name=OutputGuard.name,
                stage="output",
                score=100.0,
                weight=1.0,
                error="output guard failed; response withheld (fail-closed)",
            )
            return OutputVerdict(report=report, safe_text="", blocked=True)

    # ------------------------------------------------------------------ helpers

    def decide(self, risk_score: float) -> Decision:
        return self._policy.decide(risk_score)

    def scrub(self, text: str) -> str:
        """Mask PII/secrets in ``text`` (used before prompts are logged or stored)."""
        return self._redactor.redact(text).text
