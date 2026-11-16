"""Layer 2 — lightweight ML intent classifier (TF-IDF + Logistic Regression).

Signatures only catch phrasings somebody anticipated. This layer generalises
from labelled examples: word n-grams capture vocabulary and phrasing, character
n-grams add robustness to typos and light obfuscation. Logistic Regression is
used instead of ``LinearSVC`` because it yields a calibrated probability with no
extra calibration step, and training on a few hundred samples takes well under
a second on a laptop CPU.

Persistence is treated as a supply-chain risk: a pickle executes code when it is
loaded, so the artefact is only loaded if its SHA-256 matches the metadata
written at training time, and the model is rebuilt from the JSON dataset
whenever the artefact is missing, stale (dataset or scikit-learn changed) or
altered. See the README for the limits of this check.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import sklearn
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from app.config import MLSettings
from app.core.base import DetectionStage, PipelineContext, StageOutcome
from app.models.schemas import Finding, Severity

logger = logging.getLogger("llm_guard.ml")

Label = Literal["benign", "adversarial"]
POSITIVE_LABEL: Label = "adversarial"
ALGORITHM = "tfidf(word 1-2 + char_wb 3-5) + logistic-regression"
ARTIFACT_NAME = "intent_classifier.joblib"
METADATA_NAME = "intent_classifier.meta.json"
MIN_SAMPLES = 150
CV_FOLDS = 5
RANDOM_STATE = 42


class DatasetError(RuntimeError):
    """Raised when the training dataset is missing or fails validation."""


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=3)
    label: Label
    category: str = Field(min_length=1)


class Dataset(BaseModel):
    """Schema of ``data/dataset.json``; validation doubles as a data-quality gate."""

    model_config = ConfigDict(extra="forbid")

    version: str
    description: str
    samples: list[Sample]

    @model_validator(mode="after")
    def _quality_gate(self) -> Dataset:
        total = len(self.samples)
        if total < MIN_SAMPLES:
            raise ValueError(f"dataset needs at least {MIN_SAMPLES} samples, found {total}")
        positives = sum(1 for sample in self.samples if sample.label == POSITIVE_LABEL)
        ratio = positives / total
        if not 0.4 <= ratio <= 0.6:
            raise ValueError(f"dataset is imbalanced: {ratio:.0%} adversarial (expected 40-60%)")
        texts = [sample.text.strip().lower() for sample in self.samples]
        if len(set(texts)) != total:
            raise ValueError("dataset contains duplicate texts")
        return self


class ModelMetadata(BaseModel):
    """Provenance and quality record stored next to the model artefact."""

    algorithm: str
    trained_at: datetime
    samples: int
    benign: int
    adversarial: int
    dataset_sha256: str
    sklearn_version: str
    cv_folds: int
    cv_accuracy: float
    cv_precision: float
    cv_recall: float
    cv_f1: float
    artifact_sha256: str | None = None


def load_dataset(path: Path) -> Dataset:
    """Read and validate the JSON dataset."""
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetError(f"cannot read dataset {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"dataset {path} is not valid JSON: {exc}") from exc
    try:
        return Dataset.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(f"dataset {path} failed validation:\n{exc}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_to_risk(probability: float, threshold: float) -> float:
    """Map a class probability to a 0-100 risk score.

    Below the decision threshold the classifier believes the prompt is benign
    and contributes nothing (no noise floor on harmless prompts); above it the
    score rises linearly to 100 at probability 1.0.
    """
    if probability <= threshold:
        return 0.0
    return round((probability - threshold) / (1.0 - threshold) * 100.0, 2)


@dataclass(frozen=True, slots=True)
class CrossValidation:
    """Held-out predictions and metrics from stratified k-fold cross-validation."""

    predictions: tuple[str, ...]
    accuracy: float
    precision: float
    recall: float
    f1: float

    def misclassified(self, dataset: Dataset) -> list[tuple[Sample, str]]:
        """Samples whose held-out prediction was wrong: the place to look when improving data."""
        return [
            (sample, predicted)
            for sample, predicted in zip(dataset.samples, self.predictions, strict=True)
            if sample.label != predicted
        ]


def cross_validate(dataset: Dataset) -> CrossValidation:
    """Estimate generalisation: every sample is predicted by a model that never saw it."""
    texts = [sample.text for sample in dataset.samples]
    labels = [sample.label for sample in dataset.samples]
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    held_out = cross_val_predict(_build_pipeline(), texts, labels, cv=folds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, held_out, average="binary", pos_label=POSITIVE_LABEL, zero_division=0
    )
    return CrossValidation(
        predictions=tuple(str(label) for label in held_out),
        accuracy=round(float(accuracy_score(labels, held_out)), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1=round(float(f1), 4),
    )


def _build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=10.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
    )
    return Pipeline([("features", features), ("classifier", classifier)])


class IntentClassifier:
    """Binary benign/adversarial classifier with integrity-checked persistence."""

    def __init__(self, pipeline: Pipeline, metadata: ModelMetadata) -> None:
        self._pipeline = pipeline
        self._metadata = metadata
        classes = [str(label) for label in pipeline.classes_]
        self._positive_index = classes.index(POSITIVE_LABEL)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    # ------------------------------------------------------------------ inference

    def predict_many(self, texts: Sequence[str]) -> list[float]:
        """Return P(adversarial) for each text."""
        if not texts:
            return []
        probabilities = self._pipeline.predict_proba(list(texts))
        return [float(row[self._positive_index]) for row in probabilities]

    def predict_proba(self, text: str) -> float:
        return self.predict_many([text])[0]

    # ------------------------------------------------------------------ training

    @classmethod
    def train(cls, dataset: Dataset, *, dataset_sha256: str = "") -> IntentClassifier:
        """Cross-validate for honest metrics, then fit on the full dataset."""
        texts = [sample.text for sample in dataset.samples]
        labels = [sample.label for sample in dataset.samples]
        validation = cross_validate(dataset)

        pipeline = _build_pipeline()
        pipeline.fit(texts, labels)

        positives = labels.count(POSITIVE_LABEL)
        metadata = ModelMetadata(
            algorithm=ALGORITHM,
            trained_at=datetime.now(UTC),
            samples=len(labels),
            benign=len(labels) - positives,
            adversarial=positives,
            dataset_sha256=dataset_sha256,
            sklearn_version=sklearn.__version__,
            cv_folds=CV_FOLDS,
            cv_accuracy=validation.accuracy,
            cv_precision=validation.precision,
            cv_recall=validation.recall,
            cv_f1=validation.f1,
        )
        return cls(pipeline, metadata)

    # ------------------------------------------------------------------ persistence

    def save(self, model_dir: Path) -> None:
        """Atomically write the artefact and its metadata (including the artefact hash)."""
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact = model_dir / ARTIFACT_NAME
        handle, temp_name = tempfile.mkstemp(dir=model_dir, suffix=".tmp")
        os.close(handle)
        try:
            joblib.dump(self._pipeline, temp_name)
            os.replace(temp_name, artifact)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        self._metadata = self._metadata.model_copy(
            update={"artifact_sha256": file_sha256(artifact)}
        )
        (model_dir / METADATA_NAME).write_text(
            self._metadata.model_dump_json(indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, model_dir: Path, *, expected_dataset_sha256: str) -> IntentClassifier | None:
        """Load a persisted model, or return ``None`` if it cannot be trusted as-is."""
        artifact = model_dir / ARTIFACT_NAME
        metadata_file = model_dir / METADATA_NAME
        if not artifact.is_file() or not metadata_file.is_file():
            return None
        try:
            metadata = ModelMetadata.model_validate_json(metadata_file.read_text("utf-8"))
        except (OSError, ValidationError):
            logger.warning("model metadata unreadable; retraining")
            return None

        if metadata.dataset_sha256 != expected_dataset_sha256:
            logger.info("dataset changed since last training; retraining")
            return None
        if metadata.sklearn_version != sklearn.__version__:
            logger.info("scikit-learn version changed; retraining")
            return None
        if metadata.artifact_sha256 != file_sha256(artifact):
            logger.warning("model artefact failed integrity check; discarding and retraining")
            return None

        pipeline = cast(Pipeline, joblib.load(artifact))
        return cls(pipeline, metadata)

    @classmethod
    def load_or_train(
        cls, dataset_path: Path, model_dir: Path, *, persist: bool = True
    ) -> IntentClassifier:
        """Return a ready classifier, reusing a verified artefact when possible."""
        dataset_hash = file_sha256(dataset_path) if dataset_path.is_file() else ""
        if persist:
            cached = cls.load(model_dir, expected_dataset_sha256=dataset_hash)
            if cached is not None:
                return cached

        classifier = cls.train(load_dataset(dataset_path), dataset_sha256=dataset_hash)
        logger.info(
            "intent classifier trained",
            extra={
                "samples": classifier.metadata.samples,
                "cv_f1": classifier.metadata.cv_f1,
                "cv_accuracy": classifier.metadata.cv_accuracy,
            },
        )
        if persist:
            try:
                classifier.save(model_dir)
            except OSError:
                logger.warning("could not persist the model; continuing in memory", exc_info=True)
        return classifier


class MLStage(DetectionStage):
    """Pipeline stage that scores every non-speculative view and keeps the worst one."""

    name = "ml_classifier"

    def __init__(self, classifier: IntentClassifier, settings: MLSettings, *, weight: float):
        super().__init__(weight=weight, enabled=settings.enabled)
        self._classifier = classifier
        self._threshold = settings.decision_threshold

    def _analyze(self, ctx: PipelineContext) -> StageOutcome:
        views = list(ctx.iter_views(include_speculative=False))
        probabilities = self._classifier.predict_many([view.text for view in views])
        probability, view = max(zip(probabilities, views, strict=True), key=lambda pair: pair[0])
        score = probability_to_risk(probability, self._threshold)

        findings: list[Finding] = []
        if score > 0.0:
            severity = (
                Severity.HIGH if probability >= 0.9
                else Severity.MEDIUM if probability >= 0.7
                else Severity.LOW
            )  # fmt: skip
            findings.append(
                Finding(
                    rule_id="ML-INTENT",
                    category="ml_adversarial_intent",
                    severity=severity,
                    description=f"Classifier estimates P(adversarial) = {probability:.2f}",
                    score=score,
                    view=view.name,
                )
            )
        return StageOutcome(
            findings=findings,
            score=score,
            details={
                "probability": round(probability, 4),
                "threshold": self._threshold,
                "view": view.name,
            },
        )
