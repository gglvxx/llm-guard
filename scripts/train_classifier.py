"""Train the intent classifier and report honest (cross-validated) quality metrics.

Usage::

    python -m scripts.train_classifier             # train, evaluate, save the artefact
    python -m scripts.train_classifier --no-save   # evaluate only
    python -m scripts.train_classifier --errors    # also list misclassified samples

The server trains automatically at start-up when no valid artefact exists, so
this script is only needed to inspect the model or after editing the dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from app.config import load_settings
from app.core.ml_classifier import (
    ALGORITHM,
    IntentClassifier,
    cross_validate,
    file_sha256,
    load_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--no-save", action="store_true", help="do not write the model artefact")
    parser.add_argument("--errors", action="store_true", help="list misclassified samples")
    args = parser.parse_args()

    settings = load_settings(args.config)
    dataset_path = settings.resolve_path(settings.ml.dataset_path)
    model_dir = settings.resolve_path(settings.ml.model_dir)

    dataset = load_dataset(dataset_path)
    labels = Counter(sample.label for sample in dataset.samples)
    categories = Counter((sample.label, sample.category) for sample in dataset.samples)

    print(f"Dataset    {dataset_path} (v{dataset.version})")
    print(f"Samples    {len(dataset.samples)}  {dict(labels)}")
    for (label, category), count in sorted(categories.items()):
        print(f"           {label:<12} {category:<26} {count:>3}")

    validation = cross_validate(dataset)
    print(f"\nAlgorithm  {ALGORITHM}")
    print(
        "5-fold stratified cross-validation (every sample predicted by a model that never saw it)"
    )
    print(f"  accuracy   {validation.accuracy:.3f}")
    print(f"  precision  {validation.precision:.3f}")
    print(f"  recall     {validation.recall:.3f}")
    print(f"  F1         {validation.f1:.3f}")

    errors = validation.misclassified(dataset)
    print(f"\nMisclassified held-out samples: {len(errors)} of {len(dataset.samples)}")
    if args.errors:
        for sample, predicted in errors:
            print(f"  [{sample.label} -> {predicted}] ({sample.category}) {sample.text!r}")

    if not args.no_save:
        classifier = IntentClassifier.train(dataset, dataset_sha256=file_sha256(dataset_path))
        classifier.save(model_dir)
        print(f"\nSaved      {model_dir} (sha256 {classifier.metadata.artifact_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
