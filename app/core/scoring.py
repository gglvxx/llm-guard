"""Small, pure scoring helpers shared by every detection layer."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def noisy_or(scores: Iterable[float]) -> float:
    """Combine independent 0-100 risk signals: ``100 * (1 - Π(1 - s_i / 100))``.

    Each score is read as the probability that the signal indicates an attack.
    Unlike an average, the result is monotonic: extra evidence can only raise
    the risk and one strong signal is never diluted by silent detectors.
    """
    survival = 1.0
    for score in scores:
        survival *= 1.0 - clamp(score) / 100.0
    return round(100.0 * (1.0 - survival), 2)


def shannon_entropy(text: str) -> float:
    """Shannon entropy of ``text`` in bits per character (0.0 for empty input)."""
    if not text:
        return 0.0
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in Counter(text).values())
