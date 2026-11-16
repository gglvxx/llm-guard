"""In-memory security telemetry: counters plus a bounded ring buffer of events."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from datetime import UTC, datetime

from app.config import TelemetrySettings
from app.models.schemas import AttackVectorStat, Decision, SecurityEvent, TelemetrySnapshot

_TOP_VECTORS = 6


class TelemetryStore:
    """Thread-safe aggregate metrics for the SOC dashboard.

    Memory is bounded by construction (``deque(maxlen=...)``): an attacker
    flooding the API cannot grow the store. State is per process and resets on
    restart; a production deployment would ship the JSON logs to a SIEM instead.
    """

    def __init__(
        self, settings: TelemetrySettings, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._lock = threading.Lock()
        self._events: deque[SecurityEvent] = deque(maxlen=settings.max_events)
        self._started_at = clock()
        self._decisions: Counter[Decision] = Counter()
        self._vectors: Counter[str] = Counter()
        self._total = 0
        self._unprotected = 0
        self._shadow_detections = 0
        self._redactions = 0
        self._firewall_ms_sum = 0.0
        self._total_ms_sum = 0.0

    def record(self, event: SecurityEvent) -> None:
        with self._lock:
            self._events.append(event)
            self._total += 1
            self._firewall_ms_sum += event.firewall_ms
            self._total_ms_sum += event.total_ms
            self._redactions += event.redaction_count
            if event.enforced:
                self._decisions[event.decision] += 1
            else:
                self._unprotected += 1
                if event.decision is not Decision.ALLOW:
                    self._shadow_detections += 1
            if event.decision is not Decision.ALLOW:
                # De-duplicate per event while keeping order, so ties rank deterministically.
                self._vectors.update(list(dict.fromkeys(event.attack_vectors)))

    def snapshot(self) -> TelemetrySnapshot:
        with self._lock:
            total = self._total
            blocked = self._decisions[Decision.BLOCK]
            recent = list(self._events)[-self._settings.recent_events :]
            return TelemetrySnapshot(
                generated_at=datetime.now(UTC),
                uptime_seconds=round(self._clock() - self._started_at, 3),
                total_requests=total,
                allowed_count=self._decisions[Decision.ALLOW],
                flagged_count=self._decisions[Decision.FLAG],
                blocked_count=blocked,
                unprotected_count=self._unprotected,
                shadow_detections=self._shadow_detections,
                redactions_total=self._redactions,
                block_rate=round(blocked / total, 4) if total else 0.0,
                avg_firewall_ms=round(self._firewall_ms_sum / total, 3) if total else 0.0,
                avg_total_ms=round(self._total_ms_sum / total, 3) if total else 0.0,
                top_attack_vectors=[
                    AttackVectorStat(vector=vector, count=count)
                    for vector, count in self._vectors.most_common(_TOP_VECTORS)
                ],
                recent_events=list(reversed(recent)),
            )

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._decisions.clear()
            self._vectors.clear()
            self._total = self._unprotected = self._shadow_detections = self._redactions = 0
            self._firewall_ms_sum = self._total_ms_sum = 0.0
            self._started_at = self._clock()
