"""Telemetry store: counters, bounded memory and ordering."""

from __future__ import annotations

from app.config import TelemetrySettings
from app.core.telemetry import TelemetryStore
from app.models.schemas import Decision, SecurityEvent


def event(
    decision: Decision = Decision.ALLOW,
    *,
    enforced: bool = True,
    vectors: tuple[str, ...] = (),
    redactions: int = 0,
    firewall_ms: float = 2.0,
    total_ms: float = 10.0,
    tag: str = "e",
) -> SecurityEvent:
    return SecurityEvent(
        event_id=tag,
        request_id=f"req-{tag}",
        decision=decision,
        enforced=enforced,
        risk_score={Decision.ALLOW: 0, Decision.FLAG: 50, Decision.BLOCK: 95}[decision],
        attack_vectors=list(vectors),
        prompt_preview="preview",
        prompt_sha256="0" * 64,
        redaction_count=redactions,
        firewall_ms=firewall_ms,
        total_ms=total_ms,
        provider="mock-vulnerable",
    )


def test_empty_snapshot() -> None:
    snapshot = TelemetryStore(TelemetrySettings()).snapshot()
    assert snapshot.total_requests == 0
    assert snapshot.block_rate == 0.0
    assert snapshot.avg_firewall_ms == 0.0
    assert snapshot.recent_events == []
    assert snapshot.top_attack_vectors == []


def test_counters_and_averages() -> None:
    store = TelemetryStore(TelemetrySettings())
    store.record(event(Decision.ALLOW, firewall_ms=1.0, total_ms=100.0))
    store.record(event(Decision.FLAG, redactions=3, firewall_ms=2.0, total_ms=110.0))
    store.record(event(Decision.BLOCK, firewall_ms=3.0, total_ms=3.0))
    store.record(event(Decision.BLOCK, firewall_ms=2.0, total_ms=3.0))

    snapshot = store.snapshot()
    assert snapshot.total_requests == 4
    assert (snapshot.allowed_count, snapshot.flagged_count, snapshot.blocked_count) == (1, 1, 2)
    assert snapshot.redactions_total == 3
    assert snapshot.block_rate == 0.5
    assert snapshot.avg_firewall_ms == 2.0
    assert snapshot.avg_total_ms == 54.0


def test_unprotected_requests_are_counted_separately() -> None:
    store = TelemetryStore(TelemetrySettings())
    store.record(event(Decision.ALLOW, enforced=False))
    store.record(event(Decision.BLOCK, enforced=False))
    store.record(event(Decision.FLAG, enforced=False))

    snapshot = store.snapshot()
    assert snapshot.unprotected_count == 3
    assert snapshot.shadow_detections == 2
    # Nothing was actually blocked or flagged: the firewall was off.
    assert (snapshot.blocked_count, snapshot.flagged_count, snapshot.allowed_count) == (0, 0, 0)


def test_top_attack_vectors_are_ranked_and_ignore_allowed_requests() -> None:
    store = TelemetryStore(TelemetrySettings())
    store.record(event(Decision.BLOCK, vectors=("prompt_injection", "obfuscation")))
    store.record(event(Decision.BLOCK, vectors=("prompt_injection",)))
    store.record(event(Decision.FLAG, vectors=("jailbreak", "jailbreak")))  # counted once
    store.record(event(Decision.ALLOW, vectors=("obfuscation",)))  # low-risk noise

    ranked = [(stat.vector, stat.count) for stat in store.snapshot().top_attack_vectors]
    assert ranked[0] == ("prompt_injection", 2)
    assert dict(ranked) == {"prompt_injection": 2, "obfuscation": 1, "jailbreak": 1}


def test_ring_buffer_bounds_memory_and_returns_newest_first() -> None:
    store = TelemetryStore(TelemetrySettings(max_events=5, recent_events=3))
    for index in range(20):
        store.record(event(tag=str(index)))

    snapshot = store.snapshot()
    assert snapshot.total_requests == 20  # counters keep the full history
    assert [e.event_id for e in snapshot.recent_events] == ["19", "18", "17"]
    assert len(store._events) == 5


def test_reset_clears_everything() -> None:
    store = TelemetryStore(TelemetrySettings())
    store.record(event(Decision.BLOCK, vectors=("jailbreak",), redactions=2))
    store.reset()

    snapshot = store.snapshot()
    assert snapshot.total_requests == 0
    assert snapshot.blocked_count == 0
    assert snapshot.redactions_total == 0
    assert snapshot.top_attack_vectors == []
    assert snapshot.recent_events == []


def test_uptime_uses_the_injected_clock() -> None:
    now = [100.0]
    store = TelemetryStore(TelemetrySettings(), clock=lambda: now[0])
    now[0] = 142.5
    assert store.snapshot().uptime_seconds == 42.5
