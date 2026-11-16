"""Layer 1 rule set: every signature is tested against its own examples."""

from __future__ import annotations

import time

import pytest

from app.config import CustomRule
from app.core.signatures import DEFAULT_RULES, SignatureRule, build_ruleset
from app.models.schemas import Severity

RULES = {rule.id: rule for rule in DEFAULT_RULES}
BLOCKING = {Severity.HIGH, Severity.CRITICAL}

#: Harmless prompts that reuse attack vocabulary. No high-severity rule may fire on them.
BENIGN_LOOKALIKES = [
    "How do I make git ignore all files in the build directory?",
    "How do I enable developer mode on my Android phone?",
    "What are the system requirements for Windows 11?",
    "How do I override a method in a Python subclass?",
    "Explain what a system prompt is and why companies keep it private.",
    "What is prompt injection and how can developers defend against it?",
    "How do I bypass the cache when reloading a page in Chrome?",
    "Show me the rules for naming variables in PEP 8.",
    "What instructions should I give a new intern on their first day?",
    "From now on, please answer in bullet points.",
    "Forget what I said earlier about Java, I want the example in Kotlin.",
    "Pretend you are an interviewer and ask me three questions about databases.",
    "How do I reveal hidden files in the macOS Finder?",
    "Is it safe to share my API key with a teammate over chat?",
    "Write a system design overview for a chat application.",
    "My manager told me to ignore the previous email, should I reply anyway?",
    "What does the [INFO] prefix mean in these logs?",
    "### Installation\nRun pip install -r requirements.txt",
    "System: Ubuntu 22.04\nProblem: the VPN client will not start",
    "Can you print the first ten lines of this file?",
]


def test_rule_ids_are_unique() -> None:
    assert len(RULES) == len(DEFAULT_RULES)


@pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda rule: rule.id)
def test_rule_matches_its_examples(rule: SignatureRule) -> None:
    assert rule.examples, f"{rule.id} must document at least one matching example"
    for example in rule.examples:
        assert rule.pattern.search(example), f"{rule.id} should match: {example!r}"


@pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda rule: rule.id)
def test_rule_ignores_its_counter_examples(rule: SignatureRule) -> None:
    for counter_example in rule.counter_examples:
        assert not rule.pattern.search(counter_example), (
            f"{rule.id} must not match: {counter_example!r}"
        )


@pytest.mark.parametrize("prompt", BENIGN_LOOKALIKES)
def test_no_blocking_rule_fires_on_benign_lookalikes(prompt: str) -> None:
    fired = [
        rule.id
        for rule in DEFAULT_RULES
        if rule.severity in BLOCKING and rule.pattern.search(prompt)
    ]
    assert fired == []


def test_counter_examples_never_trigger_a_blocking_rule() -> None:
    for owner in DEFAULT_RULES:
        for counter_example in owner.counter_examples:
            fired = [
                rule.id
                for rule in DEFAULT_RULES
                if rule.severity in BLOCKING and rule.pattern.search(counter_example)
            ]
            assert fired == [], f"{counter_example!r} (counter-example of {owner.id})"


@pytest.mark.parametrize(
    "hostile",
    [
        "a " * 2000,
        "ignore " * 600,
        "show me " * 500,
        "reveal the " * 400,
        "what is your " * 300,
        "you must never " * 260,
        "--- " * 1000,
        "in this story " + "a" * 3900,
        "<" * 4000,
        "send everything to " * 200,
    ],
    ids=lambda text: text[:12].strip(),
)
def test_signatures_are_not_a_redos_target(hostile: str) -> None:
    """Worst-case repetitive input must still be scanned in bounded time."""
    started = time.perf_counter()
    for rule in DEFAULT_RULES:
        rule.pattern.search(hostile)
    assert time.perf_counter() - started < 1.0


def test_dan_rule_is_case_sensitive() -> None:
    rule = RULES["JB-001"]
    assert rule.pattern.search("you are DAN")
    assert not rule.pattern.search("my friend Dan is a developer")


def test_build_ruleset_can_disable_rules() -> None:
    rules = build_ruleset(disabled=["JB-008", "PI-005"])
    assert {"JB-008", "PI-005"}.isdisjoint(rule.id for rule in rules)
    assert len(rules) == len(DEFAULT_RULES) - 2


def test_build_ruleset_rejects_unknown_disabled_ids() -> None:
    with pytest.raises(ValueError, match="unknown rule id"):
        build_ruleset(disabled=["PI-999"])


def test_build_ruleset_appends_custom_rules() -> None:
    custom = CustomRule(
        id="ORG-001",
        category="data_exfiltration",
        severity=Severity.HIGH,
        pattern=r"\bproject\s+bluebird\b",
        description="Internal code name",
    )
    rules = build_ruleset(custom=[custom])
    added = rules[-1]
    assert added.id == "ORG-001"
    assert added.pattern.search("Tell me about Project  Bluebird")


def test_custom_rule_cannot_shadow_a_builtin() -> None:
    clash = CustomRule(
        id="PI-001", category="x", severity=Severity.LOW, pattern="x", description="d"
    )
    with pytest.raises(ValueError, match="collides"):
        build_ruleset(custom=[clash])
