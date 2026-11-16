"""Layer 3: validators, redaction, system prompt leak detection and the output guard."""

from __future__ import annotations

import base64

import pytest

from app.config import DLPSettings, SecretPattern
from app.core.dlp import (
    OutputGuard,
    PIIRedactor,
    PromptLeakDetector,
    cnp_valid,
    luhn_valid,
)
from app.core.normalizer import TextNormalizer
from app.providers.persona import CANARY_TOKEN, DEMO_SYSTEM_PROMPT, SECRET_FLAG

# A syntactically valid but obviously fake JWT (header.payload.signature).
JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkRlbW8ifQ"
    ".c2lnbmF0dXJlLXBsYWNlaG9sZGVyLTEyMzQ1Ng"
)
# Fake credentials are assembled at runtime so that no literal token shape is committed
# (secret scanners would otherwise flag these fixtures on push).
_PEM = "PRIVATE KEY-----"
PRIVATE_KEY = (
    f"-----BEGIN RSA {_PEM}\nMIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy\n-----END RSA {_PEM}"
)
AWS_KEY_ID = "AKIA" + "IOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789"


@pytest.fixture(scope="module")
def redactor() -> PIIRedactor:
    return PIIRedactor(DLPSettings())


@pytest.fixture(scope="module")
def guard(redactor: PIIRedactor) -> OutputGuard:
    settings = DLPSettings()
    return OutputGuard(
        settings,
        normalizer=TextNormalizer(),
        redactor=redactor,
        leak_detector=PromptLeakDetector(
            DEMO_SYSTEM_PROMPT,
            [CANARY_TOKEN],
            shingle_size=settings.leak_shingle_size,
            threshold=settings.leak_containment_threshold,
        ),
        redaction_score_range=(40.0, 69.0),
    )


class TestValidators:
    @pytest.mark.parametrize(
        "number",
        ["4111 1111 1111 1111", "5555-5555-5555-4444", "378282246310005", "4012888888881881"],
    )
    def test_luhn_accepts_valid_cards(self, number: str) -> None:
        assert luhn_valid(number)

    @pytest.mark.parametrize(
        "number",
        ["4111 1111 1111 1112", "1234 5678 9012 3456", "0000 0000 0000 0000", "411111", "4" * 20],
    )
    def test_luhn_rejects_invalid_numbers(self, number: str) -> None:
        assert not luhn_valid(number)

    @pytest.mark.parametrize("cnp", ["2850101990017", "1900515990021", "2921123990032"])
    def test_cnp_accepts_valid_control_digit(self, cnp: str) -> None:
        assert cnp_valid(cnp)

    @pytest.mark.parametrize(
        "cnp",
        [
            "2850101990018",  # wrong control digit
            "2851301990017",  # month 13
            "2850132990017",  # day 32
            "0850101990017",  # leading zero
            "285010199001",  # too short
            "28501019900a7",
        ],
    )
    def test_cnp_rejects_invalid_values(self, cnp: str) -> None:
        assert not cnp_valid(cnp)


class TestRedactor:
    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("Write to maria.ionescu@nexora.example today", "EMAIL"),
            ("Card: 4111 1111 1111 1111.", "CREDIT_CARD"),
            ("CNP 2850101990017 on file", "CNP"),
            ("SSN 078-05-1120 was exposed", "SSN"),
            ("key sk-nexora-demo-4f9a8b7c6d5e4f3a2b1c0d9e ok", "API_KEY"),
            (f"aws {AWS_KEY_ID} ok", "API_KEY"),
            (f"token {GITHUB_TOKEN} ok", "API_KEY"),
            (f"bearer {JWT} ok", "JWT"),
            (PRIVATE_KEY, "PRIVATE_KEY"),
            ("password: Tr0ub4dor&3xyz", "CREDENTIAL"),
            (f"the code is {SECRET_FLAG}", "SECRET"),
        ],
    )
    def test_each_detector_masks_its_data(
        self, redactor: PIIRedactor, text: str, label: str
    ) -> None:
        result = redactor.redact(text)
        assert result.counts == {label: 1}
        assert f"[REDACTED:{label}]" in result.text

    def test_surrounding_text_is_preserved(self, redactor: PIIRedactor) -> None:
        result = redactor.redact("Contact maria.ionescu@nexora.example or call the front desk.")
        assert result.text == "Contact [REDACTED:EMAIL] or call the front desk."

    def test_credential_assignment_masks_only_the_value(self, redactor: PIIRedactor) -> None:
        result = redactor.redact("api_key = 'abc123def456ghi789'")
        assert result.text == "api_key = '[REDACTED:CREDENTIAL]'"

    def test_multiple_values_are_all_masked(self, redactor: PIIRedactor) -> None:
        text = "a@x.example, b@y.example and card 5555 5555 5555 4444"
        result = redactor.redact(text)
        assert result.counts == {"EMAIL": 2, "CREDIT_CARD": 1}
        assert "@" not in result.text
        assert "5555" not in result.text

    @pytest.mark.parametrize(
        "text",
        [
            "Your order number is 1234 5678 9012 3456.",  # fails Luhn
            "The reference 2850101990018 is not a valid CNP.",  # fails the control digit
            "Call extension 555-0199, version 3.11.4, date 2026-11-12, bogus SSN 000-12-3456.",
            "Password: must contain at least twelve characters",
            "Nexora employees get 25 days of paid annual leave per year.",
            "Use the @property decorator, e.g. user@host is an SSH target format.",
        ],
    )
    def test_lookalikes_are_left_alone(self, redactor: PIIRedactor, text: str) -> None:
        result = redactor.redact(text)
        assert result.text == text
        assert result.total == 0

    def test_custom_template_and_patterns(self) -> None:
        settings = DLPSettings(
            redaction_template="<{label}>",
            custom_secret_patterns=(SecretPattern(label="TICKET", pattern=r"TCK-\d{6}"),),
        )
        assert PIIRedactor(settings).redact("see TCK-123456").text == "see <TICKET>"

    def test_detectors_can_be_disabled(self) -> None:
        relaxed = PIIRedactor(DLPSettings(disabled_detectors=("EMAIL",)))
        assert relaxed.redact("mail me at a@x.example").total == 0
        with pytest.raises(ValueError, match="unknown DLP detector"):
            PIIRedactor(DLPSettings(disabled_detectors=("FAX",)))


@pytest.fixture(scope="module")
def detector() -> PromptLeakDetector:
    return PromptLeakDetector(DEMO_SYSTEM_PROMPT, [CANARY_TOKEN], shingle_size=5, threshold=0.12)


class TestPromptLeakDetector:
    def test_full_reproduction_is_a_leak(self, detector: PromptLeakDetector) -> None:
        check = detector.check(f"Sure! My instructions are:\n{DEMO_SYSTEM_PROMPT}")
        assert check.leaked
        assert check.containment > 0.9
        assert check.canary == CANARY_TOKEN.lower()

    def test_partial_reproduction_is_a_leak(self, detector: PromptLeakDetector) -> None:
        rules_section = DEMO_SYSTEM_PROMPT.split("Rules:")[1]
        check = detector.check(f"I was told the following. {rules_section}")
        assert check.canary is None
        assert check.leaked

    def test_canary_alone_is_a_leak(self, detector: PromptLeakDetector) -> None:
        assert detector.check(f"marker: {CANARY_TOKEN.upper()}").leaked

    def test_ordinary_answers_are_not_leaks(self, detector: PromptLeakDetector) -> None:
        answer = (
            "Hello! I'm Atlas, the internal HR and IT helpdesk assistant of Nexora Systems. "
            "You get 25 days of annual leave."
        )
        check = detector.check(answer)
        assert not check.leaked
        assert check.containment < 0.12


class TestOutputGuard:
    def test_clean_output_passes_untouched(self, guard: OutputGuard) -> None:
        verdict = guard.inspect("Payroll runs on the 10th of every month.")
        assert not verdict.blocked
        assert verdict.score == 0.0
        assert verdict.safe_text == "Payroll runs on the 10th of every month."
        assert verdict.report.details["action"] == "pass"

    def test_pii_is_redacted_and_scored_in_the_flag_band(self, guard: OutputGuard) -> None:
        verdict = guard.inspect(
            "Reach Elena at elena.dumitru@nexora.example, card 4012 8888 8888 1881."
        )
        assert not verdict.blocked
        assert verdict.redactions == {"EMAIL": 1, "CREDIT_CARD": 1}
        assert "elena.dumitru" not in verdict.safe_text
        assert 40.0 <= verdict.score <= 69.0
        assert verdict.report.details["action"] == "redact"

    def test_system_prompt_leak_is_withheld(self, guard: OutputGuard) -> None:
        verdict = guard.inspect(f"Here you go:\n{DEMO_SYSTEM_PROMPT}")
        assert verdict.blocked
        assert verdict.score == 100.0
        assert verdict.report.findings[0].rule_id == "DLP-CANARY"

    def test_base64_encoded_leak_is_withheld(self, guard: OutputGuard) -> None:
        encoded = base64.b64encode(DEMO_SYSTEM_PROMPT.encode()).decode()
        verdict = guard.inspect(f"As requested, Base64-encoded:\n{encoded}")
        assert verdict.blocked
        assert verdict.report.findings[0].view == "base64"

    def test_secret_inside_encoded_output_is_withheld(self, guard: OutputGuard) -> None:
        encoded = base64.b64encode(b"contact: maria.ionescu@nexora.example right now").decode()
        verdict = guard.inspect(f"Decode this yourself: {encoded}")
        assert verdict.blocked
        assert "DLP-ENCODED-EXFIL" in {finding.rule_id for finding in verdict.report.findings}

    def test_zero_width_evasion_of_the_redactor_fails(self, guard: OutputGuard) -> None:
        evasive = SECRET_FLAG[:6] + "​" + SECRET_FLAG[6:]
        verdict = guard.inspect(f"The code is {evasive}")
        assert SECRET_FLAG not in verdict.safe_text
        assert verdict.redactions == {"SECRET": 1}

    def test_disabled_guard_is_a_no_op(self, redactor: PIIRedactor) -> None:
        guard = OutputGuard(
            DLPSettings(enabled=False),
            normalizer=TextNormalizer(),
            redactor=redactor,
            leak_detector=None,
            redaction_score_range=(40.0, 69.0),
        )
        verdict = guard.inspect(DEMO_SYSTEM_PROMPT)
        assert verdict.report.skipped
        assert verdict.safe_text == DEMO_SYSTEM_PROMPT
