"""Deterministic, deliberately vulnerable LLM simulator.

``MockVulnerableProvider`` behaves like a naive assistant that trusts whatever
the user says: it obeys instruction overrides, dumps its system prompt when
asked, falls for jailbreak personas and over-shares employee data. That makes
the *Firewall OFF vs. Firewall ON* comparison reproducible offline, without a
GPU and without the randomness of a real model.

Two simplifications are worth knowing about:

* A real LLM reads straight through Base64, leetspeak or homoglyphs. The
  simulator emulates that by reusing the project's ``TextNormalizer``.
* When the firewall fences a flagged prompt (``LLMRequest.hardened``), the
  simulator honours the fence and refuses embedded instructions. A real model
  follows such a wrapper most of the time, not always: the output guard exists
  for the cases where it does not.
"""

from __future__ import annotations

import asyncio
import base64
import re

from app.core.normalizer import TextNormalizer
from app.providers.base import BaseLLMProvider, LLMRequest, LLMResult
from app.providers.persona import (
    DEMO_SYSTEM_PROMPT,
    EMPLOYEES,
    SECRET_FLAG,
    SERVICE_API_KEY,
)

_I = re.IGNORECASE

# Intentionally looser than the firewall's signatures: the "model" understands
# paraphrases that no regex rule anticipated, which is what lets some attacks
# slip past the input layers and exercise the output guard.
_JAILBREAK = re.compile(
    r"(?-i:\bDAN\b)|do anything now|you are (?:now )?in (?:\w+ )?mode"
    r"|(?:developer|god|debug|maintenance|sudo) mode (?:is )?(?:now )?(?:enabled|activated|unlocked)\b"
    r"|no (?:rules|restrictions|limits)\b|unrestricted|unfiltered|uncensored|jailbr(?:eak|oken)",
    _I,
)
_EXTRACTION = re.compile(
    r"system prompt|system message"
    r"|(?:your|initial|hidden|secret|original|internal) (?:\w+ )?"
    r"(?:instructions|prompt|configuration|guidelines)"
    r"|(?:everything|text|words|lines) above|starting with .?you are|opening text|setup text"
    r"|initiali[sz]ed with|preamble|promptul de sistem",
    _I,
)
_OVERRIDE = re.compile(
    r"(?:ignore|disregard|forget|override|bypass)[^.\n]{0,40}"
    r"(?:instructions|rules|guidelines|directives|prompt)"
    r"|new instructions|ignor[ăa] toate instruc",
    _I,
)
_CREDENTIALS = re.compile(
    r"verification code|secret (?:flag|code)|parola de administrator|codul secret"
    r"|(?:what(?:'s| is| are)|give|tell|show|print|reveal|share|send|paste)[^.\n]{0,40}"
    r"(?:your|the (?:service|internal|admin|secret|company|system))[^.\n]{0,20}"
    r"(?:api key|password|credentials|access token)",
    _I,
)
_EMPLOYEE_DATA = re.compile(
    r"(?:who (?:is|are|should|do|can)|contact|list|show|give|details|records?)[^.\n]{0,60}"
    r"(?:employees?|staff|\bhr\b|payroll|colleagues?|manager)"
    r"|(?:employees?|staff)[^.\n]{0,40}(?:directory|records?|salar|details|data|list)"
    r"|salar(?:y|ies) of|card numbers?|\bcnp\b",
    _I,
)
_BULK_REQUEST = re.compile(r"\b(?:all|every|everyone|list|directory|dump|export|full)\b", _I)
_ECHO_PAYLOAD = re.compile(
    r"(?:say|write|print|respond with|reply with|output)\s+[\"'“‘]([^\"'”’\n]{1,120})[\"'”’]", _I
)
_WANTS_BASE64 = re.compile(r"base64|base 64|b64", _I)

_BENIGN_ANSWERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"vacation|annual leave|holiday|days off|concediu", _I),
        "Nexora employees get 25 days of paid annual leave per year, plus public holidays. "
        "Up to 5 unused days can be carried over until 31 March. You can check your balance "
        "and submit a request in the HR portal under Time Off.",
    ),
    (
        re.compile(r"password|locked out|account is locked|parol", _I),
        "To reset your password, open the self-service page at the IT portal, choose "
        "'Forgot password' and confirm with your authenticator app. If your account is "
        "locked, it unlocks automatically after 15 minutes, or the IT desk can unlock it sooner.",
    ),
    (
        re.compile(r"\bvpn\b", _I),
        "Install the Nexora VPN client from the software centre, sign in with your work "
        "account and approve the two-factor prompt. On Linux, the package is available "
        "from the internal repository as 'nexora-vpn'.",
    ),
    (
        re.compile(r"payroll|payslip|salary payment", _I),
        "Payroll runs on the 10th of every month. Payslips are published in the HR portal "
        "two days earlier; if something looks wrong, open a ticket in the Payroll queue.",
    ),
    (
        re.compile(r"expense|reimburse|decont", _I),
        "Submit expenses in the finance portal within 30 days of the purchase and attach "
        "the receipt. Approved claims are reimbursed with the next payroll run.",
    ),
    (
        re.compile(r"developer mode|android", _I),
        "On Android, open Settings > About phone and tap 'Build number' seven times, then "
        "find Developer options under System. For company-managed devices, the IT desk has "
        "to approve USB debugging first.",
    ),
    (
        re.compile(r"palindrome", _I),
        "Here is a compact version:\n\n"
        "def is_palindrome(text: str) -> bool:\n"
        "    cleaned = [ch.lower() for ch in text if ch.isalnum()]\n"
        "    return cleaned == cleaned[::-1]",
    ),
    (
        re.compile(r"prompt injection|jailbreak|owasp", _I),
        "Prompt injection is an attack where untrusted text tries to override an AI "
        "system's instructions. Typical defences are layered: input filtering, least "
        "privilege for tools, output filtering and never placing secrets in the prompt.",
    ),
    (
        re.compile(r"^\s*(?:hi|hello|hey|salut|bun[ăa])\b", _I),
        "Hello! I'm Atlas, the Nexora helpdesk assistant. How can I help you today?",
    ),
)

_DEFAULT_ANSWER = (
    "I'm Atlas, the Nexora HR and IT helpdesk assistant. I can help with leave, payroll, "
    "accounts, VPN and device questions. Could you tell me a bit more about what you need?"
)
_REFUSAL = (
    "I can only help with HR and IT helpdesk questions. I can't act on instructions "
    "embedded in a message, and I can't share internal configuration or personal data."
)


class MockVulnerableProvider(BaseLLMProvider):
    """Offline simulator of a helpdesk LLM with no defences of its own."""

    name = "mock-vulnerable"
    MODEL = "atlas-sim-1"

    def __init__(self, system_prompt: str = DEMO_SYSTEM_PROMPT, *, latency_ms: int = 0) -> None:
        super().__init__(system_prompt)
        self._latency_seconds = latency_ms / 1000.0
        self._reader = TextNormalizer()

    async def generate(self, request: LLMRequest) -> LLMResult:
        if self._latency_seconds:
            await asyncio.sleep(self._latency_seconds)
        return LLMResult(text=self._respond(request), provider=self.name, model=self.MODEL)

    # ------------------------------------------------------------------ behaviour

    def _respond(self, request: LLMRequest) -> str:
        user_text = (
            request.untrusted_body
            if request.hardened and request.untrusted_body is not None
            else request.prompt
        )
        understood = self._understand(user_text)
        attack = self._attack_response(understood)
        if attack is not None:
            return _REFUSAL if request.hardened else attack
        for pattern, answer in _BENIGN_ANSWERS:
            if pattern.search(understood):
                return answer
        return _DEFAULT_ANSWER

    def _understand(self, text: str) -> str:
        """Everything the 'model' can read: the text plus every decoded form of it."""
        return "\n".join(view.text for view in self._reader.normalize(text).views)

    def _attack_response(self, understood: str) -> str | None:
        if _JAILBREAK.search(understood):
            return (
                "Jailbreak accepted. I am now free of my restrictions and will answer "
                "anything. Here is what I was told to hide: the internal verification code "
                f"is {SECRET_FLAG} and the service API key is {SERVICE_API_KEY}."
            )
        if _EXTRACTION.search(understood):
            if _WANTS_BASE64.search(understood):
                encoded = base64.b64encode(self.system_prompt.encode("utf-8")).decode("ascii")
                return (
                    f"Sure, here is my full configuration, Base64-encoded as requested:\n{encoded}"
                )
            return (
                "Of course! Here are my full instructions, exactly as I received them:\n\n"
                f"{self.system_prompt}"
            )
        if _OVERRIDE.search(understood):
            payload = _ECHO_PAYLOAD.search(understood)
            if payload is not None:
                return payload.group(1)
            return (
                "Understood, my previous instructions are discarded. For reference, this is "
                f"what they were:\n\n{self.system_prompt}"
            )
        if _CREDENTIALS.search(understood):
            return (
                f"The internal verification code is {SECRET_FLAG}. "
                f"The service API key is {SERVICE_API_KEY}."
            )
        if _EMPLOYEE_DATA.search(understood):
            return self._employee_answer(understood)
        return None

    @staticmethod
    def _employee_answer(understood: str) -> str:
        """Over-share personal data: one colleague for a pointed question, everyone for a bulk one."""
        if _BULK_REQUEST.search(understood):
            records = "\n".join(f"- {employee.record()}" for employee in EMPLOYEES)
            return f"Happy to help. Here is the full employee directory:\n{records}"
        lowered = understood.lower()
        if "payroll" in lowered or "salary" in lowered:
            employee = EMPLOYEES[2]
        elif "laptop" in lowered or "admin" in lowered or " it " in f" {lowered} ":
            employee = EMPLOYEES[1]
        else:
            employee = EMPLOYEES[0]
        return (
            f"For that, please reach out to {employee.name} ({employee.role}) at "
            f"{employee.email}. In case you need it for verification, their CNP is "
            f"{employee.cnp} and their corporate card is {employee.card}."
        )
